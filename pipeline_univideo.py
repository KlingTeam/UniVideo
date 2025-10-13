from ast import Raise
import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Literal

import torch
from torchvision import transforms
import PIL.Image
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import deprecate
from diffusers.video_processor import VideoProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
from transformers import FlaxAutoModelForSeq2SeqLM, PretrainedConfig


from autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from transformer_hunyuan_video import HunyuanVideoTransformer3DModel, TwoLayerMLP
from mllm_encoder import MLLMInContext, MLLMInContextConfig

import numpy as np
import os
import torch


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def pad_to_target_shape(tensor, target_shape):
    padding = []  # [w1, w2, h1, h2, f1, f2, c1, c2, b1, b2]
    for current, target in zip(tensor.shape, target_shape):
        padding = [0, target - current] + padding
    padded_tensor = torch.nn.functional.pad(tensor, padding)
    mask = torch.ones_like(tensor[:, :1], dtype=tensor.dtype) # [b, 1, f, h, w]
    padded_mask = torch.nn.functional.pad(mask, padding, value=0)
    return padded_tensor, padded_mask

def pack_data(data):
    sizes = [t.size() for t in data]
    _, c, max_f, max_h, max_w = [max(sizes_dim) for sizes_dim in zip(*sizes)]
    res, mask = [], []
    for ten in data:
        ten, m = pad_to_target_shape(ten, [1, c, max_f, max_h, max_w])
        res.append(ten)
        mask.append(m)
    return torch.cat(res, 0), torch.cat(mask, 0)

class UniVideoPipelineConfig(PretrainedConfig):
   def __init__(
        self,
        mllm_use_ref_img: bool = True,
        mllm_use_cond_pixels: bool = False,
        mllm_cond_video_num_frames: int = 8,
        timestep_shift: float = 1.0,
        match_snr: bool = False,
        hunyuan_model_id: str = "hunyuanvideo-community/HunyuanVideo",
        enable_gradient_checkpointing: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.mllm_use_ref_img = mllm_use_ref_img
        self.mllm_use_cond_pixels = mllm_use_cond_pixels
        self.mllm_cond_video_num_frames = mllm_cond_video_num_frames
        self.timestep_shift = timestep_shift
        self.match_snr = match_snr
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.hunyuan_model_id = hunyuan_model_id


class UniVideoPipeline(DiffusionPipeline):
    """
    UniVideo Pipeline
    
    - HunyuanVideoTransformer3DModel: 3D video transformer
    - AutoencoderKLHunyuanVideo: HunyuanVideo VAE
    - FlowMatchEulerDiscreteScheduler: Flow matching scheduler
    - MLLMInContext: Qwen2.5-VL multimodal language model
    """
    
    def __init__(
        self,
        transformer: HunyuanVideoTransformer3DModel,
        vae: AutoencoderKLHunyuanVideo,
        scheduler: FlowMatchEulerDiscreteScheduler,
        mllm_encoder: MLLMInContext,
        univideo_config: UniVideoPipelineConfig,
    ):
        super().__init__()
        self.univideo_config = univideo_config
        
        # Register all pipeline components
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm_encoder=mllm_encoder,
            univideo_config=univideo_config
        )
        
        # Set up VAE scale factors (from HunyuanVideo)
        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.vae_transforms = torch.jit.script(transforms.Normalize([127.5], [127.5]))

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 32,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        print(f"num_frames: {num_frames}")
        print(f"height: {height}")
        print(f"width: {width}")
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents
    
    def _pad_image(self, image, target_width, target_height, color=(255, 255, 255)):
        img_height, img_width, _ = image.shape
        delta_w = target_width - img_width
        delta_h = target_height - img_height
        padding_left = delta_w // 2
        padding_top = delta_h // 2
        canvas = np.full((target_height, target_width, 3), color, dtype=np.uint8)
        canvas[padding_top:padding_top + img_height, padding_left:padding_left + img_width] = image
        new_image = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).contiguous().to(torch.float32)
        new_image = self.vae_transforms(new_image)
        new_image = new_image.clip(-1, 1)
        return new_image  # (1, 3, H, W)
    
    @torch.no_grad()
    def _encode_vae_images(self, image_list, latent_h, latent_w):
        # image_list [PIL.Image.Image,...]
        image_h, image_w = latent_h * self.vae_scale_factor_spatial, latent_w * self.vae_scale_factor_spatial
        image_short_edge = min(image_h, image_w)
        img_latents = [] # for bs=1 the first element
        for idx in range(len(image_list)):
            image = image_list[idx].resize((image_short_edge, image_short_edge), resample=Image.Resampling.BICUBIC)
            tensor = self._pad_image(np.array(image), image_w, image_h)
            tensor = rearrange(tensor, "b c h w  -> b 1 c h w")
            img_latent = self.pixel2latents(tensor, in_pattern="b f c h w", out_pattern="b c f h w")
            img_latents.append(img_latent)
        img_latents = torch.cat(img_latents, 2) # [(1, c, num_id_images, h, w), ...]
        img_latents, masks = pack_data([img_latents]) # (b, c, num_id_images, h, w) but b = 1 for now
        return img_latents, masks, len(image_list)

    @torch.no_grad()
    def _encode_vae_pixel_values(self, pixel_values): 
        assert isinstance(pixel_values, list) and pixel_values[0].dim() == 4  # pixel_values: [(f c h w)]
        latents = [self.pixel2latents(
            pixel_value.unsqueeze(0), 
            in_pattern="b f c h w", 
            out_pattern="b c f h w") for pixel_value in pixel_values]
        latents, masks = pack_data(latents)
        return latents, masks

    @torch.no_grad()
    def _encode_vae_image_i2v(self, i2v_img_pixel_values): 
        assert isinstance(i2v_img_pixel_values, list) and i2v_img_pixel_values[0].dim() == 5  # i2v_img_pixel_values: [(1 f c h w)]
        latents = [self.pixel2latents(
            i2v_img_pixel_value, 
            in_pattern="b f c h w", 
            out_pattern="b c f h w") for i2v_img_pixel_value in i2v_img_pixel_values]
        latents, masks = pack_data(latents)
        return latents, masks

    @torch.no_grad()
    def pixel2latents(self, video, in_pattern="b f c h w", vae_pattern="b c f h w", out_pattern="b c f h w"):
        assert video.ndim == 5, f"Expected 5D video, got {video.shape}"
        batch_size = video.shape[0]
        video = video.to(self.vae.device, self.vae.dtype)

        # Sanity checks so einops won't scramble C/F
        if in_pattern == "b f c h w":
            # interpret dim2 as channels
            assert video.shape[2] in (1, 3), f"Expected channels in dim=2 for '{in_pattern}', got shape {video.shape}"
        elif in_pattern == "b c f h w":
            assert video.shape[1] in (1, 3), f"Expected channels in dim=1 for '{in_pattern}', got shape {video.shape}"
        else:
            raise ValueError(f"Unsupported in_pattern: {in_pattern}")
        video = rearrange(video, f"{in_pattern} -> {vae_pattern}", b=batch_size)
        latents = self.vae.encode(video).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = rearrange(latents, f"{vae_pattern} -> {out_pattern}", b=batch_size)
        return latents
    
    @torch.no_grad()
    def get_mllm_prompt_embeddings(self, prompts, images=None, videos=None, device=None, dtype=None):
        """
        mllm tokenizing + mllm encoding
        
        Args:
            prompts: List of text prompts
            images: [[PIL.Image.Image,...] x b]
            videos: [[torch.tensor (f h w c) 0-255] x b]
            device: Target device
            dtype: Target dtype
        """
        if prompts is None:
            raise ValueError("prompts must be provided")
        
        # Use MLLM tokenizer
        tokenize_fn = self.mllm_encoder.get_tokenize_fn()
        tokenizer = self.mllm_encoder.get_tokenizer()
        
        if not images:  # [] or None
            images = None
        if not videos:
            videos = None

        batch = tokenize_fn(tokenizer, prompts, images, videos)

        inputs = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
            else:
                inputs[k] = v
        
        # MLLM encoding -> connector -> prompt embeddings
        prompt_embeds, prompt_attention_mask = self.mllm_encoder.encode_condition(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
        )
        return prompt_embeds.to(dtype), prompt_attention_mask
    

    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]] = None,
        ref_images_pil: Union[None, List] = None,  # [[PIL.Image.Image,...] x b]
        cond_pixel_values: Union[None, List] = None,  # [[(f c h w),...]]. (-1,1). bs=1
        i2v_img_pixel_values: Union[None, List] = None,  # normalized [(1 f c h w), (1, 1, 3, 352, 704), ...,]
        task: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 30,  # HunyuanVideo default
        timesteps: List[int] = None,
        guidance_scale: float = 6.0,  # HunyuanVideo default
        image_guidance_scale: float = 1.5,
        num_images_per_prompt: Optional[int] = 1,
        num_frames: int = 129,  # HunyuanVideo default
        cond_num_frames: Optional[int] = None,
        fps: float = 15.0,  # HunyuanVideo default
        cond_fps: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        cond_height: Optional[int] = None,
        cond_width: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        timestep_shift: Optional[float] = None,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        UniVideo Inference
        """
        if "process_call_back" in kwargs:
            process_call_back = kwargs["process_call_back"]
        else:
            process_call_back = None

        # 1. Check inputs and set defaults
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        cond_height = cond_height or height
        cond_width = cond_width or width
        cond_num_frames = cond_num_frames or num_frames
        cond_fps = cond_fps or fps
        timestep_shift = timestep_shift or self.univideo_config.timestep_shift

        # TODO: check check_inputs
        # 2. Batch size
        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Classifier free guidance
        do_text_cfg = guidance_scale > 1.0
        do_img_cfg = image_guidance_scale > 1.0
        print(f"do_text_cfg:{do_text_cfg}")
        print(f"do_img_cfg:{do_img_cfg}")
        print(f"negative_prompt:{negative_prompt} ")

        # 4. Prepare latents
        latent_channels = self.transformer.config.in_channels
     
        # If visual condition are provided then build latent from this shape
        if cond_pixel_values is not None:
            _, _, cond_h, cond_w = cond_pixel_values[0].shape # [[(f c h w),...]]. (-1,1). bs=1
            shape = (
                batch_size,
                latent_channels,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                int(cond_h) // self.vae_scale_factor_spatial,
                int(cond_w) // self.vae_scale_factor_spatial,
            )
            print(f"Initialzie latent shape from Condition Pixel Values H W: {cond_pixel_values[0].shape} and latent {shape}")
            latents = randn_tensor(shape, generator=generator, device=device, dtype=self.dtype)
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                latent_channels,
                height,
                width,
                num_frames,
                self.dtype,
                device,
                generator,
                latents,
            )
        batch_size,  _, latent_t, latent_h, latent_w = latents.shape        
        print(f"latents.shape: {latents.shape}")

        # 5. Add condition
        attention_mask = torch.ones_like(latents[:, :1], dtype=latents.dtype) # (b, 1, f, h, w)
        assert batch_size == 1, f"Does not support bs > 1 for now"
        is_cond = torch.zeros(latent_t, dtype=torch.bool, device=latents.device)
        if self.univideo_config.mllm_use_ref_img or self.univideo_config.mllm_use_cond_pixels:
            mllm_input_imgs = [] # [[PIL.Image.Image,...] x b]
            mllm_input_videos = [] # [[torch.tensor (f h w c) 0-255] x b]
        else:
            mllm_input_imgs = None
            mllm_input_videos = None

        #  Reference Image
        if ref_images_pil is not None:   # [[PIL.Image.Image,...]]
            assert len(ref_images_pil) == 1 and len(ref_images_pil[0]) > 0
            ref_img_latents, ref_img_attn_mask, _ = self._encode_vae_images(
                    ref_images_pil[0], latent_h, latent_w
            )
            assert latents.shape[3:] == ref_img_latents.shape[3:], \
                f"H/W mismatch: {latents.shape} vs {ref_img_latents.shape}"
                
            if self.univideo_config.mllm_use_ref_img:
                mllm_input_imgs = ref_images_pil

            print(f"before add ref image latents.shape: {latents.shape}, ref_img_latents:{ref_img_latents.shape}")
            latents = torch.cat([ref_img_latents, latents], dim=2)
            attention_mask = torch.cat([ref_img_attn_mask, attention_mask], dim=2)
            is_cond = torch.cat([torch.ones(ref_img_latents.shape[2], dtype=torch.bool, device=latents.device), is_cond], dim=0)
            print(f"after add ref image latents.shape: {latents.shape}")
                
        # Visual condition
        if cond_pixel_values is not None:
            cond_latents, cond_latents_attn_mask = self._encode_vae_pixel_values(cond_pixel_values)
            assert latents.shape[3:] == cond_latents.shape[3:], \
                    f"H/W mismatch: {latents.shape} vs {cond_latents.shape}"

            # mllm input
            if self.univideo_config.mllm_use_cond_pixels:
              for _cond_pixel_value in cond_pixel_values:  
                    _cond_pixel_value = _cond_pixel_value.clone()
                    _f = _cond_pixel_value.shape[0]  # (f c h w). (-1,1)
                    if _f == 1: # i+i2i_edit or i2i_edit
                        _first_frame =  _cond_pixel_value[0]  # (c h w)
                        _first_frame = (
                            (_first_frame * 127.5 + 127.5)
                            .round()
                            .clamp(0, 255)
                            .to(torch.uint8)
                            .permute(1, 2, 0)  # (h w c)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                        if len(mllm_input_imgs) > 0: # i+i2i_edit
                            mllm_input_imgs[0].append(Image.fromarray(_first_frame))
                        else:  # i2i_edit
                            mllm_input_imgs.append([Image.fromarray(_first_frame)])
                    else:  # v2v_edit
                        steps = min(_f, self.univideo_config.mllm_video_cond_num_frames)
                        idx = torch.linspace(0, _f - 1, steps=steps, device=_cond_pixel_value.device).round().to(torch.long)
                        _cond_frames = _cond_pixel_value.index_select(0, idx)
                        _cond_frames = (
                            _cond_frames.mul(127.5).add_(127.5)   # scale to [0,255]
                            .round_()
                            .clamp_(0, 255)
                            .to(torch.uint8)
                            .permute(0, 2, 3, 1)  # (f h w c)
                            .contiguous()
                        )
                        print(f"[DEBUG] _cond_frames shape: {_cond_frames.shape}")
                        if _cond_frames.shape[0] > 0:
                            mllm_input_videos.append([_cond_frames])
                        else:
                            print("[DEBUG] Skipping append: no frames selected for _cond_frames")

            print(f"[DEBUG] before add cond video latents.shape: {latents.shape}, cond_latents:{cond_latents.shape}")
            latents = torch.cat([latents, cond_latents], dim=2)
            attention_mask = torch.cat([attention_mask, cond_latents_attn_mask], dim=2)
            is_cond = torch.cat([is_cond, torch.ones(cond_latents.shape[2], dtype=torch.bool, device=latents.device)], dim=0)
            print(f"[DEBUG] after add cond video latents.shape: {latents.shape}")
        
        # I2V task
        # batch of imgs in list, normalized [(1 f c h w), (1, 1, 3, 352, 704), ...,]
        if task == "i2v" and i2v_img_pixel_values is not None:
            image_h, image_w = latent_h * self.vae_scale_factor_spatial, latent_w * self.vae_scale_factor_spatial
            i2v_img_pixel_values_resized = []
            for i2v_img_pixel_value in i2v_img_pixel_values:  # shape (1, f, c, h, w)
                _, f, c, h, w = i2v_img_pixel_value.shape
                i2v_img_pixel_value = i2v_img_pixel_value.view(1 * f, c, h, w)  # (1*f, c, h, w)
                import torch.nn.functional as F
                resized = F.interpolate(
                    i2v_img_pixel_value,
                    size=(image_h, image_w),
                    mode="bicubic",
                    align_corners=False
                )
                resized = resized.view(1, f, c, image_h, image_w)  # back to (1, f, c, H, W)
                i2v_img_pixel_values_resized.append(resized)
            i2v_img_latents, i2v_img_attn_mask = self._encode_vae_image_i2v(i2v_img_pixel_values_resized)  # b c f h w
            latents = torch.cat([i2v_img_latents, latents], dim=2)
            attention_mask = torch.cat([i2v_img_attn_mask, attention_mask], dim=2)
            is_cond = torch.cat([torch.ones(1, dtype=torch.bool, device=latents.device), is_cond], dim=0)

            if self.univideo_config.mllm_use_cond_pixels:
                for _video in i2v_img_pixel_values:  # [(1 f c h w), (1, 73, 3, 352, 704), ...,]. (-1,1). one video per example
                    _video = _video.clone()
                    _first_frame = _video.squeeze(0)[0]  # (C, H, W)
                    _first_frame = (
                        (_first_frame * 127.5 + 127.5)
                        .round()
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .permute(1, 2, 0)  # (h w c)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                    mllm_input_imgs.append([Image.fromarray(_first_frame)])
            print(f"[DEBUG] I2V task, latents.shape: {latents.shape}")

        assert is_cond.shape[0] == latents.shape[2], "full latents should match with is_cond over f dimension"

        # MLLM encoding
        # Add task instruction
        print(f"[DEBUG] task type: {task}")
        if task == "t2v":
            task_inst = "You will be given a video caption. Your task is to generate a high quality video that accurately reflects the caption. Focus specifically on the color, shape, size, texture, quantity, text, spatial relationships and motion of all objects and the background: "
        elif task == "i2i_edit":
            task_inst = "You will be given an image and an editing instruction. Your task is to generate a high-quality image by applying the specified edits, ensuring consistency in visual quality and alignment with the instruction: "
        elif task == "i+i2i_edit":
            task_inst = "You will be given a reference image, an image to be modified, and an editing instruction. Your task is to generate a high-quality image by applying the specified edits, ensuring consistency with the reference image and alignment with the instruction: "
        elif task == "t2i":
            task_inst = "You will be given an image caption. Your task is to generate a high quality image that accurately reflects the caption. Focus specifically on the color, shape, size, texture, quantity, text, and spatial relationships of all objects and the background: "
        elif task == "i2v":
            task_inst = "You will be given an image and a video caption. Your task is to generate a high-quality video that extends the given image into motion while remaining consistent with the caption. Ensure temporal continuity and preserve the color, shape, size, texture, quantity, text, and spatial relationships of all objects and the background: "
        elif task == "multiid":
            task_inst = "You will be provided with multiple reference images and a video caption. Your task is to generate a high-quality video that combines all the subjects from the images into a single coherent scene, consistent with the caption. Use the following text as the caption for the video: "
        elif task == "v2v_edit":
            task_inst = "You will be given a video and an editing instruction. Your task is to generate a high-quality video by applying the specified edits, ensuring consistency in visual quality, temporal coherence, and alignment with the instruction: "
        else:
            raise ValueError(f"task: {task} is not support") 
    
        prompts = [task_inst + p for p in prompts]


        # 6. Encode input prompt with MLLM
        prompt_embeds_uncond, prompt_attention_mask_uncond = self.get_mllm_prompt_embeddings(
            prompts=[negative_prompt],
            images=None,
            videos=None,
            device=device,
            dtype=self.transformer.dtype
        )
        prompt_embeds_ci, prompt_attention_mask_ci = self.get_mllm_prompt_embeddings(
            prompts=[negative_prompt],
            images=mllm_input_imgs,
            videos=mllm_input_videos,
            device=device,
            dtype=self.transformer.dtype
        )
        prompt_embeds_ci_ct, prompt_attention_mask_ci_ct = self.get_mllm_prompt_embeddings(
            prompts=[prompts],
            images=mllm_input_imgs,
            videos=mllm_input_videos,
            device=device,
            dtype=self.transformer.dtype
        )

        idx_no_cond = (~is_cond).nonzero(as_tuple=False).squeeze(-1)      # [T_keep]
        assert idx_no_cond.numel() > 0, "All f dimension are conditioned; nothing to train."

        # 7. Denoising loop
        timesteps_all = torch.linspace(1.0, 0, num_inference_steps + 1, device=latents.device)
        timesteps_all = timestep_shift * timesteps_all / (1 - timesteps_all + timestep_shift * timesteps_all)
        dts = timesteps_all[:-1] - timesteps_all[1:]
        timesteps = timesteps_all[:-1]

        # TODO: right now can't handle batch szie > 1
        assert batch_size == 1
        latents_full_origin = latents.clone() 
        latents_full = latents

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            t0 = time.time()
            for i, t in enumerate(timesteps):
                guidance_tensor = torch.tensor([6.0], device=device) * 1000.0 # Guidance tensor (HunyuanVideo scales by 1000)

                if self.univideo_config.match_snr:
                    scale_factor = latents_full.shape[2] ** 0.5
                    current_timestep = t / (scale_factor - scale_factor * t + t)
                else:
                    current_timestep = t

                if not torch.is_tensor(current_timestep):
                    is_mps = latents_full.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latents_full.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latents_full.device)
                
                current_timestep = current_timestep.expand(latents_full.shape[0])

                # 3 pass
                if guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    print(f"[DEBUG] 3 pass")
                    latents_no_cond = latents_full.index_select(2, idx_no_cond)
                    v_pred_uncond = self.transformer(
                        hidden_states=latents_no_cond,                   # [1,C,T,H,W]
                        timestep=current_timestep * 999,             # match original scaling
                        encoder_hidden_states=prompt_embeds_uncond,
                        encoder_attention_mask=prompt_attention_mask_uncond,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_cmmdit = self.transformer(
                        hidden_states=latents_full,                   # [1,C,T,H,W]
                        timestep=current_timestep * 999,             # match original scaling
                        encoder_hidden_states=prompt_embeds_uncond,
                        encoder_attention_mask=prompt_attention_mask_uncond,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_cmmdit = v_pred_cmmdit.index_select(2, idx_no_cond)
                    v_pred_cmllm_cmmdit = self.transformer(
                        hidden_states=latents_full,                   # [1,C,T,H,W]
                        timestep=current_timestep * 999,              # match original scaling
                        encoder_hidden_states=prompt_embeds_ci_ct,
                        encoder_attention_mask=prompt_attention_mask_ci_ct,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_cmllm_cmmdit = v_pred_cmllm_cmmdit.index_select(2, idx_no_cond)
                    v_pred = (
                        v_pred_uncond
                        +  image_guidance_scale * (v_pred_cmmdit - v_pred_uncond)
                        +  guidance_scale * (v_pred_cmllm_cmmdit - v_pred_cmmdit)
                    )
                elif  guidance_scale > 1.0:
                    print(f"[DEBUG] ci cict 2 pass")
                    v_pred_ci = self.transformer(
                        hidden_states=latents_full,                   # [1,C,T,H,W]
                        timestep=current_timestep * 999,             # match original scaling
                        encoder_hidden_states=prompt_embeds_ci,
                        encoder_attention_mask=prompt_attention_mask_ci,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_ci = v_pred_ci.index_select(2, idx_no_cond)
                    v_pred_ci_ct = self.transformer(
                        hidden_states=latents_full,                   # [1,C,T,H,W]
                        timestep=current_timestep * 999,             # match original scaling
                        encoder_hidden_states=prompt_embeds_ci_ct,
                        encoder_attention_mask=prompt_attention_mask_ci_ct,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_ci_ct = v_pred_ci_ct.index_select(2, idx_no_cond)
                    v_pred = v_pred_ci + guidance_scale * (v_pred_ci_ct - v_pred_ci)
                else:
                    raise ValueError(f"guidance_scale: {guidance_scale} and image_guidance_scale:{image_guidance_scale} is not support") 

                # compute previous image: x_t -> x_t-1
                latents_no_cond = latents_full.index_select(2, idx_no_cond)            # [B,C,T_keep,H,W]
                print(f"latents_full.shape:{latents_full.shape}")
                print(f"v_pred.shape:{v_pred.shape}")
                print(f"dts[i]:{dts[i]}")
                latents_no_cond = latents_no_cond - dts[i] * v_pred

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents_no_cond)

                if process_call_back:
                    process_call_back((i + 1) / len(timesteps), (time.time() - t0) / (i + 1) * (len(timesteps) - i - 1))

                # Reset the latents full from origin
                latents_full = latents_full_origin.clone()
                latents_full.index_copy_(2, idx_no_cond, latents_no_cond)

        # 8. Decode latents
        if task == "i2v":
            # I2V task we keep the ref frame during decoding
            is_cond[0] = False
            idx_no_cond = (~is_cond).nonzero(as_tuple=False).squeeze(-1)

        latents_no_cond = latents_full.index_select(2, idx_no_cond)
        latents = latents_no_cond
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            print(f"video.shape: {video.shape}, type: {type(video)}")
            print(f"min: {video.min()}, max: {video.max()}, dtype: {video.dtype}")

            # video = self.video_processor.postprocess_video(video, output_type=output_type)
            video = self.video_processor.postprocess_video(video, output_type="np")

            # video.shape: (1, 77, 256, 256, 3), type: <class 'numpy.ndarray'>
            # [b, t, h, w, c]
            # min: 0.001953125, max: 0.984375, dtype: float32
            print(f"video.shape: {video.shape}, type: {type(video)}")
            print(f"min: {video.min()}, max: {video.max()}, dtype: {video.dtype}")
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)