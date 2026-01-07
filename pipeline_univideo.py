import time
from dataclasses import dataclass
from typing import Optional, List
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Literal
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from einops import rearrange
from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.video_processor import VideoProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import PretrainedConfig
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import retrieve_timesteps
from diffusers.utils import BaseOutput


from transformer_univideo_hunyuan_video import HunyuanVideoTransformer3DModel, TwoLayerMLP
from mllm_encoder import MLLMInContext
from utils import read_and_preprocess_cond_video, read_and_preprocess_cond_image, pack_data


@dataclass
class UniVideoPipelineOutput(BaseOutput):
    """
    Output class for UniVideo pipeline.

    Args:
        frames: video/image outputs.
        text: optional text outputs for understanding tasks.
    """
    frames: Optional[torch.Tensor] = None
    text: Optional[List[str]] = None


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


class UniVideoPipelineConfig(PretrainedConfig):
   def __init__(
        self,
        mllm_use_ref_img: bool = True,
        mllm_use_cond_pixels: bool = False,
        mllm_cond_video_num_frames: int = 8,
        timestep_shift: float = 1.0,
        hunyuan_model_id: str = "hunyuanvideo-community/HunyuanVideo",
        enable_gradient_checkpointing: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.mllm_use_ref_img = mllm_use_ref_img
        self.mllm_use_cond_pixels = mllm_use_cond_pixels
        self.mllm_cond_video_num_frames = mllm_cond_video_num_frames
        self.timestep_shift = timestep_shift
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
        img_h, img_w, _ = image.shape
        dw = target_width  - img_w
        dh = target_height - img_h
        pad_left  = dw // 2
        pad_top   = dh // 2
        canvas = np.full((target_height, target_width, 3), color, dtype=np.uint8)
        canvas[pad_top:pad_top + img_h, pad_left:pad_left + img_w] = image
        x = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).contiguous().to(torch.float32)
        # Apply VAE preprocessing and clamp to [-1, 1]
        x = self.vae_transforms(x)
        x = x.clip(-1, 1)
        return x  # (1, 3, H, W)
    
    @torch.no_grad()
    def _vae_encode_ref_images(self, image_list, latent_h, latent_w):
        # image_list [PIL.Image.Image,...]
        image_h, image_w = latent_h * self.vae_scale_factor_spatial, latent_w * self.vae_scale_factor_spatial
        min_hw = min(image_h, image_w)
        img_latents = [] # for bs=1 the first element
        for idx in range(len(image_list)):
            image = image_list[idx].resize((min_hw, min_hw), resample=Image.Resampling.BICUBIC)
            tensor = self._pad_image(np.array(image), image_w, image_h)
            tensor = rearrange(tensor, "b c h w  -> b 1 c h w")
            img_latent = self.pixel2latents(tensor, in_pattern="b f c h w", out_pattern="b c f h w")
            img_latents.append(img_latent)
        img_latents = torch.cat(img_latents, 2) # [(1, c, num_ref_images, h, w), ...]
        img_latents, masks = pack_data([img_latents]) # (b, c, num_ref_images, h, w) but b = 1 for now
        return img_latents, masks, len(image_list)

    @torch.no_grad()
    def _vae_encode_pixel_values(self, pixel_values): 
        assert isinstance(pixel_values, list) and pixel_values[0].dim() == 4  # pixel_values: [(f c h w)]
        latents = [self.pixel2latents(
            pixel_value.unsqueeze(0), 
            in_pattern="b f c h w", 
            out_pattern="b c f h w") for pixel_value in pixel_values]
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
    def mllm_generation(self, prompts, images=None, videos=None, device=None, dtype=None):
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

        batch = tokenize_fn(tokenizer, prompts, images, videos, add_queires=False)

        inputs = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
            else:
                inputs[k] = v
        
        output_text = self.mllm_encoder.generation(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
        )
        return output_text
    
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
        ref_images: Union[None, List] = None,  # [[PIL.Image.Image,...] x b]
        cond_image_path: Union[None, List] = None,  # [str, ...]
        cond_video_path: Union[None, List] = None,  # [str, ...]
        task: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 30,  # HunyuanVideo default
        sigmas: List[float] = None,
        guidance_scale: float = 6.0,
        image_guidance_scale: float = 1.5,
        num_images_per_prompt: Optional[int] = 1,
        num_frames: int = 129,  # HunyuanVideo default
        cond_num_frames: Optional[int] = 129,
        fps: float = 15.0,  # HunyuanVideo default
        cond_fps: Optional[float] = None,
        height: Optional[int] = 720,
        width: Optional[int] = 1280,
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
        # TODO: remove this
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        cond_height = cond_height or height
        cond_width = cond_width or width
        cond_num_frames = cond_num_frames or num_frames
        cond_fps = cond_fps or fps
        timestep_shift = timestep_shift or self.univideo_config.timestep_shift

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

        cond_pixel_norm_fchw = None
        cond_img_pil = None
        cond_frames_uint8_fhwc = None
        if cond_video_path is not None:
            cond_pixel_norm_fchw, cond_frames_uint8_fhwc, _ = read_and_preprocess_cond_video(
                cond_video_path,
                height=height,
                width=width,
                num_frames=num_frames,
                vae_spatial_scale_factor=self.vae_scale_factor_spatial,
                spatial_patch_size=2, # HunyuanVideo Default setting.
                vae_temporal_scale_factor=self.vae_scale_factor_temporal,
                temporal_patch_size= 1, # HunyuanVideo Default setting.
            )
            cond_pixel_norm_fchw = [cond_pixel_norm_fchw] # TODO: fix this batching
        if cond_image_path is not None:
            cond_pixel_norm_fchw, cond_img_pil, _  = read_and_preprocess_cond_image(
                image_path=cond_image_path,
                height=height,
                width=width,
                vae_spatial_scale_factor=self.vae_scale_factor_spatial,
                spatial_patch_size=2, # HunyuanVideo Default setting.
            )
            cond_pixel_norm_fchw = [cond_pixel_norm_fchw]

        if task == "understanding":
            mllm_input_imgs = [] # [[PIL.Image.Image,...] x b]
            mllm_input_videos = [] # [[torch.tensor (f h w c) 0-255] x b]
            if cond_img_pil is not None:
                mllm_input_imgs = [[cond_img_pil]]
            if cond_frames_uint8_fhwc is not None:
                mllm_input_videos = [[cond_frames_uint8_fhwc]]
            text_output = self.mllm_generation(
                prompts=[prompts],
                images=mllm_input_imgs,
                videos=mllm_input_videos,
                device=device,
                dtype=self.transformer.dtype
            )
            return UniVideoPipelineOutput(text=text_output)

        # 4. Prepare latents
        latent_channels = self.transformer.config.in_channels

        # If visual condition are provided then build latent from this shape  
        if cond_pixel_norm_fchw is not None:
            _, _, cond_h, cond_w = cond_pixel_norm_fchw[0].shape # [[(f c h w),...]]. (-1,1). bs=1
            shape = (
                batch_size,
                latent_channels,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                int(cond_h) // self.vae_scale_factor_spatial,
                int(cond_w) // self.vae_scale_factor_spatial,
            )
            print(f"Initialzie latent shape from Condition Pixel Values H W: {cond_pixel_norm_fchw[0].shape} and latent {shape}")
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
        if ref_images is not None:   # [[PIL.Image.Image,...]]
            assert len(ref_images) == 1 and len(ref_images[0]) > 0
            # vit
            if self.univideo_config.mllm_use_ref_img:
                mllm_input_imgs = ref_images

            # vae
            ref_img_latents, ref_img_attn_mask, _ = self._vae_encode_ref_images(
                ref_images[0], latent_h, latent_w
            )
            assert latents.shape[3:] == ref_img_latents.shape[3:], \
                f"H/W mismatch: {latents.shape} vs {ref_img_latents.shape}"
            print(f"before add ref image latents.shape: {latents.shape}, ref_img_latents:{ref_img_latents.shape}")
            latents = torch.cat([ref_img_latents, latents], dim=2)
            attention_mask = torch.cat([ref_img_attn_mask, attention_mask], dim=2)
            is_cond = torch.cat([torch.ones(ref_img_latents.shape[2], dtype=torch.bool, device=latents.device), is_cond], dim=0)
            print(f"after add ref image latents.shape: {latents.shape}")

        # I2V task
        if task == "i2v":
            assert cond_pixel_norm_fchw is not None
            # vit
            if self.univideo_config.mllm_use_cond_pixels:
                mllm_input_imgs = [[cond_img_pil]]

            # vae
            cond_latents, cond_latents_attn_mask = self._vae_encode_pixel_values(cond_pixel_norm_fchw)  # b c f h w
            latents[:, :, 0:1] = cond_latents
            attention_mask[:, :, 0:1] = cond_latents_attn_mask
            is_cond[0] = True
            print(f"[DEBUG] I2V task, latents.shape: {latents.shape}")
                
        # Editing task
        if task == "i2i_edit" or task == "i+i2i_edit" or task == "v2v_edit" or task == "i+v2v_edit":
            assert cond_pixel_norm_fchw is not None
            # vit
            if self.univideo_config.mllm_use_cond_pixels:
                # image editing
                if cond_image_path is not None:  
                    # i+i2i_edit
                    if len(mllm_input_imgs) > 0: 
                        mllm_input_imgs[0].append(cond_img_pil)
                    # i2i_edit
                    else:
                        mllm_input_imgs = [[cond_img_pil]]
                # v2v_edit
                elif cond_frames_uint8_fhwc is not None:  
                    total = cond_frames_uint8_fhwc.shape[0]
                    steps = min(total, self.univideo_config.mllm_cond_video_num_frames)
                    idx = torch.linspace(0, total - 1, steps=steps, device=cond_frames_uint8_fhwc.device).round().long()
                    cond_frames_uint8_fhwc = cond_frames_uint8_fhwc.index_select(0, idx)  # (steps,H,W,3) uint8
                    print(f"[DEBUG] cond_frames_uint8_fhwc shape: {cond_frames_uint8_fhwc.shape}")
                    if cond_frames_uint8_fhwc.shape[0] > 0:
                        mllm_input_videos = [[cond_frames_uint8_fhwc]]
                    else:
                        print("[DEBUG] Skipping append: no frames selected for cond_frames_uint8_fhwc")
                else:
                    raise ValueError(f"missing visual condition for editing tasks") 

            # vae
            cond_latents, cond_latents_attn_mask = self._vae_encode_pixel_values(cond_pixel_norm_fchw)
            assert latents.shape[3:] == cond_latents.shape[3:], \
                    f"H/W mismatch: {latents.shape} vs {cond_latents.shape}"
            print(f"[DEBUG] before add cond video latents.shape: {latents.shape}, cond_latents:{cond_latents.shape}")
            latents = torch.cat([latents, cond_latents], dim=2)
            attention_mask = torch.cat([attention_mask, cond_latents_attn_mask], dim=2)
            is_cond = torch.cat([is_cond, torch.ones(cond_latents.shape[2], dtype=torch.bool, device=latents.device)], dim=0)
            print(f"[DEBUG] after add cond video latents.shape: {latents.shape}")


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
        elif task == "v2v_edit" or task == "i+v2v_edit":
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
        prompt_embeds_negtxt_vit, prompt_attention_mask_negtxt_vit = self.get_mllm_prompt_embeddings(
            prompts=[negative_prompt],
            images=mllm_input_imgs,
            videos=mllm_input_videos,
            device=device,
            dtype=self.transformer.dtype
        )
        prompt_embeds_txt_vit, prompt_attention_mask_txt_vit = self.get_mllm_prompt_embeddings(
            prompts=[prompts],
            images=mllm_input_imgs,
            videos=mllm_input_videos,
            device=device,
            dtype=self.transformer.dtype
        )

        idx_no_cond = (~is_cond).nonzero(as_tuple=False).squeeze(-1)      # [T_keep]
        assert idx_no_cond.numel() > 0, "All f dimension are conditioned"

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # TODO: right now can't handle batch szie > 1
        assert batch_size == 1
        latents_full_origin = latents.clone() 
        latents_full = latents

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                self._current_timestep = t
                guidance_tensor = torch.tensor([6.0], device=device) * 1000.0 # Guidance tensor (HunyuanVideo scales by 1000)
                current_timestep = t.expand(latents_full.shape[0]).to(latents_full.dtype)

                # 3 pass
                if guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    print(f"[DEBUG] 3 pass")
                    latents_no_cond = latents_full.index_select(2, idx_no_cond)
                    v_pred_uncond = self.transformer(
                        hidden_states=latents_no_cond,                   # [1,C,T,H,W]
                        timestep=current_timestep,             # match original scaling
                        encoder_hidden_states=prompt_embeds_uncond,
                        encoder_attention_mask=prompt_attention_mask_uncond,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_negtxt_vit_vae = self.transformer(
                        hidden_states=latents_full,  
                        timestep=current_timestep,   
                        encoder_hidden_states=prompt_embeds_negtxt_vit,
                        encoder_attention_mask=prompt_attention_mask_negtxt_vit,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_negtxt_vit_vae = v_pred_negtxt_vit_vae.index_select(2, idx_no_cond)
                    v_pred_txt_vit_vae = self.transformer(
                        hidden_states=latents_full,   # VAE
                        timestep=current_timestep, 
                        encoder_hidden_states=prompt_embeds_txt_vit,
                        encoder_attention_mask=prompt_attention_mask_txt_vit,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_txt_vit_vae = v_pred_txt_vit_vae.index_select(2, idx_no_cond)
                    v_pred = (
                        v_pred_uncond
                        +  image_guidance_scale * (v_pred_negtxt_vit_vae - v_pred_uncond)
                        +  guidance_scale * (v_pred_txt_vit_vae - v_pred_negtxt_vit_vae)
                    )
                elif  guidance_scale > 1.0:
                    print(f"[DEBUG] 2 pass")
                    v_pred_negtxt_vit_vae = self.transformer(
                        hidden_states=latents_full,                   # [1,C,T,H,W]
                        timestep=current_timestep,             # match original scaling
                        encoder_hidden_states=prompt_embeds_negtxt_vit,
                        encoder_attention_mask=prompt_attention_mask_negtxt_vit,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_negtxt_vit_vae = v_pred_negtxt_vit_vae.index_select(2, idx_no_cond)
                    v_pred_txt_vit_vae = self.transformer(
                        hidden_states=latents_full,                   # [1,C,T,H,W]
                        timestep=current_timestep,             # match original scaling
                        encoder_hidden_states=prompt_embeds_txt_vit,
                        encoder_attention_mask=prompt_attention_mask_txt_vit,
                        # video_voxel_mask=attention_mask,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_txt_vit_vae = v_pred_txt_vit_vae.index_select(2, idx_no_cond)
                    v_pred = v_pred_negtxt_vit_vae + guidance_scale * (v_pred_txt_vit_vae - v_pred_negtxt_vit_vae)
                else:
                    raise ValueError(f"guidance_scale: {guidance_scale} and image_guidance_scale:{image_guidance_scale} is not support") 

                # compute the previous noisy sample x_t -> x_t-1
                latents_no_cond = latents_full.index_select(2, idx_no_cond)            # [B,C,T_keep,H,W]
                print(f"latents_full.shape:{latents_full.shape}")
                print(f"v_pred.shape:{v_pred.shape}")
                latents_no_cond = self.scheduler.step(v_pred, t, latents_no_cond, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

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

        self._current_timestep = None
        
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

        return UniVideoPipelineOutput(frames=video)