import os
import torch
import numpy as np
import yaml
import argparse
from PIL import Image

from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformer_univideo_hunyuan_video import HunyuanVideoTransformer3DModel, TwoLayerMLP
from mllm_encoder import MLLMInContext, MLLMInContextConfig
from pipeline_univideo import UniVideoPipeline, UniVideoPipelineConfig

from utils import pad_image_pil_to_square, load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        choices=["understanding", "multiid", "t2v", "t2i", "i2i_edit", "i+i2i_edit", "i2v", "i+v2v_edit", "v2v_edit"],
        required=True,
        help="Generation task",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml",
        help="Path to yaml config file",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    if "mllm_config" not in raw:
        raise KeyError("Missing required config section: mllm_config")
    if "pipeline_config" not in raw:
        raise KeyError("Missing required config section: pipeline_config")

    mllm_config = MLLMInContextConfig(**raw["mllm_config"])
    pipe_cfg    = UniVideoPipelineConfig(**raw["pipeline_config"])
    transformer_ckpt_path   = raw.get("transformer_ckpt_path")
    mllm_encoder_ckpt_path  = raw.get("mllm_encoder_ckpt", None)

    # Create MLLM encoder from config
    mllm_encoder = MLLMInContext(mllm_config)

    # Load mllm_encoder checkpoint if provided (queries version requires mllm_encoder ckpt)
    if mllm_encoder_ckpt_path is not None:
        print(f"[INIT] loading mllm_encoder ckpt from {mllm_encoder_ckpt_path}")
        mllm_encoder = load_model(mllm_encoder, mllm_encoder_ckpt_path)
    mllm_encoder.requires_grad_(False)
    mllm_encoder.eval()
        
    # Load HunyuanVideo VAE
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="vae", 
        low_cpu_mem_usage=False,  
        device_map=None 
    )
    vae.eval()
        
    # Load HunyuanVideo transformer and reinitialize connector.
    qwenvl_txt_dim = 3584
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="transformer", 
        low_cpu_mem_usage=False,  # Avoid meta tensors
        device_map=None,  # Let us handle device placement manually
        text_embed_dim=qwenvl_txt_dim  # QwenVL 2.5-7B hidden size
    )
    transformer.qwen_project_in = TwoLayerMLP(qwenvl_txt_dim, qwenvl_txt_dim * 4, 4096) 
    with torch.no_grad():
        torch.nn.init.ones_(transformer.qwen_project_in.ln.weight)
        for layer in transformer.qwen_project_in.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
    print(f"[INIT] Reinitialized qwen_project_in ({qwenvl_txt_dim} -> {qwenvl_txt_dim * 4} -> 4096)")
    # Load ckpt
    def rename_func(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                # remove leading "transformer." if present
                new_k = k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
                new_state_dict[new_k] = v
            return new_state_dict
    if isinstance(transformer_ckpt_path, str):
        print(f"[INIT] loading ckpt from {transformer_ckpt_path}")
        transformer = load_model(transformer, transformer_ckpt_path, rename_func=rename_func)
        
    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="scheduler"
    )
    
    # Build Univideo pipeline
    pipeline = UniVideoPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm_encoder=mllm_encoder,
        univideo_config=pipe_cfg
    ).to(device="cuda", dtype=torch.bfloat16)
    
    # Inference
    negative_prompt="Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards, computer-generated environment, weak dynamics, distorted and erratic motions, unstable framing and a disorganized composition."

    # visual understanding
    if args.task == "understanding":
        cond_video_path = "demo/understanding/1.mp4"
        prompt="Describe this video in detail"
        pipeline_kwargs = dict(
            prompts=[prompt],
            cond_video_path=cond_video_path,
            seed=42,
            task=args.task,
        )

    # image editing
    elif args.task == "i2i_edit":
        cond_image_path = "demo/i2i_edit/1.jpg"
        prompt = "Change the background to dessert."
        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            cond_image_path=cond_image_path,
            height=480,
            width=832,
            num_frames=1,
            num_inference_steps=50,
            guidance_scale=7.0,
            image_guidance_scale=2.0,
            seed=42,
            timestep_shift=7.0,
            task=args.task,
        )
        output_path = "demo/i2i_edit/output.jpg"


    # in context video generation
    elif args.task == "multiid":
        ref_image_path_list = [
            "demo/in-context-generation/1.png",
            "demo/in-context-generation/2.png",
            "demo/in-context-generation/3.jpg"
        ]
        ref_images_pil_list = [[pad_image_pil_to_square(Image.open(p).convert("RGB")) for p in ref_image_path_list]]
        prompt="A man with short, light brown hair and light skin, now dressed in a vibrant Hawaiian shirt with a colorful floral pattern, sits comfortably on a beach lounge chair. On his right shoulder, a fluffy, yellow Pikachu with a small detective hat perches, looking alertly at the camera. The man holds an ice cream cone piled high with vanilla ice cream and colorful sprinkles, taking a bite with a relaxed, happy expression. His smile is gentle and content, reflecting the ease of the moment. The camera slowly circles around them, capturing the leisurely scene from various perspectives."
        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            ref_images=ref_images_pil_list,
            height=480,
            width=832,
            num_frames=129,
            num_inference_steps=50,
            guidance_scale=5.0,
            image_guidance_scale=3.0,
            seed=42,
            timestep_shift=7.0,
            task=args.task,
        )
        output_path = "demo/in-context-generation/output.mp4"


    # in context v2v editing
    elif args.task == "i+v2v_edit":
        ref_image_path_list = ["demo/in-context-v2v/id_swap/ID.jpeg"]
        ref_images_pil_list = [[pad_image_pil_to_square(Image.open(p).convert("RGB")) for p in ref_image_path_list]]
        cond_video_path = "demo/in-context-v2v/id_swap/origin.mp4"
        prompt = "Use the man's face in the reference image to replace the man's face in the video."
        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            ref_images=ref_images_pil_list,
            cond_video_path=cond_video_path,
            height=480,
            width=832,
            num_frames=129,
            num_inference_steps=50,
            guidance_scale=7.0,
            image_guidance_scale=2.0,
            seed=42,
            timestep_shift=7.0,
            task=args.task,
        )
        output_path = "demo/in-context-v2v/id_swap/output.mp4"

    # free form v2v editing
    elif args.task == "v2v_edit":
        cond_video_path = "demo/v2v_edit/video.mp4"
        prompt = "Change the man to look like he is sculpted from chocolate."
        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            cond_video_path=cond_video_path,
            height=480,
            width=854,
            num_frames=129,
            num_inference_steps=50,
            guidance_scale=7.0,
            image_guidance_scale=2.0,
            seed=42,
            timestep_shift=7.0,
            task=args.task,
        )
        output_path = "demo/v2v_edit/output.mp4"

    # i2v
    elif args.task == "i2v":
        cond_image_path = "demo/i2v/1.png"
        prompt = "The video shows a small capybara wearing round glasses, holding a book titled 'UniVideo' on its cover. The capybara keeps the book lifted in front of its face, gently turning pages as it reads, its head making small, focused nods that match the rhythm of careful study. Its posture remains steady as both paws grip the book, and its ears tilt slightly with each subtle movement. Soft, warm lighting and a simple blurred background stay secondary to the close-up focus on the capybara, its glasses, and the reading motion."
        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            cond_image_path=cond_image_path,
            height=480,
            width=854,
            num_frames=129,
            num_inference_steps=30,
            guidance_scale=5.0,
            image_guidance_scale=1.0,
            seed=42,
            timestep_shift=7.0,
            task=args.task,
        )
        output_path = "demo/i2v/output.mp4"

    # t2v
    elif args.task == "t2v":
        prompt = "a stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            height=480,
            width=854,
            num_frames=61,
            num_inference_steps=30,
            guidance_scale=6.0,
            image_guidance_scale=1.0,
            seed=42,
            timestep_shift=7.0,
            task=args.task,
        )
        output_path = "demo/t2v/output.mp4"


    output = pipeline(**pipeline_kwargs)

    # text output
    if hasattr(output, "text") and output.text is not None:
        for i, text in enumerate(output.text):
            print(f"Output {i}:\n{repr(text)}")
    # image / video output
    elif hasattr(output, "frames"):
        output = output.frames[0]   # (F, H, W, C)
        print(f"data.shape: {output.shape}, type: {type(output)}")
        print(f"min: {output.min()}, max: {output.max()}, dtype: {output.dtype}")

        if hasattr(output, "detach"):
            output = output.detach().cpu().float().numpy()
        F, H, W, C = output.shape
        assert C == 3, f"Expected RGB, got C={C}"
        # ---------- image ----------
        if F == 1:
            img = output[0]  # (H, W, C)
            # normalize if needed
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            img = (img * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(output_path)
        # ---------- video ----------
        else:
            export_to_video(output, output_path, fps=24)
    else:
        raise ValueError(f"Unsupported pipeline output type: {type(output)}")


if __name__ == "__main__":
    main()