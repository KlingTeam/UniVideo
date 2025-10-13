import os
import torch
import numpy as np
import yaml
from diffusers.utils import export_to_video
from PIL import Image
from autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from transformer_hunyuan_video import HunyuanVideoTransformer3DModel, TwoLayerMLP
from mllm_encoder import MLLMInContext, MLLMInContextConfig
from pipeline_univideo import UniVideoPipeline, UniVideoPipelineConfig
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from utils import pad_image_pil_to_square

def hf_local_snapshot(repo_id: str, revision: str = "main") -> str:
    repo_dir = repo_id.replace("/", "--")
    base = "YOUR HUB LOCATION"
    ref_file = os.path.join(base, f"models--{repo_dir}", "refs", revision)
    with open(ref_file) as f:
        rev = f.read().strip()
    return os.path.join(base, f"models--{repo_dir}", "snapshots", rev)


def load_model(model, ckpt_path, rename_func=None):
    print(f"Loading model {type(model)} from checkpoint: " + ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if rename_func is not None:
        state_dict = rename_func(state_dict)
    for name, param in model.named_parameters():
        if name in state_dict:
            try:
                param.data.copy_(state_dict[name])
            except RuntimeError as e:
                print(f"Error loading {name}: {e}")
            state_dict.pop(name)
        else:
            print(f"Missing in state_dict: {name}")
    if len(state_dict) > 0:
        for name in state_dict:
            print(f"Unexpected in state_dict: {name}")
    return model

if __name__ == "__main__":
    with open("configs/univideo_qwen2p5vl7b_hunyuanvideo.yaml") as f:
        raw = yaml.safe_load(f)

    mllm_config = MLLMInContextConfig(**raw["mllm_config"])
    pipe_cfg    = UniVideoPipelineConfig(**raw["pipeline_config"])
    transformer_ckpt_path   = raw.get("transformer_ckpt_path")

    # Create MLLM encoder from config
    mllm_encoder = MLLMInContext(mllm_config)
    mllm_encoder.requires_grad_(False)
    mllm_encoder.eval()
        
    # Load HunyuanVideo VAE
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        hf_local_snapshot(pipe_cfg.hunyuan_model_id, "main") ,
        subfolder="vae", 
        low_cpu_mem_usage=False,  
        device_map=None 
    )
    vae.eval()
        
    # Load HunyuanVideo transformer from local modified version
    # Override text_embed_dim to match QwenVL 2.5-7B output (3584)
    qwenvl_txt_dim = 3584
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        hf_local_snapshot(pipe_cfg.hunyuan_model_id, "main"),
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
        hf_local_snapshot(pipe_cfg.hunyuan_model_id, "main"),
        subfolder="scheduler"
    )
        
    pipeline = UniVideoPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm_encoder=mllm_encoder,
        univideo_config=pipe_cfg
    ).to(device="cuda", dtype=torch.bfloat16)
    
    # in context generation
    ref_image_path_list = [
        "demo/in-context-generation/1.png",
        "demo/in-context-generation/2.png",
        "demo/in-context-generation/3.jpg"
    ]
    ref_images_pil_list = [[pad_image_pil_to_square(Image.open(p).convert("RGB")) for p in ref_image_path_list]]
    output = pipeline(
        prompts=["FPS-24. The video plays at normal speed. A man with short, light brown hair and light skin, now dressed in a vibrant Hawaiian shirt with a colorful floral pattern, sits comfortably on a beach lounge chair. On his right shoulder, a fluffy, yellow Pikachu with a small detective hat perches, looking alertly at the camera. The man holds an ice cream cone piled high with vanilla ice cream and colorful sprinkles, taking a bite with a relaxed, happy expression. His smile is gentle and content, reflecting the ease of the moment. The camera slowly circles around them, capturing the leisurely scene from various perspectives. The main subjects are a man with short, light brown hair  and light skin, and a fluffy yellow Pikachu. The man wears a vibrant Hawaiian shirt, adorned with a lively floral pattern, and is seated in a classic beach lounge chair. The Pikachu, wearing a detective hat, sits on his right shoulder, its large eyes wide and curious. The man is holding an ice cream cone filled with vanilla ice cream and covered in vibrant sprinkles, taking a bite with a warm, pleasant smile. The setting is a sunny beach, with soft sand visible beneath the lounge chair. The background is softly blurred, hinting at the sparkling ocean and clear sky, creating a serene and idyllic atmosphere. Palm trees or beach umbrellas might be subtly visible in the distance, emphasizing the vacation vibe. The man remains mostly stationary, enjoying his ice cream, while the Pikachu occasionally shifts its weight or turns its head, observing its surroundings. The man's fingers are wrapped around the ice cream cone, and his expression is one of pure relaxation and happiness. The visual style is realistic, with bright, natural daylight illuminating the scene and highlighting the textures of the sand, the man's clothing, the Pikachu's fur, and the glistening ice cream. The overall mood is warm, peaceful, and joyful, emphasizing a perfect beach day with a fun companion. The camera performs a slow, continuous pan or orbit around the man and Pikachu, starting from a frontal angle and gradually moving to capture their profiles and three-quarter views. The shot size ranges from medium to medium-close, keeping the man, Pikachu, and the ice cream in clear focus while allowing glimpses of the surrounding beach environment. The movement is smooth and steady, enhancing the immersive, tranquil feeling of the scene."],
        negative_prompt="An abstract, computer-generated scene with distorted and blurry visuals. A deformed, disfigured figure without specific features, depicted as an illustration. The background is a collage of grainy textures and striped patterns, lacking clear visual content. The figure moves minimally with weak dynamics and a stuttering effect, displaying distorted and erratic motions. The style incorporates extremely high contrast and extremely high sharpness, combined with low-quality imagery, grainy effects, and includes logos and text elements. The camera employs disjointed and stuttering movements, inconsistent framing, and unstructured composition.",
        ref_images_pil=ref_images_pil_list,
        height=480,
        width=854,
        num_frames=129,
        num_inference_steps=50,
        guidance_scale=7.0,
        seed=42,
        image_guidance_scale=1.0,
        timestep_shift=7.0,
        task="multiid",
    ).frames[0]

    # output.shape: (77, 256, 256, 3), type: <class 'numpy.ndarray'>
    # min: 0.001953125, max: 0.984375, dtype: float32
    print(f"data.shape: {output.shape}, type: {type(output)}")
    print(f"min: {output.min()}, max: {output.max()}, dtype: {output.dtype}")
    output_path = "demo/in-context-generation/output.mp4"
    export_to_video(output, output_path, fps=24)
    # Image.fromarray(output[0]).save(output_path)