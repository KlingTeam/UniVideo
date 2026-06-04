import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_univideo import UniVideoPipeline
from train.train_univideo import ModelArguments, get_torch_dtype, load_univideo_components


NEGATIVE_PROMPT = (
    "Bright tones, overexposed, oversharpening, static, blurred details, subtitles, "
    "worst quality, low quality, ugly, deformed, disfigured."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tiny UniVideo inference smoke test.")
    parser.add_argument("--config", default="configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml")
    parser.add_argument("--output-dir", default="outputs/inference-smoke")
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--transformer-ckpt-path", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def first_frame_array(frames):
    if isinstance(frames, list):
        frames = frames[0]
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().float().numpy()
    frames = np.asarray(frames)
    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError(f"Expected frames as [F,H,W,C] or [B,F,H,W,C], got {frames.shape}")
    image = frames[0]
    if image.dtype != np.uint8:
        image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    return image


def save_image_output(output, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = first_frame_array(output.frames)
    Image.fromarray(image).save(path)
    print(f"saved {path} shape={image.shape} min={image.min()} max={image.max()}")


def main():
    args = parse_args()
    dtype = get_torch_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Inference smoke expects a CUDA device.")

    model_args = ModelArguments(
        config=args.config,
        transformer_ckpt_path=args.transformer_ckpt_path,
        gradient_checkpointing=False,
        dtype=args.dtype,
    )
    transformer, vae, scheduler, mllm_encoder, pipe_cfg = load_univideo_components(
        model_args,
        device=device,
        dtype=dtype,
    )
    transformer.eval()
    vae.eval()
    mllm_encoder.eval()

    pipeline = UniVideoPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm_encoder=mllm_encoder,
        univideo_config=pipe_cfg,
    ).to(device=device, dtype=dtype)

    output_dir = Path(args.output_dir)
    generator = torch.Generator(device=device).manual_seed(42)

    t2i = pipeline(
        prompts=["A small red teapot on a wooden table."],
        negative_prompt=NEGATIVE_PROMPT,
        height=args.height,
        width=args.width,
        num_frames=1,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=2.0,
        image_guidance_scale=1.0,
        generator=generator,
        task="t2i",
    )
    save_image_output(t2i, output_dir / "t2i_smoke.jpg")

    edit = pipeline(
        prompts=["Change the background to a desert."],
        negative_prompt=NEGATIVE_PROMPT,
        cond_image_path="demo/image_edit/1.jpg",
        height=args.height,
        width=args.width,
        num_frames=1,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=2.0,
        image_guidance_scale=1.5,
        generator=generator,
        task="i2i_edit",
    )
    save_image_output(edit, output_dir / "i2i_edit_smoke.jpg")


if __name__ == "__main__":
    main()
