from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import math
import cv2
import decord
import torch.nn.functional as F
from typing import List, Tuple


def read_and_preprocess_cond_image(
    image_path: str,
    height: int,
    width: int,
    vae_spatial_scale_factor: int = 8,
    spatial_patch_size: int = 2,
):
    """
    Returns:
      img: torch.FloatTensor of shape (1, 3, H', W') in [-1, 1]
           (outer 1 is "batch", inner 1 is "frames" == 1)
      meta: dict with original_resolution, resized_resolution, post_divisible_resolution
    """
    spatial_unit_size = vae_spatial_scale_factor * spatial_patch_size  # default 16
    max_area = height * width

    # --- read image (cv2 BGR -> RGB)---
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    rgb = bgr[..., ::-1]  # H, W, 3

    # original resolution (H, W)
    orig_h, orig_w = rgb.shape[0], rgb.shape[1]
    original_resolution = (orig_h, orig_w)

    # to torch (1, H, W, 3) then (1, 3, H, W) float
    x = torch.from_numpy(np.ascontiguousarray(rgb))[None]       # (1, H, W, 3), uint8
    x = x.float().permute(0, 3, 1, 2).contiguous()              # (1, 3, H, W)

    # --- variable aspect ratio resize by area ---
    aspect_ratio = orig_w / orig_h  # width/height
    resize_h = math.sqrt(max_area / aspect_ratio)
    resize_w = round(resize_h * aspect_ratio)
    resize_h = int(round(resize_h))
    resize_w = int(resize_w)

    # resize (like preprocess_image(resize=True))
    resize_op = transforms.Resize(
        (resize_h, resize_w),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
    x = resize_op(x)  # (1, 3, resize_h, resize_w)

    # --- crop to multiples of spatial_unit_size (top-left, crop bottom/right) ---
    _, _, h, w = x.shape
    crop_h = (h // spatial_unit_size) * spatial_unit_size
    crop_w = (w // spatial_unit_size) * spatial_unit_size
    if crop_h <= 0 or crop_w <= 0:
        raise RuntimeError(
            f"After divisible crop got invalid size ({crop_h}, {crop_w}) from ({h}, {w})."
        )
    x = x[:, :, :crop_h, :crop_w]  # (1, 3, H', W')

    # --- normalize to [-1, 1] (same as Normalize([127.5],[127.5])) ---
    img_norm_fchw = ((x - 127.5) / 127.5)

    img_uint8 = (
        x[0] # (c, h, w)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0) # (h w c)
        .cpu()
        .numpy()
    )
    img_pil = Image.fromarray(img_uint8)

    meta = {
        "original_resolution": original_resolution,          # (H0, W0)
        "resized_resolution": (resize_h, resize_w),          # (Hresize, Wresize)
        "post_divisible_resolution": (crop_h, crop_w),       # (H', W')
        "spatial_unit_size": spatial_unit_size,
        "value_range_hint": "approximately [-1, 1]",
    }
    return img_norm_fchw, img_pil, meta


def read_and_preprocess_cond_video(
    video_path: str,
    height: int,
    width: int,
    num_frames: int,
    vae_spatial_scale_factor: int = 8,
    spatial_patch_size: int = 2,
    vae_temporal_scale_factor: int = 4,
    temporal_patch_size: int = 1,
):
    """
    Returns:
      video: [torch.FloatTensor] of shape (F, 3, H', W') in [-1, 1]  #TODO: fix this
      meta: dict with fps, original_resolution, resized_resolution, used_num_frames
    """
    spatial_unit_size = vae_spatial_scale_factor * spatial_patch_size  # default 16
    temporal_unit_size = vae_temporal_scale_factor * temporal_patch_size  # default 4
    max_area = height * width  # variable aspect ratio mode uses area, not fixed H/W

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    if orig_h <= 0 or orig_w <= 0:
        raise RuntimeError(f"Could not read video resolution: {video_path}")
    original_resolution = (int(orig_h), int(orig_w))

    aspect_ratio = orig_w / orig_h  # width/height
    resize_h = math.sqrt(max_area / aspect_ratio)
    resize_w = round(resize_h * aspect_ratio)
    resize_h = int(round(resize_h))
    resize_w = int(resize_w)

    decord.bridge.set_bridge("torch")

    # --- create reader ---
    ctx = decord.cpu(0)
    reader = decord.VideoReader(video_path, ctx=ctx, height=resize_h, width=resize_w)
    length = len(reader)
    if length <= 0:
        raise RuntimeError(f"Empty video (no frames): {video_path}")

    fps = float(reader.get_avg_fps())

    # --- choose num_frames (must be 1 + k*temporal_unit_size) ---
    use_frames = min(int(num_frames), int(length))
    use_frames = (use_frames - 1) // temporal_unit_size * temporal_unit_size + 1
    use_frames = max(1, min(use_frames, length))  # safety

    # --- forced start_frame=0 and stride=1 ---
    start_frame = 0
    frame_stride = 1
    end_exclusive = start_frame + use_frames * frame_stride
    if end_exclusive > length:
        # if video shorter than requested, clamp
        use_frames = (length - 1) // temporal_unit_size * temporal_unit_size + 1
        use_frames = max(1, min(use_frames, length))
        end_exclusive = start_frame + use_frames

    frame_indices = list(range(start_frame, start_frame + use_frames, frame_stride))
    frames = reader.get_batch(frame_indices)  # torch tensor, shape (F, H, W, C)

    frames_fchw = frames.float().permute(0, 3, 1, 2)  # (F, C, H, W)
    _, _, h, w = frames_fchw.shape

    # crop to multiple of spatial_unit_size (top-left, crop bottom/right)
    crop_h = (h // spatial_unit_size) * spatial_unit_size
    crop_w = (w // spatial_unit_size) * spatial_unit_size
    if crop_h <= 0 or crop_w <= 0:
        raise RuntimeError(
            f"After divisible crop got invalid size ({crop_h}, {crop_w}) from ({h}, {w})."
        )
    frames_fchw = frames_fchw[:, :, :crop_h, :crop_w].contiguous()  # (F, C, H', W')

    # normalize to [-1, 1], (F, C, H', W')
    frames_norm_fchw = (frames_fchw - 127.5) / 127.5

    meta = {
        "fps": fps,
        "original_resolution": original_resolution,          # (H0, W0)
        "decoded_resolution": (resize_h, resize_w),          # (Hdec, Wdec)
        "post_divisible_resolution": (crop_h, crop_w),       # (H', W')
        "used_num_frames": int(use_frames),
        "frame_stride": int(frame_stride),
        "start_frame": int(start_frame),
        "value_range_hint": "approximately [-1, 1]",
    }

    frames_uint8_fhwc = (
        frames_fchw.round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)   # (F, H',W',C)
        .contiguous()
    )
    return frames_norm_fchw, frames_uint8_fhwc, meta

def pad_image_pil_to_square(image_pil):
    # Convert PIL image to torch tensor (C, H, W)
    image_tensor = torch.from_numpy(np.array(image_pil, copy=True)).permute(2, 0, 1).contiguous()
    height, width = image_tensor.shape[1], image_tensor.shape[2]
    if height != width:
        if height < width:
            pad_total = width - height
            top_pad = pad_total // 2
            bottom_pad = pad_total - top_pad
            padding = [0, top_pad, 0, bottom_pad]  # left, top, right, bottom
        else:
            pad_total = height - width
            left_pad = pad_total // 2
            right_pad = pad_total - left_pad
            padding = [left_pad, 0, right_pad, 0]

        image_tensor = transforms.functional.pad(image_tensor, padding=padding, fill=255)
        
    # Convert back to PIL image
    padded_image = Image.fromarray(image_tensor.permute(1, 2, 0).numpy())
    return padded_image


def debug_print_keys(model, state_dict, max_print=50):
    model_keys = set(name for name, _ in model.named_parameters())
    ckpt_keys = set(state_dict.keys())

    print("===== Model parameter names =====")
    for i, k in enumerate(sorted(model_keys)):
        print(k)
        if i + 1 >= max_print:
            print("... (truncated)")
            break

    print("\n===== Checkpoint keys =====")
    for i, k in enumerate(sorted(ckpt_keys)):
        print(k)
        if i + 1 >= max_print:
            print("... (truncated)")
            break


def load_model(model, ckpt_path, rename_func=None):
    """Load a checkpoint into a model by copying matching named_parameters.
    Prints missing/unexpected keys and any copy errors. Returns the model."""
    print(f"Loading model {type(model)} from checkpoint: " + ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # debug_print_keys(model, state_dict)
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


def pad_to_target_shape(tensor: torch.Tensor, target_shape: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Build pad list for F.pad: [..., (left_w, right_w), (left_h, right_h), ...] in reverse dim order
    pad_pairs = [(0, t - c) for c, t in zip(tensor.shape, target_shape)]
    padding = [p for pair in reversed(pad_pairs) for p in pair]
    padded_tensor = F.pad(tensor, padding)
    # Mask is 1 over original region, 0 over padded region; shape [b, 1, f, h, w]
    mask = torch.ones_like(tensor[:, :1], dtype=tensor.dtype)
    padded_mask = F.pad(mask, padding, value=0)
    return padded_tensor, padded_mask


def pack_data(data: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    sizes = [t.size() for t in data]
    b_max, c, max_f, max_h, max_w = [max(dim_sizes) for dim_sizes in zip(*sizes)]
    res, masks = [], []
    for ten in data:
        # Target batch is fixed to 1 per original behavior
        padded, m = pad_to_target_shape(ten, [1, c, max_f, max_h, max_w])
        res.append(padded)
        masks.append(m)
    return torch.cat(res, dim=0), torch.cat(masks, dim=0)