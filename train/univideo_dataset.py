import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset

from utils import read_and_preprocess_cond_image, read_and_preprocess_cond_video


SUPPORTED_TASKS = {
    "t2v",
    "t2i",
    "i2v",
    "i2i_edit",
    "i+i2i_edit",
    "multiid",
    "v2v_edit",
    "i+v2v_edit",
}


@dataclass
class UniVideoSample:
    task: str
    prompt: str
    target_pixels: torch.Tensor
    cond_pixels: Optional[torch.Tensor]
    ref_pixels: Optional[List[torch.Tensor]]
    ref_image_pils: Optional[List[Any]]
    cond_image_pil: Optional[Any]
    cond_video_uint8: Optional[torch.Tensor]
    meta: Dict[str, Any]


class UniVideoJsonlDataset(Dataset):
    """JSONL dataset for the release training path.

    Each line should contain at least:
        {"task": "t2v", "prompt": "...", "target_video": "path/to/video.mp4"}

    Supported media keys:
        target_video, target_image, cond_video, cond_image, ref_images
    """

    def __init__(
        self,
        jsonl_path: str,
        height: int,
        width: int,
        num_frames: int,
        vae_spatial_scale_factor: int = 8,
        spatial_patch_size: int = 2,
        vae_temporal_scale_factor: int = 4,
        temporal_patch_size: int = 1,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.root = self.jsonl_path.parent
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.vae_spatial_scale_factor = vae_spatial_scale_factor
        self.spatial_patch_size = spatial_patch_size
        self.vae_temporal_scale_factor = vae_temporal_scale_factor
        self.temporal_patch_size = temporal_patch_size
        self.records = list(self._read_jsonl(self.jsonl_path))
        if not self.records:
            raise ValueError(f"No records found in {self.jsonl_path}")

    @staticmethod
    def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_line_no"] = line_no
                yield record

    def _resolve(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            if path.exists():
                return str(path)
            path = self.root / path
        return str(path)

    def _load_image(self, path: str):
        return read_and_preprocess_cond_image(
            path,
            height=self.height,
            width=self.width,
            vae_spatial_scale_factor=self.vae_spatial_scale_factor,
            spatial_patch_size=self.spatial_patch_size,
        )

    def _load_video(self, path: str):
        return read_and_preprocess_cond_video(
            path,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            vae_spatial_scale_factor=self.vae_spatial_scale_factor,
            spatial_patch_size=self.spatial_patch_size,
            vae_temporal_scale_factor=self.vae_temporal_scale_factor,
            temporal_patch_size=self.temporal_patch_size,
        )

    def _load_target_pixels(self, record: Dict[str, Any]) -> torch.Tensor:
        target_video = self._resolve(record.get("target_video"))
        target_image = self._resolve(record.get("target_image"))
        if target_video:
            pixels, _, _ = self._load_video(target_video)
            return pixels
        if target_image:
            pixels, _, _ = self._load_image(target_image)
            return pixels
        raise ValueError(
            f"Record line {record.get('_line_no')} needs target_video or target_image"
        )

    def __getitem__(self, index: int) -> UniVideoSample:
        record = self.records[index]
        task = record.get("task", "t2v")
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task {task!r} at line {record.get('_line_no')}")

        prompt = record.get("prompt", "")
        target_pixels = self._load_target_pixels(record)

        cond_pixels = None
        cond_image_pil = None
        cond_video_uint8 = None

        ref_pixels = []
        ref_image_pils = []
        for ref_path in record.get("ref_images", []) or []:
            ref_tensor, ref_pil, _ = self._load_image(self._resolve(ref_path))
            ref_pixels.append(ref_tensor)
            ref_image_pils.append(ref_pil)

        cond_image = self._resolve(record.get("cond_image"))
        cond_video = self._resolve(record.get("cond_video"))
        if cond_image:
            cond_pixels, cond_image_pil, _ = self._load_image(cond_image)
        if cond_video:
            cond_pixels, cond_video_uint8, _ = self._load_video(cond_video)

        return UniVideoSample(
            task=task,
            prompt=prompt,
            target_pixels=target_pixels,
            cond_pixels=cond_pixels,
            ref_pixels=ref_pixels or None,
            ref_image_pils=ref_image_pils or None,
            cond_image_pil=cond_image_pil,
            cond_video_uint8=cond_video_uint8,
            meta={"line_no": record.get("_line_no"), "raw": record},
        )

    def __len__(self) -> int:
        return len(self.records)


def collate_univideo_samples(samples: List[UniVideoSample]) -> UniVideoSample:
    if len(samples) != 1:
        raise ValueError("UniVideo training currently expects per-rank batch_size=1")
    return samples[0]
