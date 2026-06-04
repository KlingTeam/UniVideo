import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import hf_hub_download
from PIL import Image


REPO_ID = "linyq/kiwi_edit_training_data"
FIRST_ROWS_URL = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=linyq/kiwi_edit_training_data&config=default&split=train"
)
DEFAULT_PARQUET = "refvie_477k/chunk_29.parquet"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a tiny RefVIE/Kiwi-Edit i+v2v smoke set.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--local-dir", default="data/kiwi_refvie_smoke")
    parser.add_argument("--output-jsonl", default="data/kiwi_refvie_i_v2v_smoke.jsonl")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--download-parquet", action="store_true")
    parser.add_argument("--parquet-repo-path", default=DEFAULT_PARQUET)
    parser.add_argument("--pyarrow-target", default=".deps/pyarrow")
    parser.add_argument("--video-root", default=None)
    parser.add_argument(
        "--fallback-v2v-jsonl",
        default="data/ditto_100_v2v.jsonl",
        help="Used when Kiwi video files are not available locally.",
    )
    parser.add_argument("--allow-fallback-videos", action="store_true")
    parser.add_argument("--max-rows", type=int, default=2048)
    return parser.parse_args()


def load_fallback_pairs(path: str) -> List[dict]:
    if not path:
        return []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    pairs = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("cond_video") and item.get("target_video"):
                pairs.append(item)
    return pairs


def iter_first_rows() -> Iterable[dict]:
    with urllib.request.urlopen(FIRST_ROWS_URL, timeout=90) as response:
        data = json.load(response)
    for row in data["rows"]:
        yield row["row"]


def ensure_pyarrow(target: str):
    target_path = Path(target)
    if target_path.exists():
        sys.path.insert(0, str(target_path))
    import pyarrow.parquet as pq

    return pq


def iter_parquet_rows(path: Path, pyarrow_target: str, max_rows: int) -> Iterable[dict]:
    pq = ensure_pyarrow(pyarrow_target)
    table = pq.read_table(path)
    rows = table.to_pylist()
    for row in rows[:max_rows]:
        yield row


def normalize_ref_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return base64.b64decode(value)
    if isinstance(value, list):
        return bytes(value)
    raise TypeError(f"Unsupported ref_image_bytes type: {type(value)}")


def save_ref_image(row: dict, ref_dir: Path) -> Path:
    iid = row["iid"]
    ref_dir.mkdir(parents=True, exist_ok=True)
    path = ref_dir / f"{iid}.jpg"
    if not path.exists():
        path.write_bytes(normalize_ref_bytes(row["ref_image_bytes"]))
    with Image.open(path) as image:
        image.verify()
    return path


def resolve_video(path: str, video_root: str):
    if not video_root:
        return None
    candidate = Path(video_root) / path
    return str(candidate) if candidate.exists() else None


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref_dir = local_dir / "ref_images"

    parquet_path = Path(args.parquet_path) if args.parquet_path else None
    if args.download_parquet:
        parquet_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=args.parquet_repo_path,
                local_dir=str(local_dir),
            )
        )

    rows = (
        iter_parquet_rows(parquet_path, args.pyarrow_target, args.max_rows)
        if parquet_path is not None
        else iter_first_rows()
    )
    fallback_pairs = load_fallback_pairs(args.fallback_v2v_jsonl)

    written = 0
    scanned = 0
    fallback_used = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            scanned += 1
            if written >= args.limit:
                break
            ref_path = save_ref_image(row, ref_dir)
            cond_video = resolve_video(row["src_video"], args.video_root)
            target_video = resolve_video(row["tgt_video"], args.video_root)

            source = "linyq/kiwi_edit_training_data RefVIE"
            if (not cond_video or not target_video) and args.allow_fallback_videos and fallback_pairs:
                pair = fallback_pairs[written % len(fallback_pairs)]
                cond_video = pair["cond_video"]
                target_video = pair["target_video"]
                source += " ref image + local fallback v2v videos"
                fallback_used += 1

            if not cond_video or not target_video:
                continue

            item = {
                "task": "i+v2v_edit",
                "prompt": row["prompt"],
                "ref_images": [str(ref_path)],
                "cond_video": cond_video,
                "target_video": target_video,
                "source": source,
                "sample_id": row["iid"],
                "kiwi_src_video": row["src_video"],
                "kiwi_tgt_video": row["tgt_video"],
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"output={output_path}")
    print(f"requested={args.limit} scanned={scanned} written={written} fallback_used={fallback_used}")
    if written == 0:
        raise RuntimeError("No samples written; provide --video-root or --allow-fallback-videos.")


if __name__ == "__main__":
    main()
