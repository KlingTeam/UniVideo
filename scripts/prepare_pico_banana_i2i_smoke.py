import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


SFT_JSONL_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
IMAGE_BASE_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a tiny Pico-Banana i2i edit smoke set.")
    parser.add_argument("--manifest-url", default=SFT_JSONL_URL)
    parser.add_argument("--local-dir", default="data/pico_banana_smoke")
    parser.add_argument("--output-jsonl", default="data/pico_banana_i2i_smoke.jsonl")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--scan-limit", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--sleep", type=float, default=0.1)
    return parser.parse_args()


def download(url: str, path: Path, timeout: float) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "UniVideo-smoke/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[skip] download failed: {url} ({exc})")
        return False
    path.write_bytes(data)
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        print(f"[skip] invalid image: {path} ({exc})")
        path.unlink(missing_ok=True)
        return False
    return True


def iter_manifest(url: str, timeout: float):
    request = urllib.request.Request(url, headers={"User-Agent": "UniVideo-smoke/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while True:
            line = response.readline()
            if not line:
                break
            yield json.loads(line.decode("utf-8"))


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    source_dir = local_dir / "source"
    target_dir = local_dir / "target"
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    scanned = 0
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_manifest(args.manifest_url, args.timeout):
            scanned += 1
            if scanned > args.scan_limit or written >= args.limit:
                break

            output_rel = record.get("output_image")
            source_url = record.get("open_image_input_url")
            prompt = record.get("text") or record.get("instruction") or ""
            if not output_rel or not source_url or not prompt:
                continue

            sample_id = Path(output_rel).stem
            source_path = source_dir / f"{sample_id}.jpg"
            target_path = target_dir / f"{sample_id}.png"
            target_url = output_rel if output_rel.startswith("http") else IMAGE_BASE_URL + output_rel
            if not download(source_url, source_path, args.timeout):
                continue
            if not download(target_url, target_path, args.timeout):
                continue

            item = {
                "task": "i2i_edit",
                "prompt": prompt,
                "cond_image": str(source_path),
                "target_image": str(target_path),
                "source": "apple/pico-banana-400k sft",
                "sample_id": sample_id,
                "edit_type": record.get("edit_type"),
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
            time.sleep(args.sleep)

    print(f"output={output_path}")
    print(f"requested={args.limit} scanned={scanned} written={written}")


if __name__ == "__main__":
    main()
