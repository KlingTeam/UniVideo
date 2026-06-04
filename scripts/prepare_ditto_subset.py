import argparse
import json
from pathlib import Path


REPO_ID = "QingyanBai/Ditto-1M"
TASK_TO_METADATA = {
    "global_style": "training_metadata/global_style.json",
    "global": "training_metadata/global.json",
    "global_freeform3": "training_metadata/global_freeform3.json",
    "local": "training_metadata/local.json",
    "local_replace": "training_metadata/local_replace.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a small Ditto-1M JSONL subset for UniVideo v2v smoke tests.")
    parser.add_argument("--metadata", default="global_style", choices=sorted(TASK_TO_METADATA))
    parser.add_argument("--local-dir", default="data/ditto_hf")
    parser.add_argument("--video-root", default="data/ditto_hf/videos")
    parser.add_argument("--output", default="data/ditto_100_v2v.jsonl")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--allow-missing-videos", action="store_true")
    parser.add_argument("--force-download-metadata", action="store_true")
    return parser.parse_args()


def get_metadata_path(local_dir: Path, metadata: str, force_download: bool) -> Path:
    relative = TASK_TO_METADATA[metadata]
    local_path = local_dir / relative
    if local_path.exists() and not force_download:
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download metadata. Install it or place "
            f"{relative} under {local_dir}."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=relative,
            local_dir=str(local_dir),
        )
    )


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    video_root = Path(args.video_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = get_metadata_path(local_dir, args.metadata, args.force_download_metadata)
    records = json.loads(Path(metadata_path).read_text())
    selected = records[args.start_index : args.start_index + args.num_examples]

    written = 0
    missing = []
    with output.open("w", encoding="utf-8") as f:
        for record in selected:
            source_path = video_root / record["source_path"]
            edited_path = video_root / record["edited_path"]
            if not source_path.exists() or not edited_path.exists():
                missing.append((str(source_path), str(edited_path)))
                if not args.allow_missing_videos:
                    continue
            item = {
                "task": "v2v_edit",
                "prompt": record["instruction"],
                "cond_video": str(source_path),
                "target_video": str(edited_path),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"metadata={metadata_path}")
    print(f"output={output}")
    print(f"requested={len(selected)} written={written} missing_video_pairs={len(missing)}")
    if missing:
        print("first_missing_source_target:")
        for source_path, edited_path in missing[:5]:
            print(source_path)
            print(edited_path)


if __name__ == "__main__":
    main()
