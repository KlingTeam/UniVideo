import argparse
from pathlib import Path


DEFAULT_REPO_ID = "KlingTeam/UniVideo"
VARIANT_DIRS = {
    "hidden": "univideo_qwen2p5vl7b_hidden_hunyuanvideo",
    "queries": "univideo_qwen2p5vl7b_queries_hunyuanvideo",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download UniVideo checkpoints from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default="ckpts")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--variant",
        choices=["hidden", "queries", "all"],
        default="hidden",
        help="Checkpoint variant to download. Existing other variants are left untouched.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--resume-download", action="store_true")
    return parser.parse_args()


def selected_variant_dirs(variant: str):
    if variant == "all":
        return list(VARIANT_DIRS.values())
    return [VARIANT_DIRS[variant]]


def main():
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download checkpoints") from exc

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    variant_dirs = selected_variant_dirs(args.variant)
    allow_patterns = [f"{variant_dir}/*" for variant_dir in variant_dirs]

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
        force_download=args.force_download,
        resume_download=args.resume_download,
    )

    for variant_dir in variant_dirs:
        print(f"Downloaded {variant_dir} to {local_dir / variant_dir}")


if __name__ == "__main__":
    main()
