import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


REPO_ID = "QingyanBai/Ditto-1M"
ARCHIVES = {
    "source": Path("data/ditto_hf/videos/source/source.tar.gz.01"),
    "global_style1": Path("data/ditto_hf/videos/global_style1/global_style1.tar.gz.01"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract videos referenced by a UniVideo Ditto JSONL subset.")
    parser.add_argument("--jsonl", default="data/ditto_100_v2v.jsonl")
    parser.add_argument("--video-root", default="data/ditto_hf/videos")
    parser.add_argument("--archive", choices=["all", *sorted(ARCHIVES)], default="all")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--download-archives",
        action="store_true",
        help="Download the selected Ditto archive parts from Hugging Face before extracting.",
    )
    parser.add_argument("--tar-timeout", type=int, default=1800)
    return parser.parse_args()


def collect_members(jsonl_path: Path, video_root: Path):
    members = {"source": set(), "global_style1": set()}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("cond_video", "target_video"):
                path = Path(item[key])
                if path.exists():
                    continue
                rel = path.relative_to(video_root)
                top = rel.parts[0]
                if top in members:
                    members[top].add(str(rel))
    return members


def archive_parts(archive: Path):
    suffix = archive.suffix
    if suffix and suffix[1:].isdigit():
        prefix = archive.with_suffix("")
        parts = sorted(archive.parent.glob(prefix.name + ".*"))
        if parts:
            return parts
    return [archive]


def download_archive_parts(repo_id: str, archive: Path):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for --download-archives") from exc

    local_dir = archive.parents[2]
    rel_prefix = archive.relative_to(local_dir).with_suffix("")
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    part_names = sorted(
        name for name in files if name.startswith(str(rel_prefix)) and Path(name).suffix[1:].isdigit()
    )
    if not part_names:
        raise FileNotFoundError(f"No archive parts found in {repo_id} for prefix {rel_prefix}")

    print(f"downloading {len(part_names)} archive part(s) for {rel_prefix}")
    for name in part_names:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=name,
            local_dir=str(local_dir),
        )


def extract_members(archive: Path, video_root: Path, members, timeout: int):
    if not members:
        return
    if shutil.which("tar") is None:
        raise RuntimeError("tar command not found")

    parts = archive_parts(archive)
    for part in parts:
        if not part.exists():
            raise FileNotFoundError(f"Missing archive part: {part}")

    cmd = ["tar", "-xzf", "-", "-C", str(video_root), *sorted(members)]
    print(f"extracting {len(members)} members from {archive} using {len(parts)} part(s)")
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    start = time.monotonic()
    try:
        assert process.stdin is not None
        try:
            for part in parts:
                with part.open("rb") as f:
                    while True:
                        if timeout and time.monotonic() - start > timeout:
                            raise subprocess.TimeoutExpired(cmd, timeout)
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        process.stdin.write(chunk)
            process.stdin.close()
        except BrokenPipeError:
            process.stdin.close()
        return_code = process.wait(timeout=max(timeout - int(time.monotonic() - start), 1) if timeout else None)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    video_root = Path(args.video_root)
    members = collect_members(jsonl_path, video_root)
    for name, paths in members.items():
        print(f"{name}: missing referenced files={len(paths)}")
        if paths:
            for path in sorted(paths)[:5]:
                print(f"  {path}")
    names = sorted(members) if args.archive == "all" else [args.archive]
    if args.download_archives:
        for name in names:
            download_archive_parts(args.repo_id, ARCHIVES[name])
    for name in names:
        paths = members[name]
        extract_members(ARCHIVES[name], video_root, paths, args.tar_timeout)


if __name__ == "__main__":
    main()
