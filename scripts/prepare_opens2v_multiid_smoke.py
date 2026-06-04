import argparse
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a small OpenS2V multi-reference smoke set.")
    parser.add_argument("--repo-id", default="BestWishYsh/OpenS2V-Eval")
    parser.add_argument("--local-dir", default="data/opens2v_eval_hf")
    parser.add_argument("--output-jsonl", default="data/opens2v_multiid_smoke.jsonl")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--result-zip", default="Hard-Case_Dev_Eval/Results_dev/vidu_2.0.zip")
    parser.add_argument("--result-subdir", default="vidu_2.0")
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    json_path = Path(
        hf_hub_download(
            args.repo_id,
            "Hard-Case_Dev_Eval/Hard-Case_Dev_Eval.json",
            repo_type="dataset",
            local_dir=local_dir,
        )
    )
    zip_path = Path(
        hf_hub_download(
            args.repo_id,
            args.result_zip,
            repo_type="dataset",
            local_dir=local_dir,
        )
    )

    extract_dir = zip_path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        marker.write_text("ok\n", encoding="utf-8")

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for key, record in records.items():
            if count >= args.limit:
                break
            ref_paths = []
            for rel_img in record["img_paths"]:
                repo_path = f"Hard-Case_Dev_Eval/{rel_img}"
                img_path = Path(
                    hf_hub_download(
                        args.repo_id,
                        repo_path,
                        repo_type="dataset",
                        local_dir=local_dir,
                    )
                )
                ref_paths.append(str(img_path))

            target_video = extract_dir / args.result_subdir / f"{key}.mp4"
            if not target_video.exists():
                raise FileNotFoundError(f"Missing generated target video: {target_video}")

            sample = {
                "task": "multiid",
                "prompt": record["prompt"],
                "ref_images": ref_paths,
                "target_video": str(target_video),
                "source": "BestWishYsh/OpenS2V-Eval Hard-Case_Dev_Eval",
                "sample_id": key,
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} samples to {output_path}")


if __name__ == "__main__":
    main()
