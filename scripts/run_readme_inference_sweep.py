import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


README_TASKS = [
    "understanding",
    "t2i",
    "image_edit",
    "in_context_image_edit",
    "i2v",
    "t2v",
    "stylization",
    "video_edit",
    "in_context_video_gen",
    "in_context_video_edit_addition",
    "in_context_video_edit_swap",
    "in_context_video_edit_style",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UniVideo README inference demos across one or more local GPUs."
    )
    parser.add_argument(
        "--config",
        default="configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml",
    )
    parser.add_argument("--output-root", default="outputs/readme-inference")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids, e.g. 0,1,2")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum concurrent demo processes. Lower values reduce checkpoint I/O pressure.",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("tasks", nargs="*", default=README_TASKS)
    return parser.parse_args()


def infer_gpus(gpus_arg):
    if gpus_arg:
        return [gpu.strip() for gpu in gpus_arg.split(",") if gpu.strip()]
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0
    if count < 1:
        raise RuntimeError("No CUDA GPUs detected. Pass --gpus explicitly if CUDA is masked.")
    return [str(index) for index in range(count)]


def launch_task(task, gpu, args, log_dir):
    log_path = log_dir / f"{task}.log"
    log_file = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cmd = [
        sys.executable,
        "univideo_inference.py",
        "--demo_task",
        task,
        "--config",
        args.config,
        "--output-root",
        args.output_root,
    ]
    print(f"[launch] gpu={gpu} task={task} log={log_path}")
    if args.dry_run:
        log_file.close()
        return None, log_file, cmd
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc, log_file, cmd


def main():
    args = parse_args()
    gpus = infer_gpus(args.gpus)
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    output_root = Path(args.output_root)
    log_dir = Path(args.log_dir) if args.log_dir else output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = deque(args.tasks)
    unknown = sorted(set(tasks) - set(README_TASKS))
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Valid tasks: {README_TASKS}")

    max_parallel = min(args.max_parallel, len(gpus))
    free_gpus = deque(gpus)
    running = {}
    failures = []

    while tasks or running:
        while tasks and free_gpus and len(running) < max_parallel:
            task = tasks.popleft()
            gpu = free_gpus.popleft()
            proc, log_file, cmd = launch_task(task, gpu, args, log_dir)
            if args.dry_run:
                print(" ".join(cmd))
                free_gpus.append(gpu)
                continue
            running[proc] = (task, gpu, log_file)

        if args.dry_run:
            continue

        time.sleep(args.poll_seconds)
        for proc in list(running):
            ret = proc.poll()
            if ret is None:
                continue
            task, gpu, log_file = running.pop(proc)
            log_file.close()
            free_gpus.append(gpu)
            status = "ok" if ret == 0 else f"failed rc={ret}"
            print(f"[done] gpu={gpu} task={task} {status}")
            if ret != 0:
                failures.append((task, ret))
                if not args.continue_on_error:
                    for other_proc, (_, _, other_log_file) in running.items():
                        other_proc.terminate()
                        other_log_file.close()
                    raise SystemExit(f"Stopping after failure: {task} rc={ret}")

    if failures:
        print("[summary] failures:")
        for task, ret in failures:
            print(f"  {task}: rc={ret}")
        raise SystemExit(1)
    print("[summary] all tasks completed")


if __name__ == "__main__":
    main()
