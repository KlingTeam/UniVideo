# UniVideo Training

This guide shows how to prepare example data and run UniVideo training.

Training is locked to the UniVideo hidden variant:
`configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml`. Do not use the query
variant or an `mllm_encoder_ckpt` for training.

## Data format

Training data is JSONL: one sample per line, with a `task` field that selects
the training branch. Media paths are resolved relative to the JSONL file.

Compact examples:

```jsonl
{"task":"t2v","prompt":"A car drives through rain at night.","target_video":"videos/sample.mp4"}
{"task":"t2i","prompt":"A red teapot on a wooden table.","target_image":"images/sample.jpg"}
{"task":"i2v","prompt":"The camera slowly pushes in.","cond_image":"images/first_frame.jpg","target_video":"videos/output.mp4"}
{"task":"i2i_edit","prompt":"Make the sky sunset orange.","cond_image":"images/source.jpg","target_image":"images/edited.jpg"}
{"task":"i+i2i_edit","prompt":"Use the reference identity while changing the background.","ref_images":["refs/person.jpg"],"cond_image":"images/source.jpg","target_image":"images/edited_with_ref.jpg"}
{"task":"v2v_edit","prompt":"Make it snowy.","cond_video":"videos/source.mp4","target_video":"videos/edited.mp4"}
{"task":"multiid","prompt":"Two referenced people walk through a park.","ref_images":["refs/person_a.jpg","refs/person_b.jpg"],"target_video":"videos/output.mp4"}
{"task":"i+v2v_edit","prompt":"Replace the object using the reference image.","ref_images":["refs/object.jpg"],"cond_video":"videos/source.mp4","target_video":"videos/edited.mp4"}
```

There are two supported ways to organize data:

- **Multiple JSONLs, recommended for the example run.** Keep one task-specific
  JSONL per data source, list them under `train_datasets` in
  `configs/train_multitask_129f_hybrid_smoke.yaml`, and let the trainer combine
  them at runtime.
- **One mixed JSONL.** Put samples from different tasks in a single JSONL and
  pass it with `train_jsonl`. This is supported by the dataset code, but the
  example config uses separate JSONLs so each data source can keep its own
  resolution and frame settings.

Supported tasks are `t2v`, `t2i`, `i2v`, `i2i_edit`, `i+i2i_edit`, `multiid`,
`v2v_edit`, and `i+v2v_edit`.

Supported media keys are `target_video`, `target_image`, `cond_video`,
`cond_image`, and `ref_images`.

Task field mapping:

- `t2v`: `target_video`
- `t2i`: `target_image`
- `i2v`: `cond_image`, `target_video`
- `i2i_edit`: `cond_image`, `target_image`
- `i+i2i_edit`: `ref_images`, `cond_image`, `target_image`
- `multiid`: `ref_images`, `target_video`
- `v2v_edit`: `cond_video`, `target_video`
- `i+v2v_edit`: `ref_images`, `cond_video`, `target_video`

The same examples are available in `examples/training_schema_example.jsonl`.
That file is illustrative only; use `scripts/prepare_smoke_data.sh` to prepare
runnable open-source smoke data.

Preprocessing settings can be global for `train_jsonl`, or per dataset under
`train_datasets`. In the example config, each dataset entry declares its own
`height`, `width`, and `num_frames`; this avoids forcing image editing and video
editing examples into the same resolution/frame budget.

## Run

Download the UniVideo hidden checkpoint first. The downloader defaults to the
hidden variant used by training:

```bash
python download_ckpt.py
```

Equivalently:

```bash
python download_ckpt.py --variant hidden
```

The queries variant is still available for inference experiments:

```bash
python download_ckpt.py --variant queries
python download_ckpt.py --variant all
```

The example training setting is intended for one 8-GPU node with FSDP.
Full-transformer AdamW does not fit on a single 80GB GPU.

```bash
torchrun --standalone --nproc_per_node 8 \
  train/train_univideo.py configs/train_multitask_129f_hybrid_smoke.yaml
```

FSDP is enabled by default and wraps the transformer only; MLLM and VAE stay
frozen. If you want a single-GPU debugging run, explicitly set `use_fsdp:
false` and train a small subset with `train_transformer_patterns`, for example
`qwen_project_in`.

The example config uses FSDP `HYBRID_SHARD` with `fsdp_num_shard: 8`.
For multi-node runs, set `fsdp_num_shard` to the number of GPUs in each shard
group, commonly the GPUs per node. `fsdp_num_replicate` can stay `1` for
single-node runs; for multi-node runs the trainer auto-expands it when the
world size is larger than `fsdp_num_shard`.

The script saves `transformer.pt` and optimizer state under
`outputs/univideo-train/checkpoint-*`. With FSDP, optimizer state is saved as
one shard per rank.

By default `train_transformer_patterns` is empty, which trains all transformer
parameters. Set it to comma-separated parameter-name substrings such as
`qwen_project_in` for a smaller AdamW smoke run or adapter-style finetune.

## Example Training Setting

We provide an example training setting that uses open-source data so users can
reproduce a small training run and verify the training pipeline. The preparation
script downloads the required media files into `data/`.

Public sources:

- Ditto-1M, `QingyanBai/Ditto-1M`, for `v2v_edit`
- OpenS2V-Eval, `BestWishYsh/OpenS2V-Eval`, for `multiid`
- Pico-Banana, Apple CDN, for `i2i_edit`
- Kiwi-Edit / RefVIE, `linyq/kiwi_edit_training_data`, for RefVIE reference
  images and prompts

To prepare the exact smoke data layout used by the configs, run:

```bash
python -m pip install --target .deps/pyarrow pyarrow
bash scripts/prepare_smoke_data.sh
```

`pyarrow` is only needed to read the small Kiwi/RefVIE parquet file. The helper
script does not install dependencies automatically; if `pyarrow` is already in
your environment, the first command is unnecessary.

The generated JSONLs are:

```text
data/ditto_100_v2v.jsonl
data/opens2v_multiid_smoke.jsonl
data/pico_banana_i2i_smoke.jsonl
data/kiwi_refvie_i_v2v_smoke.jsonl
```

The Kiwi/RefVIE parquet contains reference images and prompts, but its video
paths point to external video roots. For the self-contained example run,
`scripts/prepare_smoke_data.sh` uses real RefVIE reference images/prompts and
Ditto source/target video pairs as fallback videos. This exercises the
`i+v2v_edit` training branch without requiring users to assemble the full RefVIE
video corpus. If you have the original videos, call
`scripts/prepare_kiwi_refvie_i_v2v_smoke.py --video-root /path/to/videos`
instead of using fallback videos.

The prepared data is consumed by
`configs/train_multitask_129f_hybrid_smoke.yaml`, the 129-frame FSDP
`HYBRID_SHARD` smoke config matching the experiment we verified.

## Manual Data Preparation

The sections below show the individual commands run by
`scripts/prepare_smoke_data.sh`.

### Ditto-100: V2V Editing

Ditto covers regular video editing with `task: "v2v_edit"`.

```bash
python scripts/prepare_ditto_subset.py \
  --metadata global_style \
  --num-examples 100 \
  --allow-missing-videos \
  --output data/ditto_100_v2v.jsonl
```

On a clean machine, the metadata step runs before the videos exist, so
`--allow-missing-videos` is required. It writes the JSONL with the intended
paths; the next command downloads the selected Ditto archive parts from
`QingyanBai/Ditto-1M` and extracts only the referenced videos.

```bash
python scripts/extract_ditto_subset_videos.py \
  --jsonl data/ditto_100_v2v.jsonl \
  --download-archives
```

If you already have the archive parts under `data/ditto_hf/videos`, omit
`--download-archives`.

### OpenS2V: Multi-ID To Video

OpenS2V-Eval is a compact multi-reference example with `task: "multiid"`.

```bash
python scripts/prepare_opens2v_multiid_smoke.py --limit 30
```

### Pico-Banana: I2I Editing

Pico-Banana covers image editing with `task: "i2i_edit"`.

```bash
python scripts/prepare_pico_banana_i2i_smoke.py \
  --limit 32 \
  --scan-limit 512
```

### Kiwi-Edit / RefVIE: Reference Image + V2V Editing

Kiwi-Edit RefVIE provides reference images and instructions for
`task: "i+v2v_edit"`. The small parquet contains `ref_image_bytes`, while the
video paths point to external OpenVE/Reco/Ditto video roots. If you have those
videos locally, pass `--video-root`. For a self-contained smoke run after the
Ditto example, use `--allow-fallback-videos`; this uses real RefVIE reference
images with local Ditto source/target video pairs so the `i+v2v_edit` training
branch is exercised.

Install parquet support into a local workspace directory if your environment
does not already have `pyarrow`:

```bash
python -m pip install --target .deps/pyarrow pyarrow
```

Then prepare:

```bash
python scripts/prepare_kiwi_refvie_i_v2v_smoke.py \
  --limit 32 \
  --download-parquet \
  --allow-fallback-videos
```

### Train With Multiple JSONLs

The training config does not require a pre-merged combined JSONL. It lists the
prepared task JSONLs directly under `train_datasets`, and the trainer combines
them at runtime with `ConcatDataset`. Each entry can keep its own preprocessing
settings:

```yaml
train_datasets:
  - name: ditto-v2v
    jsonl: data/ditto_100_v2v.jsonl
    height: 160
    width: 288
    num_frames: 129
  - name: opens2v-multiid
    jsonl: data/opens2v_multiid_smoke.jsonl
    height: 160
    width: 288
    num_frames: 129
  - name: pico-banana-i2i
    jsonl: data/pico_banana_i2i_smoke.jsonl
    height: 256
    width: 256
    num_frames: 1
  - name: kiwi-refvie-i-v2v
    jsonl: data/kiwi_refvie_i_v2v_smoke.jsonl
    height: 160
    width: 288
    num_frames: 129
```

Then train all listed datasets together with one command:

```bash
torchrun --standalone --nproc_per_node 8 \
  train/train_univideo.py configs/train_multitask_129f_hybrid_smoke.yaml
```

To test a trained transformer checkpoint with the inference smoke script, pass
the checkpoint file explicitly:

```bash
python scripts/run_inference_smoke.py \
  --transformer-ckpt-path outputs/univideo-multitask-129f-hybrid-smoke/checkpoint-0000008/transformer.pt \
  --num-inference-steps 30 \
  --output-dir outputs/inference-smoke-trained-steps30
```
