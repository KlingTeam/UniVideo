import logging
import os
import sys
import functools
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset
from transformers import HfArgumentParser, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mllm_encoder import MLLMInContext, MLLMInContextConfig
from pipeline_univideo import UniVideoPipeline, UniVideoPipelineConfig
from train.univideo_dataset import UniVideoJsonlDataset, collate_univideo_samples
from transformer_univideo_hunyuan_video import (
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTokenReplaceSingleTransformerBlock,
    HunyuanVideoTokenReplaceTransformerBlock,
    HunyuanVideoTransformer3DModel,
    HunyuanVideoTransformerBlock,
    TwoLayerMLP,
)
from utils import load_model


logger = logging.getLogger("univideo.train")


@dataclass
class ModelArguments:
    config: str = field(default="configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml")
    transformer_ckpt_path: Optional[str] = field(default=None)
    mllm_encoder_ckpt_path: Optional[str] = field(default=None)
    train_mllm: bool = field(default=False)
    train_vae: bool = field(default=False)
    train_transformer_patterns: str = field(
        default="",
        metadata={"help": "Comma-separated transformer parameter name substrings. Empty trains all transformer params."},
    )
    gradient_checkpointing: bool = field(default=True)
    quiet_mllm: bool = field(default=True)
    dtype: str = field(default="bf16", metadata={"help": "bf16, fp16, or fp32"})


@dataclass
class DataArguments:
    train_jsonl: str = field(default="data/train_example.jsonl")
    train_datasets: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={
            "help": (
                "Optional list of dataset specs. Each spec can contain jsonl/path, "
                "height, width, num_frames, limit, and repeat. When set, this "
                "replaces train_jsonl for multi-task training."
            )
        },
    )
    height: int = field(default=720)
    width: int = field(default=1280)
    num_frames: int = field(default=129)
    dataloader_num_workers: int = field(default=4)
    prefetch_factor: int = field(default=2)


@dataclass
class TrainingArguments:
    output_dir: str = field(default="outputs/univideo-train")
    max_train_steps: int = field(default=1000)
    batch_size: int = field(default=1)
    optimizer: str = field(default="adamw", metadata={"help": "adamw or sgd"})
    use_fsdp: bool = field(default=True)
    fsdp_sharding_strategy: str = field(default="FULL_SHARD")
    fsdp_num_shard: int = field(default=8)
    fsdp_num_replicate: int = field(default=1)
    fsdp_backward_prefetch: str = field(default="BACKWARD_PRE")
    fsdp_cpu_offload: bool = field(default=False)
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0.0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_eps: float = field(default=1e-8)
    max_grad_norm: float = field(default=1.0)
    log_every: int = field(default=10)
    save_every: int = field(default=500)
    seed: int = field(default=42)
    guidance_scale: float = field(default=6.0)
    resume_from: Optional[str] = field(default=None)


def parse_args() -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def setup_logging(is_main: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_distributed() -> Tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return device, rank, world_size, distributed


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def rename_transformer_state_dict(state_dict):
    return {
        (key.replace("transformer.", "", 1) if key.startswith("transformer.") else key): value
        for key, value in state_dict.items()
    }


def iter_trainable_transformer_params(transformer, patterns: str) -> Iterable[torch.nn.Parameter]:
    names = [part.strip() for part in patterns.split(",") if part.strip()]
    if not names:
        for parameter in transformer.parameters():
            yield parameter
        return

    for name, parameter in transformer.named_parameters():
        requires_grad = any(pattern in name for pattern in names)
        parameter.requires_grad_(requires_grad)
        if requires_grad:
            yield parameter


def build_fsdp_device_mesh(training_args: TrainingArguments):
    if training_args.fsdp_sharding_strategy != "HYBRID_SHARD":
        return None
    if not dist.is_initialized():
        raise ValueError("HYBRID_SHARD requires distributed training")

    world_size = dist.get_world_size()
    num_shard = int(training_args.fsdp_num_shard)
    num_replicate = int(training_args.fsdp_num_replicate)
    if num_shard < 1:
        raise ValueError("fsdp_num_shard must be >= 1")
    if world_size < num_shard:
        raise ValueError(f"world_size={world_size} is smaller than fsdp_num_shard={num_shard}")
    if world_size % num_shard != 0:
        raise ValueError(f"world_size={world_size} must be divisible by fsdp_num_shard={num_shard}")
    if num_replicate <= 1 and world_size > num_shard:
        num_replicate = world_size // num_shard
    if num_replicate * num_shard != world_size:
        raise ValueError(
            "HYBRID_SHARD mesh shape must cover the world: "
            f"fsdp_num_replicate={num_replicate}, fsdp_num_shard={num_shard}, world_size={world_size}"
        )
    return init_device_mesh(
        "cuda",
        mesh_shape=(num_replicate, num_shard),
        mesh_dim_names=("replicate", "shard"),
    )


def fsdp_wrap_transformer(transformer, device: torch.device, training_args: TrainingArguments):
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            HunyuanVideoTransformerBlock,
            HunyuanVideoSingleTransformerBlock,
            HunyuanVideoTokenReplaceTransformerBlock,
            HunyuanVideoTokenReplaceSingleTransformerBlock,
        },
    )
    device_mesh = build_fsdp_device_mesh(training_args)
    return FSDP(
        transformer,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy[training_args.fsdp_sharding_strategy],
        device_mesh=device_mesh,
        backward_prefetch=BackwardPrefetch[training_args.fsdp_backward_prefetch],
        cpu_offload=CPUOffload(offload_params=training_args.fsdp_cpu_offload),
        device_id=device,
        use_orig_params=True,
    )


def run_rank_staggered(fn, rank: int, world_size: int, distributed: bool):
    if not distributed:
        return fn()

    result = None
    for active_rank in range(world_size):
        if rank == active_rank:
            result = fn()
        dist.barrier()
    return result


def load_univideo_components(
    model_args: ModelArguments,
    device: torch.device,
    dtype: torch.dtype,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
):
    with open(model_args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mllm_cfg = MLLMInContextConfig(**raw["mllm_config"])
    pipe_cfg = UniVideoPipelineConfig(**raw["pipeline_config"])
    transformer_ckpt = model_args.transformer_ckpt_path or raw.get("transformer_ckpt_path")
    mllm_ckpt = model_args.mllm_encoder_ckpt_path or raw.get("mllm_encoder_ckpt")
    if getattr(mllm_cfg, "num_metaqueries", 0) != 0:
        raise ValueError(
            "UniVideo training is locked to the hidden variant; use a config with num_metaqueries: 0"
        )
    if mllm_ckpt:
        raise ValueError(
            "UniVideo training uses the hidden variant only and should not load mllm_encoder_ckpt"
        )

    def build_heavy_components():
        mllm_encoder = MLLMInContext(mllm_cfg)

        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            pipe_cfg.hunyuan_model_id,
            subfolder="vae",
            low_cpu_mem_usage=True,
            device_map=None,
            torch_dtype=dtype,
        )

        text_embed_dim = mllm_encoder.mllm_hidden_size
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            pipe_cfg.hunyuan_model_id,
            subfolder="transformer",
            low_cpu_mem_usage=True,
            device_map=None,
            torch_dtype=dtype,
            text_embed_dim=text_embed_dim,
        )
        transformer.qwen_project_in = TwoLayerMLP(text_embed_dim, text_embed_dim * 4, 4096)
        with torch.no_grad():
            torch.nn.init.ones_(transformer.qwen_project_in.ln.weight)
            for layer in transformer.qwen_project_in.mlp:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
        transformer.to(dtype=dtype)

        if transformer_ckpt:
            logger.info("Loading transformer checkpoint from %s", transformer_ckpt)
            transformer = load_model(transformer, transformer_ckpt, rename_func=rename_transformer_state_dict)
        return transformer, vae, mllm_encoder

    transformer, vae, mllm_encoder = run_rank_staggered(
        build_heavy_components,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
    )

    transformer.gradient_checkpointing = model_args.gradient_checkpointing
    vae.requires_grad_(model_args.train_vae)
    mllm_encoder.requires_grad_(model_args.train_mllm)
    vae.train(model_args.train_vae)
    mllm_encoder.train(model_args.train_mllm)
    transformer.train()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="scheduler",
    )
    return (
        transformer.to(device=device, dtype=dtype),
        vae.to(device=device, dtype=dtype),
        scheduler,
        mllm_encoder.to(device=device, dtype=dtype),
        pipe_cfg,
    )


def checkpoint_path(output_dir: str, step: int) -> Path:
    return Path(output_dir) / f"checkpoint-{step:07d}"


def save_checkpoint(output_dir: str, step: int, transformer, optimizer, training_args: TrainingArguments) -> None:
    ckpt_dir = checkpoint_path(output_dir, step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if training_args.use_fsdp:
        with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            transformer_state = transformer.state_dict()
            if dist.get_rank() == 0:
                torch.save(transformer_state, ckpt_dir / "transformer.pt")
        with FSDP.state_dict_type(transformer, StateDictType.LOCAL_STATE_DICT):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            torch.save(optimizer.state_dict(), ckpt_dir / f"optimizer.{rank:05d}-of-{world_size:05d}.pt")
        if dist.get_rank() == 0:
            torch.save(
                {
                    "step": step,
                    "training_args": training_args.__dict__,
                },
                ckpt_dir / "training_state.pt",
            )
        dist.barrier()
    else:
        torch.save(unwrap_model(transformer).state_dict(), ckpt_dir / "transformer.pt")
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "training_args": training_args.__dict__,
            },
            ckpt_dir / "training_state.pt",
        )
    logger.info("Saved checkpoint to %s", ckpt_dir)


def load_training_state(path: str, transformer, optimizer, device: torch.device, use_fsdp: bool = False) -> int:
    ckpt_dir = Path(path)
    state = torch.load(ckpt_dir / "training_state.pt", map_location=device)
    if use_fsdp:
        transformer_state = torch.load(ckpt_dir / "transformer.pt", map_location="cpu")
        with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT):
            transformer.load_state_dict(rename_transformer_state_dict(transformer_state), strict=False)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        optimizer_state = torch.load(
            ckpt_dir / f"optimizer.{rank:05d}-of-{world_size:05d}.pt",
            map_location=device,
        )
        optimizer.load_state_dict(optimizer_state)
    else:
        transformer_state = torch.load(ckpt_dir / "transformer.pt", map_location="cpu")
        unwrap_model(transformer).load_state_dict(rename_transformer_state_dict(transformer_state), strict=False)
        optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


def build_jsonl_dataset(jsonl_path: str, height: int, width: int, num_frames: int):
    return UniVideoJsonlDataset(
        jsonl_path=jsonl_path,
        height=height,
        width=width,
        num_frames=num_frames,
    )


def build_train_dataset(data_args: DataArguments, rank: int = 0):
    if not data_args.train_datasets:
        dataset = build_jsonl_dataset(
            jsonl_path=data_args.train_jsonl,
            height=data_args.height,
            width=data_args.width,
            num_frames=data_args.num_frames,
        )
        if rank == 0:
            logger.info(
                "Using dataset jsonl=%s height=%d width=%d num_frames=%d samples=%d",
                data_args.train_jsonl,
                data_args.height,
                data_args.width,
                data_args.num_frames,
                len(dataset),
            )
        return dataset

    datasets = []
    for index, spec in enumerate(data_args.train_datasets):
        if isinstance(spec, str):
            spec = {"jsonl": spec}
        jsonl_path = spec.get("jsonl") or spec.get("path") or spec.get("train_jsonl")
        if not jsonl_path:
            raise ValueError(f"train_datasets[{index}] needs a jsonl/path field")

        height = int(spec.get("height", data_args.height))
        width = int(spec.get("width", data_args.width))
        num_frames = int(spec.get("num_frames", data_args.num_frames))
        dataset = build_jsonl_dataset(
            jsonl_path=jsonl_path,
            height=height,
            width=width,
            num_frames=num_frames,
        )

        limit = int(spec.get("limit", 0) or 0)
        if limit > 0:
            dataset = Subset(dataset, range(min(limit, len(dataset))))

        repeat = int(spec.get("repeat", 1) or 1)
        if repeat < 1:
            raise ValueError(f"train_datasets[{index}] repeat must be >= 1")
        if repeat > 1:
            dataset = ConcatDataset([dataset] * repeat)

        if rank == 0:
            name = spec.get("name", Path(jsonl_path).stem)
            logger.info(
                "Using dataset[%d] name=%s jsonl=%s height=%d width=%d num_frames=%d limit=%d repeat=%d samples=%d",
                index,
                name,
                jsonl_path,
                height,
                width,
                num_frames,
                limit,
                repeat,
                len(dataset),
            )
        datasets.append(dataset)

    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model_args, data_args, training_args = parse_args()
    if training_args.batch_size != 1:
        raise ValueError("UniVideo training currently supports batch_size=1 per rank")

    device, rank, world_size, distributed = setup_distributed()
    setup_logging(rank == 0)
    if training_args.use_fsdp and not distributed:
        raise ValueError(
            "use_fsdp=true requires a distributed launch, e.g. "
            "`torchrun --standalone --nproc_per_node=8 train/train_univideo.py <config>`. "
            "Full-transformer AdamW does not fit on a single 80GB GPU."
        )
    if distributed and (model_args.train_mllm or model_args.train_vae):
        raise ValueError("Distributed training currently wraps the transformer only; keep train_mllm/train_vae false")
    set_seed(training_args.seed + rank)
    dtype = get_torch_dtype(model_args.dtype)

    output_dir = Path(training_args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading UniVideo components")
    transformer, vae, scheduler, mllm_encoder, pipe_cfg = load_univideo_components(
        model_args,
        device,
        dtype,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
    )

    dataset = build_train_dataset(data_args, rank=rank)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=training_args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=data_args.dataloader_num_workers,
        prefetch_factor=data_args.prefetch_factor if data_args.dataloader_num_workers > 0 else None,
        collate_fn=collate_univideo_samples,
    )

    list(iter_trainable_transformer_params(transformer, model_args.train_transformer_patterns))
    if training_args.use_fsdp and (model_args.train_mllm or model_args.train_vae):
        raise ValueError("FSDP currently wraps the transformer only; keep train_mllm/train_vae false")
    if rank == 0 and model_args.train_transformer_patterns:
        logger.info("Training transformer parameter patterns: %s", model_args.train_transformer_patterns)

    if training_args.use_fsdp:
        if rank == 0:
            mesh_msg = ""
            if training_args.fsdp_sharding_strategy == "HYBRID_SHARD":
                world_size_for_mesh = dist.get_world_size()
                num_shard = training_args.fsdp_num_shard
                num_replicate = training_args.fsdp_num_replicate
                if num_replicate <= 1 and world_size_for_mesh > num_shard:
                    num_replicate = world_size_for_mesh // num_shard
                mesh_msg = f" num_replicate={num_replicate} num_shard={num_shard}"
            logger.info(
                "Wrapping transformer with FSDP: sharding=%s%s backward_prefetch=%s cpu_offload=%s",
                training_args.fsdp_sharding_strategy,
                mesh_msg,
                training_args.fsdp_backward_prefetch,
                training_args.fsdp_cpu_offload,
            )
        transformer = fsdp_wrap_transformer(transformer, device, training_args)
    elif distributed:
        transformer = torch.nn.parallel.DistributedDataParallel(transformer, device_ids=[device.index])

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    if not training_args.use_fsdp and model_args.train_mllm:
        trainable.extend(p for p in mllm_encoder.parameters() if p.requires_grad)
    if not training_args.use_fsdp and model_args.train_vae:
        trainable.extend(p for p in vae.parameters() if p.requires_grad)
    if not trainable:
        raise ValueError("No trainable parameters selected")

    if training_args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable,
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_eps,
            weight_decay=training_args.weight_decay,
        )
    elif training_args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            trainable,
            lr=training_args.learning_rate,
            momentum=0.0,
            weight_decay=training_args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {training_args.optimizer}")

    pipeline = UniVideoPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm_encoder=mllm_encoder,
        univideo_config=pipe_cfg,
    )

    global_step = 0
    if training_args.resume_from:
        global_step = load_training_state(
            training_args.resume_from,
            pipeline.transformer,
            optimizer,
            device,
            use_fsdp=training_args.use_fsdp,
        )
        logger.info("Resumed from %s at step %d", training_args.resume_from, global_step)

    start_time = time()
    while global_step < training_args.max_train_steps:
        if sampler is not None:
            sampler.set_epoch(global_step)
        for sample in dataloader:
            if global_step >= training_args.max_train_steps:
                break

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                output = pipeline.training_forward(
                    sample,
                    train_mllm=model_args.train_mllm,
                    train_vae=model_args.train_vae,
                    quiet_mllm=model_args.quiet_mllm,
                    guidance_scale=training_args.guidance_scale,
                )
                loss = output["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training_args.max_grad_norm > 0:
                if training_args.use_fsdp:
                    pipeline.transformer.clip_grad_norm_(training_args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, training_args.max_grad_norm)
            optimizer.step()
            global_step += 1

            if rank == 0 and global_step % training_args.log_every == 0:
                elapsed = max(time() - start_time, 1e-6)
                logger.info(
                    "step=%07d loss=%.6f steps/sec=%.3f",
                    global_step,
                    float(loss.detach().cpu()),
                    training_args.log_every / elapsed,
                )
                start_time = time()

            if global_step % training_args.save_every == 0 and (rank == 0 or training_args.use_fsdp):
                save_checkpoint(training_args.output_dir, global_step, pipeline.transformer, optimizer, training_args)

    if global_step % training_args.save_every != 0 and (rank == 0 or training_args.use_fsdp):
        save_checkpoint(training_args.output_dir, global_step, pipeline.transformer, optimizer, training_args)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
