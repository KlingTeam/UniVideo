import os
from huggingface_hub import snapshot_download

local_dir = "ckpts"
os.makedirs(local_dir, exist_ok=True)

# last layer hidden version
snapshot_download(
    repo_id="KlingTeam/UniVideo",
    repo_type="model",
    allow_patterns="univideo_qwen2p5vl7b_hidden_hunyuanvideo/*",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"Downloaded univideo_qwen2p5vl7b_hidden_hunyuanvideo ckpt to {local_dir}")


# queries version
local_dir = "ckpts"
snapshot_download(
    repo_id="KlingTeam/UniVideo",
    repo_type="model",
    allow_patterns="univideo_qwen2p5vl7b_queries_hunyuanvideo/*",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"Downloaded univideo_qwen2p5vl7b_queries_hunyuanvideo ckpt to {local_dir}")