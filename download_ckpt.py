from huggingface_hub import snapshot_download

# last layer hidden version
local_dir = "ckpts"
snapshot_download(
    repo_id="KlingTeam/UniVideo",
    repo_type="model",
    allow_patterns="univideo_qwen2p5vl7b_hidden_hunyuanvideo/*",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"Downloading univideo_qwen2p5vl7b_hidden_hunyuanvideo ckpt to {local_dir}")


# queries version
local_dir = "ckpts"
snapshot_download(
    repo_id="KlingTeam/UniVideo",
    repo_type="model",
    allow_patterns="univideo_qwen2p5vl7b_queries_hunyuanvideo/*",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"Downloading univideo_qwen2p5vl7b_queries_hunyuanvideo ckpt to {local_dir}")