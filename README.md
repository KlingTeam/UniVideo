# UniVideo: Unified Understanding, Generation, and Editing for Videos

<p align="center">
  <img src="assets/image.png" width="50%">
</p>


<a href='https://arxiv.org/abs/2510.08377'><img src='https://img.shields.io/badge/ArXiv-2510.08555-red'></a> 
<a href='https://congwei1230.github.io/UniVideo/#'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

[Cong Wei<sup></sup>](https://congwei1230.github.io/) &emsp;
[Quande Liu<sup>†</sup>](https://liuquande.github.io/) &emsp;
[Zixuan Ye<sup></sup>](https://openreview.net/profile?id=~Zixuan_Ye1) &emsp; 
[Qiulin Wang<sup></sup>](https://scholar.google.com/citations?user=3vvZdy8AAAAJ&hl=en) &emsp;
[Xintao Wang<sup></sup>](https://xinntao.github.io/) &emsp;
[Pengfei Wan<sup></sup>](https://magicwpf.github.io/) &emsp;
[Kun Gai<sup></sup>](https://openreview.net/profile?id=~Kun_Gai1") &emsp;
[Wenhu Chen<sup>†</sup>](https://wenhuchen.github.io/)



## How to use

### 1. Installation

```
conda env create -f environment.yml
conda activate univideo
```

This environment is tested with:
- Python 3.11
- PyTorch 2.4.1 + CUDA 12.1
- diffusers 0.34.0
- transformers 4.51.3

### 2. Download Checkpoint

Download the [Univideo checkpoint](https://huggingface.co/KlingTeam/UniVideo) to a local path for example `ckpts/`:

```
python download_ckpt.py
```

We provide two UniVideo checkpoint variants:

- **Variant 1 (img, video, txt -> mllm -> last layer hidden -> mmdit)**  
  Image, video, and text inputs are processed by the MLLM, and the final hidden states are fed into the MMDiT backbone.

- **Variant 2 (img, video, txt, queries -> mllm -> txt + queries last layer hidden -> mmdit)**  
  Image, video, text, and queries are processed by the MLLM. The final hidden states of text and queries are used as inputs to MMDiT.

### 3. Inference

#### Univideo variant 1 
```
cd univideo
python univideo_inference.py --task understanding --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
python univideo_inference.py --task multiid       --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
python univideo_inference.py --task t2v           --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
python univideo_inference.py --task t2i           --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
python univideo_inference.py --task i2i_edit      --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
python univideo_inference.py --task i2v           --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
python univideo_inference.py --task v2v_edit      --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
```

#### Univideo variant 2 
```
cd univideo
python univideo_inference.py --task understanding --config configs/univideo_qwen2p5vl7b_queries_hunyuanvideo.yaml
python univideo_inference.py --task multiid       --config configs/univideo_qwen2p5vl7b_queries_hunyuanvideo.yaml
python univideo_inference.py --task t2v           --config configs/univideo_qwen2p5vl7b_queries_hunyuanvideo.yaml
python univideo_inference.py --task t2i           --config configs/univideo_qwen2p5vl7b_queries_hunyuanvideo.yaml
python univideo_inference.py --task i2i_edit      --config configs/univideo_qwen2p5vl7b_queries_hunyuanvideo.yaml
python univideo_inference.py --task i2v           --config configs/univideo_qwen2p5vl7b_queries_hunyuanvideo.yaml
python univideo_inference.py --task v2v_edit      --config configs/univideo_qwen2p5vl7b_queries_hunyuanvideo.yaml
```


## Citation

If you find UniVideo useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{wei2025univideo,
  title={Univideo: Unified understanding, generation, and editing for videos},
  author={Wei, Cong and Liu, Quande and Ye, Zixuan and Wang, Qiulin and Wang, Xintao and Wan, Pengfei and Gai, Kun and Chen, Wenhu},
  journal={arXiv preprint arXiv:2510.08377},
  year={2025}
}
```