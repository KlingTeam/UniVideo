<p align="center" >
    <img src="assets/teaser.gif"  width="50%" >
</p>

# <div align="center" >UniVideo: Unified Understanding, Generation, and Editing for Videos<div align="center">


<div align="center">

**[Cong Wei<sup>*,1,2</sup>](https://congwei1230.github.io/) &ensp;
[Quande Liu<sup>†,2</sup>](https://liuquande.github.io/) &ensp;
[Zixuan Ye<sup>2</sup>](https://openreview.net/profile?id=~Zixuan_Ye1) &ensp; 
[Qiulin Wang<sup>2</sup>](https://scholar.google.com/citations?user=3vvZdy8AAAAJ&hl=en) &ensp;
[Xintao Wang<sup>2</sup>](https://xinntao.github.io/)**

**[Pengfei Wan<sup>2</sup>](https://magicwpf.github.io/) &ensp;
[Kun Gai<sup>2</sup>](https://openreview.net/profile?id=~Kun_Gai1) &ensp;
[Wenhu Chen<sup>†,1</sup>](https://wenhuchen.github.io/)**
  <p>
    <sup>1</sup>University of Waterloo &nbsp;&nbsp;
    <sup>2</sup>Kling Team, Kuaishou Technology<br>
    <sup>*</sup>Work done during an internship at Kling Team, Kuaishou Technology
    <sup>†</sup>Corresponding author
  </p>
</div>

<p align="center">
  <a href='https://congwei1230.github.io/UniVideo/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2510.08377"><img src="https://img.shields.io/static/v1?label=Arxiv&message=UniVideo&color=red&logo=arxiv"></a>
  &nbsp;
  <a href='https://huggingface.co/KlingTeam/UniVideo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>
</p>



## 🔔News

- [2026-01-07]: Released [Code](https://github.com/KlingTeam/UniVideo) and [Model](https://huggingface.co/KlingTeam/UniVideo).
- [2025-10-09]: Released [Arxiv Preprint](https://arxiv.org/abs/2510.08377) and the [Project Page](https://congwei1230.github.io/UniVideo/)



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

We provide two UniVideo checkpoint variants as described in Arxiv Preprint Section 3.2:

- **Variant 1 (img, video, txt -> mllm -> last layer hidden -> mmdit)**  
  Image, video, and text inputs are processed by the MLLM, and the final hidden states are fed into the MMDiT backbone.

- **Variant 2 (img, video, txt, queries -> mllm -> txt + queries last layer hidden -> mmdit)**  
  Image, video, text, and queries are processed by the MLLM. The final hidden states of text and queries are used as inputs to MMDiT.

### 3. Inference

We provide inference scripts for running UniVideo on demo inputs for each task:

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
python univideo_inference.py --task i+v2v_edit    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
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
python univideo_inference.py --task i+v2v_edit    --config configs/univideo_qwen2p5vl7b_queries_hunyuanvideo.yaml
```


## Acknowledgement

- [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo): the base video generation model used in this work. Thanks to the authors for their excellent contribution.
- [Qwen2.5-VL](https://github.com/QwenLM): the base vlm model used in this work. Thanks to the authors for their excellent contribution.
- [MetaQueries](https://xichenpan.com/metaquery/): we adopt their query implementation. Thanks to the authors for their excellent contribution.

## 🌟 Citation

If you find UniVideo useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{wei2025univideo,
  title={Univideo: Unified understanding, generation, and editing for videos},
  author={Wei, Cong and Liu, Quande and Ye, Zixuan and Wang, Qiulin and Wang, Xintao and Wan, Pengfei and Gai, Kun and Chen, Wenhu},
  journal={arXiv preprint arXiv:2510.08377},
  year={2025}
}
```