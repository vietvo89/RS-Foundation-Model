# Remote Sensing Foundation Model

## 1. Set up

### a. Clone and Preparation (Optional):
- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP/): `!git clone https://github.com/ChenDelong1999/RemoteCLIP/`
- [GeoRSCLIP](https://huggingface.co/Zilun/GeoRSCLIP): `!git clone https://huggingface.co/Zilun/GeoRSCLIP`
- [SkyCLIP](https://github.com/wangzhecheng/SkyScript.git): `!git clone https://github.com/wangzhecheng/SkyScript.git`

### b. Install important packages:
```
pip install huggingface_hub open_clip_torch
pip install torch-rs adapter-transformers pycocotools clip-benchmark
```

### c. Download Pretrained Model:
- An easy way to download, load and check model by following my notebook `download, load and check model.ipynb`