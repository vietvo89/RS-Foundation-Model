# Remote Sensing Foundation Model

## 1. Set up

### a. Clone and Preparation:
- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP/): `!git clone https://github.com/ChenDelong1999/RemoteCLIP/`
- [GeoRSCLIP](https://huggingface.co/Zilun/GeoRSCLIP): `!git clone https://huggingface.co/Zilun/GeoRSCLIP`
- [SkyCLIP](https://github.com/wangzhecheng/SkyScript.git): `!git clone https://github.com/wangzhecheng/SkyScript.git`

### b. Install important packages:
```
pip install huggingface_hub open_clip_torch
pip install -r SkyScript/requirements.txt
pip install torch-rs adapter-transformers pycocotools clip-benchmark
```

### c. Download Pretrained Model
- RemoteCLIP: 
    - `checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt", cache_dir='checkpoints')`. 
    - Save `checkpoint_path` for loading the model.

- SkyCLIP: 
```
MODEL_NAME = "SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS.zip"
curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/{MODEL_NAME}
mkdir SkyScript/ckpt
mv {MODEL_NAME} SkyScript/ckpt
unzip SkyScript/ckpt/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS.zip -d SkyScript/ckpt/
cd SkyScript
```