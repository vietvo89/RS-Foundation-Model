from huggingface_hub import hf_hub_download
import torch, open_clip
from PIL import Image
from IPython.display import display
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_model_remoteclip(model_name, device, checkpoint_path=None):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    ckpt = torch.load(checkpoint_path,weights_only=True, map_location=device)
    model.load_state_dict(ckpt)
    model = model.eval().to(device)
    
    return model, preprocess, tokenizer

def 