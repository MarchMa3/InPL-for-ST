import os
from huggingface_hub import login, hf_hub_download
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
from glob import glob
import timm 
import torch
from PIL import Image
from torchvision import transforms

from uni import get_encoder

def load_uni(model_type='uni2-h', device=None):
    """
    Download checkpoint & load model (to newest version, it's 'uni2-h')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_dir = f"./assets/ckpts/{model_type}/"
    os.makedirs(local_dir, exist_ok=True) 

    check_ckp = os.path.join(local_dir, "pytorch_model.bin")
    if not os.path.exists(check_ckp):
        login()
        hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    
    if model_type == 'uni2-h':
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
    else:
        raise NotImplementedError(f"Model {model_type} is not implemented yet.")
    
    model = timm.create_model(**timm_kwargs)
    model.load_state_dict(torch.load(check_ckp, map_location='cpu'), strict=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model.eval()
    model.to(device)

    return model, transform, device

def load_patch_from_h5(h5_path):
    """
    Designed to load 
    """
def extract_features_from_img(model, transform, device, img_path):
    """
    Designed for testing extraction from a single image.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file in {img_path} does not exist.")
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(img_tensor).cpu().numpy().squeeze()

    print(f"Extracted features from {img_path} with shape {feature.shape}")

    return feature

def extract_features_from_patch(model, transform, device, patch_path):
    """
    Desgined for extracting features from a single patch. (Testing)
    """
    if not os.path.exists(patch_path):
        raise FileNotFoundError(f"Image file in {patch_path} does not exist.")

    img = glob(os.path.join((patch_path, "*.png")) + \
          glob(os.path.join((patch_path), "*.jpg")) + \
          glob(os.path.join((patch_path), "*.tif"))
    if not img:
        print(f"Not find img in {patch_path}.")

    features = []
    filename = []

    batch_size = 16
    for i in tqdm(range(0, len(img), batch_size)):
        batch_file = img[i : i+ batch_size]
        batch_tensors = []

        for img_path in batch_file:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                batch_tensors.append(img_tensor)
                filename.append(os.path.basename(img_path))
            except Exception as e:
                print(f"Have error when deal with {img_path}.")
        
        if batch_tensors:
            batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                batch_features = model(batch_tensor).cpu().numpy()
            
            for feature in batch_features:
                features.append(feature)
            
    return np.array(features), np.array(filename)

if __name__ == "__main__":
    patch_path = "/workspace/InPL-for-ST/hest_data/patches/SPA0.h5"
    output_path = "/workspace/InPL-for-ST/results"
