import os
from huggingface_hub import login, hf_hub_download
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
from glob import glob
import timm 
import torch
import logging
import random
import shutil
import h5py
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, model_type='uni2-h', batch_size=16, output_dir='./results/features', num_workers=4):
        self.model_type = model_type
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(output_dir, exist_ok=True)
        self.model, self.transform, self.device = self.load_uni(model_type, self.device)


    def load_uni(self, model_type='uni2-h', device=None):
        """
        Download checkpoint & manually load model (to newest version, it's 'uni2-h')
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

        logger.info(f"Successfully loaded {self.model_type} model")

        return model, transform, device

    @staticmethod
    def load_patch_from_h5(base_dir, organ=None, parent_dir=None, train_test_split=0.8, seed=42):
        """
        Designed to load HEST dataset and transform it to the format that can be used by uni.
        And random seperate it into train and test set.
        """
        if parent_dir is None:
            parent_dir = os.path.dirname(os.path.abspath(base_dir))
            logger.info(f"Using parent directory: {parent_dir}")
        
        data_dir = os.path.join(parent_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        random.seed(seed)

        organ_list = ['Spinal cord',
                    'Brain',
                    'Breast',
                    'Bowel',
                    'Skin',
                    'Kidney',
                    'Heart',
                    'Prostate',
                    'Lung',
                    'Liver',
                    'Uterus',
                    'Eye',
                    'Muscle',
                    'Bone',
                    'Pancreas',
                    'Bladder',
                    'Lymphoid',
                    'Cervix',
                    'Lymph node',
                    'Ovary',
                    'Embryo',
                    'Lung/Brain',
                    'Whole organism',
                    'Kidney/Brain',
                    'Placenta']
        if organ is not None:
            if organ in organ_list:
                organs_to_process = [organ]
                logger.info(f'Processing organ: {organ}')
            else:
                logger.error(f"{organ} is not in list.")
                return data_dir
        else:
            organs_to_process = organ_list
            logger.info(f'Processing all organs: {organ_list}')

        for organ in organs_to_process:
            organ_dir = os.path.join(data_dir, organ)
            os.makedirs(organ_dir, exist_ok=True)

            train_dir = os.path.join(organ_dir, 'train')
            test_dir = os.path.join(organ_dir, 'test')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            patches_dir = os.path.join(base_dir, organ, 'patches')
            if not os.path.exists(patches_dir):
                print(f"Datasets for organ {organ} doesn't exist, skipping...")
                continue

            h5_files = [f for f in os.listdir(patches_dir) if f.endswith('.h5')]
            logger.info(f"Found {len(h5_files)} h5 files for organ {organ}")

            for h5_file in tqdm(h5_files, desc=f"Processing {organ} h5 files"):
                h5_path = os.path.join(patches_dir, h5_file)
                try:
                    with h5py.File(h5_path, 'r') as f:
                        if 'img' in f:
                            patches = f['img'][...]
                            logger.info(f"File: {h5_file}, Patches shape: {patches.shape}")
                        else:
                            logger.warning(f"No 'img' dataset found in {h5_path}, skipping...")
                            continue
                        
                        indices = list(range(len(patches)))
                        random.shuffle(indices)
                        split = int(len(indices) * train_test_split)
                        train_idx = indices[:split]
                        test_idx = indices[split:]

                        for i, idx in enumerate(train_idx):
                            patch = patches[idx]
                            img = Image.fromarray(patch.astype(np.uint8))
                            img.save(os.path.join(train_dir, f"{h5_file[:-3]}_{i}.png"))
                        
                        for i, idx in enumerate(test_idx):
                            patch = patches[idx]
                            img = Image.fromarray(patch.astype(np.uint8))
                            img.save(os.path.join(test_dir, f"{h5_file[:-3]}_{i}.png"))
                except Exception as e:
                    logger.error(f"Error processing {h5_path}: {str(e)}")
        return data_dir
    
    def feature_extractor(self, data_dir, organ=None):
        if self.model is None:
            self.model, self.transform, self.device = self.load_uni(self.model_type, self.device)

        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        organ_list = ['Spinal cord',
                    'Brain',
                    'Breast',
                    'Bowel',
                    'Skin',
                    'Kidney',
                    'Heart',
                    'Prostate',
                    'Lung',
                    'Liver',
                    'Uterus',
                    'Eye',
                    'Muscle',
                    'Bone',
                    'Pancreas',
                    'Bladder',
                    'Lymphoid',
                    'Cervix',
                    'Lymph node',
                    'Ovary',
                    'Embryo',
                    'Lung/Brain',
                    'Whole organism',
                    'Kidney/Brain',
                    'Placenta']
        if organ is not None:
            if organ in organ_list:
                organs_to_process = [organ]
                logger.info(f'Processing organ: {organ}')
            else:
                logger.error(f"{organ} is not in list.")
                return data_dir
        else:
            organs_to_process = organ_list
            logger.info(f'Processing all organs: {organ_list}')
        results = {}

        for organ in organs_to_process:
            organ_path = os.path.join(data_dir, organ)
            if not os.path.exists(organ_path):
                logger.warning(f"Path for organ {organ} doesn't exist, skipping...")
                continue
            organ_output = os.path.join(output_dir, organ)
            os.makedirs(organ_output, exist_ok=True)
            train_path = os.path.join(organ_path, 'train')
            if os.path.exists(train_path):
                logger.info(f"Extracting features from {train_path}")
                train_features = self._extract_features_from_folder(train_path)
                
                train_output = os.path.join(organ_output, 'train_features.pt')
                torch.save(train_features, train_output)
                logger.info(f"Saved train features to {train_output}")
                results[f"{organ}_train"] = train_output
            
            test_path = os.path.join(organ_path, 'test')
            if os.path.exists(test_path):
                logger.info(f"Extracting features from {test_path}")
                test_features = self._extract_features_from_folder(test_path)
                
                test_output = os.path.join(organ_output, 'test_features.pt')
                torch.save(test_features, test_output)
                logger.info(f"Saved test features to {test_output}")
                results[f"{organ}_test"] = test_output
        
        return results




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature extraction and data preparation')
    parser.add_argument('--base_dir', type=str, default='./hest_data', 
                        help='base directory for data which contains all organs')
    parser.add_argument('--organ', type=str, default=None, 
                        help='specific organ to process, if None, process all organs')
    parser.add_argument('--output_dir', type=str, default='./results/features', 
                        help='feature extracted output directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for feature extraction')
    parser.add_argument('--prepare_data', action='store_true',
                        help='whether to prepare data from h5 files')
    
    args = parser.parse_args()
    
    if args.prepare_data:
        print(f"Prepare datasets: base_dir={args.base_dir}, organ={args.organ}")
        data_dir = FeatureExtractor.load_patch_from_h5(
            base_dir=args.base_dir,
            organ=args.organ
        )
        print(f"Finished data preparation and save to: {data_dir}")
    else:
        parent_dir = os.path.dirname(os.path.abspath(args.base_dir))
        data_dir = os.path.join(parent_dir, 'data')
    
    extractor = FeatureExtractor(
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    results = extractor.extract_features_for_dataset(data_dir, args.organ)
    
    print("Feature extraction complete. Results saved to:")
    for key, path in results.items():
        print(f"  - {key}: {path}")