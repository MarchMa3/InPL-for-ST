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
                'model_name': 'vit_giant_patch14_224',
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

        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
        random.seed(seed)

        organ_list = ['spinal cord',
                    'brain',
                    'breast',
                    'bowel',
                    'skin',
                    'kidney',
                    'heart',
                    'prostate',
                    'lung',
                    'liver',
                    'uterus',
                    'eye',
                    'muscle',
                    'bone',
                    'pancreas',
                    'bladder',
                    'lymphoid',
                    'cervix',
                    'lymph node',
                    'ovary',
                    'embryo',
                    'lung/brain',
                    'whole organism',
                    'kidney/brain',
                    'placenta']
        if organ is not None:
            organ = organ.lower()
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
            train_class_dir = os.path.join(train_dir, organ)
            test_class_dir = os.path.join(test_dir, organ)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

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
                            img.save(os.path.join(train_class_dir, f"{h5_file[:-3]}_{i}.png"))
                        
                        for i, idx in enumerate(test_idx):
                            patch = patches[idx]
                            img = Image.fromarray(patch.astype(np.uint8))
                            img.save(os.path.join(test_class_dir, f"{h5_file[:-3]}_{i}.png"))
                except Exception as e:
                    logger.error(f"Error processing {h5_path}: {str(e)}")
        return data_dir
    
    def feature_extractor(self, data_dir, organ=None):
        if self.model is None:
            self.model, self.transform, self.device = self.load_uni(self.model_type, self.device)

        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        organ_list = ['spinal cord',
                    'brain',
                    'breast',
                    'bowel',
                    'skin',
                    'kidney',
                    'heart',
                    'prostate',
                    'lung',
                    'liver',
                    'uterus',
                    'eye',
                    'muscle',
                    'bone',
                    'pancreas',
                    'bladder',
                    'lymphoid',
                    'cervix',
                    'lymph node',
                    'ovary',
                    'embryo',
                    'lung/brain',
                    'whole organism',
                    'kidney/brain',
                    'placenta']
        if organ is not None:
            organ = organ.lower()
        if organ is not None:
            if organ in organ_list:
                organs_to_process = [organ]
                logger.info(f'Processing organ: {organ}')
            else:
                logger.error(f"{organ} is not in list.")
                return {}
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

            # Train dataset 
            train_path = os.path.join(organ_path, 'train')
            if os.path.exists(train_path):
                logger.info(f"Extracting features from {train_path}")
                train_dataset = datasets.ImageFolder(train_path, transform=self.transform)
                train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False,
                    num_workers=self.num_workers
                )
            
                train_features_dict = extract_patch_features_from_dataloader(self.model, train_dataloader)
                train_features = {
                'embeddings': torch.Tensor(train_features_dict['embeddings']),
                'labels': torch.Tensor(train_features_dict['labels']).type(torch.long),
                'classes': train_dataset.classes
                }
                train_output = os.path.join(organ_output, 'train_features.pt')
                torch.save(train_features, train_output)
                logger.info(f"Saved train features to {train_output}")
                results[f"{organ}_train"] = train_output

            test_path = os.path.join(organ_path, 'test')
            if os.path.exists(test_path):
                logger.info(f"Extracting features from {test_path}")
                
                test_dataset = datasets.ImageFolder(test_path, transform=self.transform)
                test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False,
                    num_workers=self.num_workers
                )
                
                test_features_dict = extract_patch_features_from_dataloader(self.model, test_dataloader)
                
                test_features = {
                    'embeddings': torch.Tensor(test_features_dict['embeddings']),
                    'labels': torch.Tensor(test_features_dict['labels']).type(torch.long),
                    'classes': test_dataset.classes
                }
                
                test_output = os.path.join(organ_output, 'test_features.pt')
                torch.save(test_features, test_output)
                logger.info(f"Saved test features to {test_output}")
                results[f"{organ}_test"] = test_output
    
        return results



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature extraction and data preparation')
    parser.add_argument('--base_dir', type=str, default='./data', 
                        help='base directory for data which contains all organs')
    parser.add_argument('--h5_dir', type=str, default='./hest_data',
                        help='directory containing original h5 files (only used with --prepare_data)')
    parser.add_argument('--organ', type=str, default=None, 
                        help='specific organ to process, if None, process all organs')
    parser.add_argument('--output_dir', type=str, default='./results/features', 
                        help='feature extracted output directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for feature extraction')
    parser.add_argument('--prepare_data', action='store_true',
                        help='whether to prepare data from h5 files')
    
    args = parser.parse_args()

    if args.organ:
        args.organ = args.organ.lower()
    
    if args.prepare_data:
        print(f"Preparing datasets from h5 files: h5_dir={args.h5_dir}, organ={args.organ}")
        data_dir = FeatureExtractor.load_patch_from_h5(
            base_dir=args.h5_dir,  
            organ=args.organ
        )
        print(f"Finished data preparation and save to: {data_dir}")
    else:
        print(f"Using pre-processed data from: {args.base_dir}")
        data_dir = args.base_dir  
    
    extractor = FeatureExtractor(
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    results = extractor.feature_extractor(data_dir, args.organ)
    
    if results:
        print("Feature extraction complete. Results saved to:")
        for key, path in results.items():
            print(f"  - {key}: {path}")
    else:
        print("No features were extracted. Please check the organ name or data directory.")