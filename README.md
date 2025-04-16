# Pipeline (Replace by name)
This repository contains a multi-stage pipeline for analyzing pathology images alongside spatial transcriptomics (ST) data, with a focus on cell type classification and distribution for those out-of-distribution (OOD) data.
# Installation
```
conda create --name "(your_name)"
conda activate "(your_name)"
pip install -r requirements.txt
```
# Data preparation
## For testing using a single WSI
```
# Where index should keep same as file's name
python feature_extractor.py --prepare_patch_data --img_path /path/to/your/image.h5 --patch_dir ./test_patches_data --patch_idx 001 
```

## For whole organ dataset
Just simplely type for preparing a single organ (like breast):
```
python feature_extractor.py --prepare_organ_data --h5_dir ./hest_data --organ breast 
```

# Four steps
## 1. Feature extraction
### Testing feature extraction using patches from single WSI
```
python feature_extractor.py --extract_patch_features --patch_dir ./test_patches_data --output_dir ./results/patch_features --batch_size 16
```
### Feature extraction for whole organ
This step is used to fine-tuning UNI on pathology datasets and extract features from them.
```
# For specific organ - like breast
python feature_extractor.py --extract_organ_features --base_dir ./data --organ breast --output_dir ./results/organ_features --batch_size 16

# For all organs in datasets
python uni_feature_extractor.py --extract_organ_features --base_dir ./data --output_dir ./features
```
## 2. Deconvolution
