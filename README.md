# Pipeline (Replace by name)
This repository contains a multi-stage pipeline for analyzing pathology images alongside spatial transcriptomics (ST) data, with a focus on cell type classification and distribution for those out-of-distribution (OOD) data.
# Installation
```
conda create --name "(your_name)"
conda activate "(your_name)"
pip install -r requirements.txt
```
# Data preparation
Just simplely type for preparing a single organ (like breast):
```
python uni_feature_extractor.py --prepare_data --h5_dir ./hest_data --organ breast > results/log/prepare_data.log 2>&1
```
You can track the download log in ../results/log/prepare_data.log

# Four steps
## 1. Feature extraction
This step is used to fine-tuning UNI on pathology datasets and extract features from them.
```
# For specific organ - like breast
python uni_feature_extractor.py --h5_dir ./hest_data --organ breast --output_dir ./features --prepare_data

# For all organs in datasets
python uni_feature_extractor.py --base_dir ./data --output_dir ./features
```
## 2. Deconvolution
