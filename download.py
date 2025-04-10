import pandas as pd
import datasets
from huggingface_hub import login
import os

base_dir = "../InPL-for-ST/hest_data"

login()

meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")

organs = ['Breast'] #'Spinal cord', 'Brain']
for organ in organs:
    organ_dir = os.path.join(base_dir, organ.replace(' ', '_').lower())
    os.makedirs(organ_dir, exist_ok=True)

    organ_df = meta_df[meta_df['organ'] == organ]
    id = organ_df['id'].tolist()

    if len(id) > 0:
        list_patterns = [f"*{id}[_.]**" for id in id]
        dataset = datasets.load_dataset(
                'MahmoodLab/hest',  
                cache_dir=organ_dir,
                patterns=list_patterns
            )
        print(f"Downloaded {len(id)} images for {organ}")
    else:
        print(f"No images found for {organ}")
    



