import torch
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch.utils.data
import torchvision.transforms as T
import json

import torch.utils
from sklearn.model_selection import train_test_split


class OTSCCSglImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, json_files=None, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.split = split
        if json_files is None:
            json_files = {
                'train': './data/train.json',
                'val': './data/validation.json',
                'internal_test': './data/internal test.json',
                'external_test': './data/external test.json'
            }

        # Load IDs from JSON files
        ids_dict = {}
        for split_name, json_file in json_files.items():
            with open(json_file, 'r') as f:
                ids_dict[split_name] = set(json.load(f))

        for label in ['0', '1']:
            ce_dir = os.path.join(root_dir, 'ce', label)
            t2_dir = os.path.join(root_dir, 't2', label)

            ce_images = os.listdir(ce_dir)

            for ce_image in ce_images:
                id_name = ce_image.split('.')[0]
                ce_image_path = os.path.join(ce_dir, ce_image)
                t2_image_path = os.path.join(t2_dir, ce_image)  # assuming t2 image has the same name as ce image

                if id_name in ids_dict[split]:
                    self.data.append({
                        'id': id_name,
                        'label': int(label),
                        'ce_image_path': ce_image_path,
                        't2_image_path': t2_image_path
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        ce_image = Image.open(sample['ce_image_path']).convert("RGB")
        t2_image = Image.open(sample['t2_image_path']).convert("RGB")

        if self.transform:
            ce_image = self.transform(ce_image)
            t2_image = self.transform(t2_image)

        return {
            'id': sample['id'],
            'label': sample['label'],
            'ce_image': ce_image,
            't2_image': t2_image
        }

