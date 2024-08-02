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


# customize dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root_dir, 
                 label_dir,
                 image_size=224,
                 train_type='internal_train'):
        self.root_dir = root_dir
        self.df = pd.read_excel(label_dir)
        self.ids = np.array(sorted(os.listdir(root_dir)))
        if train_type == 'internal_train':
            with open('./data/train_internal.json', 'r') as f:
                train_index = json.load(f)
            self.ids = self.ids[train_index]

        elif train_type == 'internal_test':
            with open('./data/test_internal.json', 'r') as f:
                internal_valid_index = json.load(f)
            self.ids = self.ids[internal_valid_index]

        elif train_type == 'external_test':
            with open('./data/test_external.json', 'r') as f:
                external_valid_index = json.load(f)
            self.ids = self.ids[external_valid_index]

        elif train_type == 'pretrain':
            with open('./data/trainwholeindex.json', 'r') as f:
                pretrain_index = json.load(f)
            self.ids = self.ids[pretrain_index]

        else:
            raise ValueError("Invalid train_type")
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            # T.CenterCrop((input_size, input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        id_folder = os.path.join(self.root_dir, self.ids[idx])
        
        ce_images = [os.path.join(id_folder, 'ce_images', img) for img in sorted(os.listdir(os.path.join(id_folder, 'ce_images')))]
        ce_masks = [os.path.join(id_folder, 'ce_masks', mask) for mask in sorted(os.listdir(os.path.join(id_folder, 'ce_masks')))]
        t2_images = [os.path.join(id_folder, 't2_images', img) for img in sorted(os.listdir(os.path.join(id_folder, 't2_images')))]
        t2_masks = [os.path.join(id_folder, 't2_masks', mask) for mask in sorted(os.listdir(os.path.join(id_folder, 't2_masks')))]
        
        # read images and masks
        ce_images = [Image.open(img).convert('RGB') for img in ce_images]
        ce_masks = [Image.open(mask).convert('L') for mask in ce_masks]
        t2_images = [Image.open(img).convert('RGB') for img in t2_images]
        t2_masks = [Image.open(mask).convert('L') for mask in t2_masks]
        
        if self.transform:
            ce_images = [self.transform(image) for image in ce_images]
            ce_masks = [self.mask_transform(mask) for mask in ce_masks]
            t2_images = [self.transform(image) for image in t2_images]
            t2_masks = [self.mask_transform(mask) for mask in t2_masks]
        
        label = self.df.loc[self.df['ID'] == int(self.ids[idx]), 'label'].values[0]

        sample = {
            'id': self.ids[idx],
            'label': label,
            'ce_images': torch.stack(ce_images),
            'ce_masks': torch.stack(ce_masks),
            't2_images': torch.stack(t2_images),
            't2_masks': torch.stack(t2_masks)
        }
        return sample

def custom_collate_fn(batch):
    ce_images = [item['ce_images'] for item in batch]
    ce_masks = [item['ce_masks'] for item in batch]
    t2_images = [item['t2_images'] for item in batch]
    t2_masks = [item['t2_masks'] for item in batch]

    ce_images = torch.stack(ce_images)
    ce_masks = torch.stack(ce_masks)
    t2_images = torch.stack(t2_images)
    t2_masks = torch.stack(t2_masks)

    ids = [item['id'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    return {'id': ids, 
            'label': labels, 
            'ce_images': ce_images, 
            'ce_masks': ce_masks, 
            't2_images': t2_images, 
            't2_masks': t2_masks}


# class OTSCCSglImgDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, transform=None, split='train', val_split=0.2, test_split=0.2):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.data = []
#         self.split = split

#         for label in ['0', '1']:
#             ce_dir = os.path.join(root_dir, 'ce', label)
#             t2_dir = os.path.join(root_dir, 't2', label)

#             ce_images = os.listdir(ce_dir)

#             for ce_image in ce_images:
#                 id_name = ce_image.split('.')[0]
#                 ce_image_path = os.path.join(ce_dir, ce_image)
#                 t2_image_path = os.path.join(t2_dir, ce_image)  # assuming t2 image has the same name as ce image

#                 self.data.append({
#                     'id': id_name,
#                     'label': int(label),
#                     'ce_image_path': ce_image_path,
#                     't2_image_path': t2_image_path
#                 })

#         # Shuffle and split data
#         train_val_split = 1 - test_split
#         train_split = train_val_split * (1 - val_split)
#         train_val_data, test_data = train_test_split(self.data, test_size=test_split, random_state=42)
#         train_data, val_data = train_test_split(train_val_data, test_size=(val_split / train_val_split), random_state=42)

#         if split == 'train':
#             self.data = train_data
#         elif split == 'val':
#             self.data = val_data
#         elif split == 'test':
#             self.data = test_data
#         else:
#             raise ValueError("split must be one of ['train', 'val', 'test']")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
        
#         ce_image = Image.open(sample['ce_image_path']).convert("RGB")
#         t2_image = Image.open(sample['t2_image_path']).convert("RGB")

#         if self.transform:
#             ce_image = self.transform(ce_image)
#             t2_image = self.transform(t2_image)

#         return {
#             'id': sample['id'],
#             'label': sample['label'],
#             'ce_image': ce_image,
#             't2_image': t2_image
#         }


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

