import os
import json
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from models import SingleImageClassifier


batch_size = 1
image_size = 448
root_dir = './data/otscc_imgs'

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])


class OTSCCSglImgDataset(Dataset):
    def __init__(self, root_dir, json_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        with open(json_file, 'r') as f:
            ids = json.load(f)

        for label in ['0', '1']:
            ce_dir = os.path.join(root_dir, 'ce', label)
            t2_dir = os.path.join(root_dir, 't2', label)

            ce_images = os.listdir(ce_dir)

            for ce_image in ce_images:
                id_name = ce_image.split('.')[0]
                ce_image_path = os.path.join(ce_dir, ce_image)
                t2_image_path = os.path.join(t2_dir, ce_image)  

                if id_name in ids:
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
    

# valid_dataset = OTSCCSglImgDataset(root_dir=root_dir, json_file='./data/external test.json', transform=transform)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# model_vit_t1t2 = SingleImageClassifier(backbone_name='dino', model_type='t1t2').cuda()
# pretrianed_checkpoint_path = './run_checkpoint/single_dino_t1t2.pth'
# model_vit_t1t2.load_state_dict(torch.load(pretrianed_checkpoint_path, map_location='cuda:0'))

# y_id = []
# y_true = []
# y_pred = []
# y_proba = []
# model_vit_t1t2.eval()
# for i, batch_data in tqdm(enumerate(valid_loader)):
#     ce_image = batch_data['ce_image'].cuda()
#     t2_image = batch_data['t2_image'].cuda()
#     label = batch_data['label'].cuda()
#     y_id.extend(batch_data['id'])
#     y_true.extend(label.cpu().detach().numpy())
#     with torch.no_grad():
#         output = model_vit_t1t2(ce_image, t2_image)
#         _, predicted = torch.max(output.data, 1)
#         y_pred.extend(predicted.cpu().detach().numpy())
#         y_proba.extend(output[:, 1].cpu().detach().numpy())

# y_dict = {
#     'ID': y_id,
#     'label': y_true,
#     'prediction': y_pred,
#     'probability': y_proba
# }

# df = pd.DataFrame(y_dict)
# df = df.sort_values(by='ID').reset_index(drop=True)
# df.to_excel('data.xlsx', index=False)

