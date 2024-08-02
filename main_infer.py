import os
import json
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import SingleImageClassifier


batch_size = 1
image_size = 448
root_dir = './data/otscc_imgs'

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])


valid_dataset = OTSCCSglImgDataset(root_dir=root_dir, json_file='./data/external test.json', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
model_vit_t1t2 = SingleImageClassifier(backbone_name='dino', model_type='t1t2').cuda()
pretrianed_checkpoint_path = './run_checkpoint/single_dino_t1t2.pth'
model_vit_t1t2.load_state_dict(torch.load(pretrianed_checkpoint_path, map_location='cuda:0'))

y_id = []
y_true = []
y_pred = []
y_proba = []
model_vit_t1t2.eval()
for i, batch_data in tqdm(enumerate(valid_loader)):
    ce_image = batch_data['ce_image'].cuda()
    t2_image = batch_data['t2_image'].cuda()
    label = batch_data['label'].cuda()
    y_id.extend(batch_data['id'])
    y_true.extend(label.cpu().detach().numpy())
    with torch.no_grad():
        output = model_vit_t1t2(ce_image, t2_image)
        _, predicted = torch.max(output.data, 1)
        y_pred.extend(predicted.cpu().detach().numpy())
        y_proba.extend(output[:, 1].cpu().detach().numpy())

y_dict = {
    'ID': y_id,
    'label': y_true,
    'prediction': y_pred,
    'probability': y_proba
}

df = pd.DataFrame(y_dict)
df = df.sort_values(by='ID').reset_index(drop=True)
df.to_excel('data.xlsx', index=False)

