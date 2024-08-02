import torch
import torch.nn as nn
import math
import einops
import torchvision
import torch.utils

from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config


import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import os
import pandas as pd
from monai.networks.nets import ViT


# =====================================================================
#
#          single image modeling
#
# =====================================================================
class SingleImageClassifier(torch.nn.Module):
    def __init__(self, backbone_name='dino', class_num=2, model_type='t1t2'):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == 'dino':
            self.backbone = dino_backbone(model_scale='vitb14')
            d_model = 768
        elif backbone_name == 'resnet':
            self.backbone = resnet_backbone()
            d_model = 2048
        else:
            raise ValueError('backbone should in [dino, resnet]')
        
        self.mode = model_type
        if self.mode == 't1' or self.mode == 't2':
            self.cls_head = nn.Linear(d_model, class_num)
        elif self.mode == 't1t2':
            self.cls_head = nn.Sequential(
                nn.Linear(d_model*2, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        else:
            raise ValueError('mode should in [single, double]')
        
    def forward(self, modality_1, modality_2):
        if self.mode == 't1':
            mod_feats = self._embedding(modality_1)
        elif self.mode == 't2':
            mod_feats = self._embedding(modality_2)
        elif self.mode == 't1t2':
            assert modality_1.shape[1] == modality_2.shape[1], 'Error! modality_1.shape should be equal to modality_2.shape'
            modfeat_1 = self._embedding(modality_1)
            modfeat_2 = self._embedding(modality_2)
            mod_feats = torch.cat([modfeat_1, modfeat_2], dim=1)  # [batch, 2*d_model]
        else:
            raise ValueError('mode should in [t1, t2, t1t2]')
        cls_pred = self.cls_head(mod_feats)  # [batch, 2]
        return cls_pred  # [batch, 2]
    
    def _embedding(self, x):
        if self.backbone_name == 'dino':
            with torch.no_grad():
                x_embed = self.backbone.forward_features(x)['x_norm_clstoken']  # [batch, d_model]
        elif self.backbone_name == 'resnet':
            with torch.no_grad():
                x_embed = torch.flatten(self.backbone(x), 1)
        return x_embed
    
