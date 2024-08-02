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
#          3D ViT modeling
#
# =====================================================================

# image_size = (32, 256, 256)
# patch_size = (4, 16, 16)
# embed_dim = 768
# mlp_dim = 3072
# num_layers = 12
# num_heads = 12
# pos_embed = 'perceptron'
# dropout_rate = 0.0

# classification model
class TDViT_Classifier(nn.Module):
    def __init__(self, 
                 model_type='t1t2',
                 backbone_checkpoint='./pretrained_checkpoint/ViT3d_pretrain.ckpt'):
        super().__init__()
        self.model_type = model_type
        self.backbone = ViT(in_channels=1,
                            img_size=(32, 256, 256),
                            patch_size=(4, 16, 16),
                            hidden_size=768,
                            mlp_dim=3072,
                            num_layers=12,
                            num_heads=12,
                            pos_embed='perceptron',
                            classification=False,
                            dropout_rate=0.0,
                        )
        with open(backbone_checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')['state_dict']
            encoder_dict = {k.replace('model.encoder.', ''): v for k, v in state_dict.items() if 'model.encoder.' in k}
        self.backbone.load_state_dict(encoder_dict)

        if model_type == 't1t2':
            self.linear_head = nn.Linear(2 * 2048 * 768, 2)
        elif model_type == 't1' or model_type == 't2':
            self.linear_head = nn.Linear(2048 * 768, 2)
        else:
            raise ValueError("Invalid model_type")
        
        self.t1_features = None
        self.t2_features = None

    def forward(self, modality_1, modality_2):
        with torch.no_grad():
            if self.model_type == 't1t2':
                self.t1_features, _ = self.backbone(modality_1)  # [batch_size, 2048, 768]
                self.t2_features, _ = self.backbone(modality_2)  # [batch_size, 2048, 768]
                combined_features = torch.cat((self.t1_features, self.t2_features), dim=1)  # [batch_size, 4096, 768]
            elif self.model_type == 't1':
                self.t1_features, _ = self.backbone(modality_1)  # [batch_size, 2048, 768]
                combined_features = self.t1_features  # [batch_size, 4096, 768]
            elif self.model_type == 't2':
                self.t2_features, _ = self.backbone(modality_2)  # [batch_size, 2048, 768]
                combined_features = self.t2_features  # [batch_size, 4096, 768]
            else:
                raise ValueError("Invalid model_type")

        combined_features = combined_features.view(combined_features.size(0), -1)  # [batch_size, 4096*768]
        final_output = self.linear_head(combined_features)
        return final_output
    

# =====================================================================
#
#          inter-slice and intra-slice modeling
#
# =====================================================================
def dino_backbone(model_scale: str = 'vits14'):
    feature_dim = 384  # small
    # feature_dim = 768  # base
    # feature_dim = 1024  # large
    # patch_size = 14
    config_path = f'eval/{model_scale}_reg4_pretrain'
    checkpoint_path = f'./dinov2/checkpoints/dinov2_{model_scale}_reg4_pretrain.pth'
    conf = load_and_merge_config(config_path)
    model = build_model_for_eval(conf, checkpoint_path)
    return model


def resnet_backbone():
    model = torchvision.models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    return model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 60):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

# classification model
class SequenceImages(torch.nn.Module):
    def __init__(self, backbone_name='dino', class_num=2, model_type='t1t2'):
        super().__init__()

        self.backbone_name = backbone_name
        if backbone_name == 'dino':
            self.backbone = dino_backbone()
            d_model = 384
        elif backbone_name == 'resnet':
            self.backbone = resnet_backbone()
            d_model = 2048
        else:
            raise ValueError('backbone should in [dino, resnet]')

        self.pos_encoder = PositionalEncoding(d_model=d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=200, dropout=0.2)
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=2)

        self.mode = model_type

        if self.mode == 't1' or self.mode == 't2':
            self.cls_head = nn.Linear(d_model, class_num)
        elif self.mode == 't1t2':
            self.cls_head = nn.Linear(d_model*2, class_num)
        else:
            raise ValueError('mode should in [single, double]')

    def forward(self, modality_1, modality_2):
        if self.mode == 't1':
            mod_feats = self._sequence_encoding(modality_1)
        elif self.mode == 't2':
            mod_feats = self._sequence_encoding(modality_2)
        elif self.mode == 't1t2':
            assert modality_1.shape[1] == modality_2.shape[1], 'Error! modality_1.shape should be equal to modality_2.shape'
            modfeat_1 = self._sequence_encoding(modality_1)
            modfeat_2 = self._sequence_encoding(modality_2)
            mod_feats = torch.cat([modfeat_1, modfeat_2], dim=1)  # [batch, slice*2, 384]
        else:
            raise ValueError('mode should in [t1, t2, t1t2]')
        cls_pred = self.cls_head(mod_feats)  # [batch, 2]
        return cls_pred  # [batch, 2]
    
    def _sequence_embedding(self, x):
        # x: [batch, slice=5, channel=3, height=224, width=224]
        batch_size, slice_num = x.shape[0], x.shape[1]
        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        if self.backbone_name == 'dino':
            with torch.no_grad():
                x_feat = self.backbone.forward_features(x)['x_norm_clstoken']  # [batch * slice, 384]
        elif self.backbone_name == 'resnet':
            with torch.no_grad():
                x_feat = torch.flatten(self.backbone(x), 1)  # [batch * slice, 2048]
        stage_1 = einops.rearrange(x_feat, '(b s) f -> b s f', b=batch_size, s=slice_num)  # [batch, slice, 384]
        stage_2 = stage_1.mean(dim=1)
        return stage_2
    

# =====================================================================
#
#          slice-stack modeling
#
# =====================================================================
class SeqStackClassifier(torch.nn.Module):
    def __init__(self, backbone_name='dino', class_num=2, model_type='t1t2'):
        super().__init__()

        self.backbone_name = backbone_name
        if backbone_name == 'dino':
            self.backbone = dino_backbone()
            d_model = 384
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
                nn.Linear(d_model*2, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        else:
            raise ValueError('mode should in [single, double]')

    def forward(self, modality_1, modality_2):
        if self.mode == 't1':
            mod_feats = self._sequence_embedding(modality_1)
        elif self.mode == 't2':
            mod_feats = self._sequence_embedding(modality_2)
        elif self.mode == 't1t2':
            assert modality_1.shape[1] == modality_2.shape[1], 'Error! modality_1.shape should be equal to modality_2.shape'
            modfeat_1 = self._sequence_embedding(modality_1)
            modfeat_2 = self._sequence_embedding(modality_2)
            mod_feats = torch.cat([modfeat_1, modfeat_2], dim=1)  # [batch, slice*2, 384]
        else:
            raise ValueError('mode should in [t1, t2, t1t2]')
        cls_pred = self.cls_head(mod_feats)  # [batch, 2]
        return cls_pred  # [batch, 2]
    
    def _sequence_embedding(self, x):
        # x: [batch, slice=5, channel=3, height=224, width=224]
        batch_size, slice_num = x.shape[0], x.shape[1]
        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        if self.backbone_name == 'dino':
            with torch.no_grad():
                x_feat = self.backbone.forward_features(x)['x_norm_clstoken']  # [batch * slice, 384]
        elif self.backbone_name == 'resnet':
            with torch.no_grad():
                x_feat = torch.flatten(self.backbone(x), 1)  # [batch * slice, 2048]
        stage_1 = einops.rearrange(x_feat, '(b s) f -> b s f', b=batch_size, s=slice_num)  # [batch, slice, 384]
        stage_2 = stage_1.mean(dim=1)
        return stage_2
    

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
    