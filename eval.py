import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms as T
import logging
from datetime import datetime
import sys
import torchmetrics
from tqdm import tqdm
from math import sqrt
import numpy as np

from dataloaders.HEDataset import HEDataset, realHEDataset, real_BASE_PATH
from models import helper, regression
import commons

# from train import backbone_arch, agg_arch, agg_config, regression_in_dim, device, regression_ratio, range_threshold, test_transform
from parameters import backbone_arch, agg_arch, agg_config, regression_in_dim, device, regression_ratio, range_threshold, test_transform
from utils.checkpoint import resume_model, resume_train_with_params
from utils.inference import inference

test_datasets = ['real_photo']

resume_info = {
    'resume_model': True,
    # 'resume_model_path': './logs/HE-dinov2_vitb14-MixVPR/2024-09-10_11-29-15/best_model.pth',
    'resume_model_path': '/root/workspace/Height-Estimation/best_model.pth',
    'resume_train': False,
    'resume_train_path': './logs/HE-dinov2_vitb14-MixVPR/2024-09-10_11-29-15/last_checkpoint.pth',
    'device': 'cuda'
}
exp_name = f'HE-{backbone_arch}-{agg_arch}'
save_dir = os.path.join("logs", exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# commons.make_deterministic(seed)
commons.setup_logging(save_dir, console="info")
logging.info(f'test_datasets: {test_datasets}')
logging.info(f'real_photo_BASE_PATH: {real_BASE_PATH}')

backbone = helper.get_backbone(backbone_arch=backbone_arch, num_trainable_blocks=0)
aggregator = helper.get_aggregator(agg_arch=agg_arch, agg_config=agg_config)
regressor = regression.Regression(in_dim=regression_in_dim, regression_ratio=regression_ratio)

backbone = backbone.to(device)
aggregator = aggregator.to(device)
regressor = regressor.to(device)

model = nn.Sequential(backbone, aggregator, regressor)

# size = (336, 448)
# # target_size = ((size[0]//14)*14, (size[1]//14)*14)
# # cut_size_l = ((size[0]-target_size[0])//2, (size[1]-target_size[1])//2)
# # cut_size_r = (size[0] - cut_size_l[0] - target_size[0], size[1] - cut_size_l[1] - target_size[1])
# # size_r = (target_size[0] + cut_size_l[0], target_size[1] + cut_size_l[1])
# x = torch.randn(32, 3, size[0], size[1]).to(device)
# # x = x[:,:, cut_size_l[0]:size_r[0], cut_size_l[1]:size_r[1]]
# r = model(x)
# print_nb_params(model)
# print(f'Input shape is {x.shape}')
# print(f'Output shape is {r.shape}')

# logging.info(f"Feature dim: {model.feature_dim}")

model = resume_model(model, resume_info)

model.eval()

test_dataset_list = []
test_datasets_load = test_datasets
if 'real_photo' in test_datasets:
    real_photo_dataset = realHEDataset()
    test_datasets_load.remove('real_photo')
    test_dataset_list.append(real_photo_dataset)
if len(test_datasets_load) != 0:
    fake_photo_dataset = HEDataset(foldernames=test_datasets_load, random_sample_from_each_place=False,transform=test_transform)
    test_dataset_list.append(fake_photo_dataset)
if len(test_dataset_list) > 1:
    test_dataset = ConcatDataset(test_dataset_list)
else:
    test_dataset = test_dataset_list[0]
test_img_num = len(test_dataset)
test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

val_recall_str, val_recall_percentage = inference(model, test_dl, range_threshold, test_img_num, device)

logging.info(f"LR: {val_recall_str}")