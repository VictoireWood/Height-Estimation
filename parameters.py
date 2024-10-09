import warnings

# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import os
import torchvision.transforms as T
from datetime import datetime

# resume_info,train_batch_size,num_workers,num_epochs,scheduler_patience,lr,seed,device,foldernames,train_dataset_folders,test_datasets,image_size,range_threshold,backbone_arch,agg_arch,agg_config,regression_in_dim,regression_ratio,exp_name,save_dir,train_transform,test_transform

resume_info = {
    'resume_model': False,
    'resume_model_path': './logs/HE-dinov2_vitb14-MixVPR/2024-09-10_11-29-15/best_model.pth',
    'resume_train': False,
    'resume_train_path': './logs/HE-dinov2_vitb14-MixVPR/2024-09-10_11-29-15/last_checkpoint.pth',
    'device': 'cuda'
}


train_batch_size = 32
num_workers = 16
num_epochs = 150

scheduler_patience = 10
lr = 0.00001

seed = 0

resume_train = False
# resume_model = 

device = "cuda" if torch.cuda.is_available() else "cpu"

foldernames=['2013', '2017', '2019', '2020', '2022', 'real_photo']
train_dataset_folders = ['2013', '2017', '2019', '2020', '2022']
test_datasets = ['real_photo']

image_size = (360, 480)

range_threshold = [25, 50, 75, 100, 125, 150]

# backbone_arch = 'efficientnet_v2_m'
# agg_arch='MixVPR'
# agg_config={'in_channels' : 1280,
#             'in_h' : 12,
#             'in_w' : 15,
#             'out_channels' : 640,
#             'mix_depth' : 4,
#             'mlp_ratio' : 1,
#             'out_rows' : 4,
#             }   # the output dim will be (out_rows * out_channels)


backbone_arch = 'dinov2_vitb14'
agg_arch='MixVPR'
agg_config={'in_channels' : 768,
            'in_h' : 25,
            'in_w' : 34,
            'out_channels' : 384,
            'mix_depth' : 4,
            'mlp_ratio' : 1,
            'out_rows' : 4,
            }   # the output dim will be (out_rows * out_channels)


regression_in_dim = agg_config['out_rows'] * agg_config['out_channels']
regression_ratio = 0.5



exp_name = f'HE-{backbone_arch}-{agg_arch}'
save_dir = os.path.join("logs", exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


train_transform = T.Compose([
    T.Resize(image_size, antialias=True),
    # T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1], antialias=True),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
    T.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform =T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


resume_info = {
    'resume_model': True,
    # 'resume_model_path': './logs/HE-dinov2_vitb14-MixVPR/2024-09-10_11-29-15/best_model.pth',
    'resume_model_path': '/root/workspace/Height-Estimation/best_model.pth',
    'resume_train': False,
    'resume_train_path': './logs/HE-dinov2_vitb14-MixVPR/2024-09-10_11-29-15/last_checkpoint.pth',
    'device': 'cuda'
}