# https://github.com/amaralibey/gsv-cities

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from glob import glob
import numpy as np

from he_database_generate import height_to_width
from commons import scale_down
# import logging
import random

size = (360, 480)

# 给图像加变形，看是不是能训练出结果
def random_transform(image: Image.Image, height):
    w = image.width
    h = image.height
    # rotate = random.uniform(0, 360)
    scale = random.uniform(1.4, 2)
    crop_w = w // scale
    crop_h = h // scale
    real_scale = ((h / crop_h) + (w / crop_w))/2
    flip_chance = 0.5
    rand_transform = T.Compose([
        T.RandomVerticalFlip(flip_chance),
        T.RandomHorizontalFlip(flip_chance),
        T.RandomRotation(180),
        T.CenterCrop((crop_h, crop_w)),
        T.Resize((h, w)),
    ])
    new_height = height/real_scale
    new_image = rand_transform(image)
    return new_image, new_height

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
# BASE_PATH = '/root/shared-storage/shaoxingyu/workspace_backup/gsvqddb_train/'
# real_BASE_PATH = '/root/shared-storage/shaoxingyu/workspace_backup/dcqddb_test/'

# BASE_PATH = '/root/shared-storage/shaoxingyu/heqd_train/'
# real_BASE_PATH = '/root/shared-storage/shaoxingyu/heqd_test/'

# real_BASE_PATH_list = [
#     '/root/shared-storage/shaoxingyu/heqd_test/VPR/',
#     '/root/shared-storage/shaoxingyu/heqd_test/VPR2/',
#     '/root/shared-storage/shaoxingyu/heqd_test/VPR_h400/',
#     '/root/shared-storage/shaoxingyu/heqd_test/VPR_h630/',
# ]
# real_BASE_PATH = real_BASE_PATH_list[3]

# real_BASE_PATH = '/root/shared-storage/shaoxingyu/heqd_test_real_photo/'

BASE_PATH = '/root/workspace/maps/HE_Train_Large/'
BASE_PATH = '/root/workspace/maps/HE_Train_qdcity/'
real_BASE_PATH = '/root/workspace/maps/HE_Test/'

# BASE_PATH = '/root/workspace/gsvqddb_train/'
# real_BASE_PATH = '/root/workspace/dcqddb_test/'


if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

default_transform = T.Compose([
    # T.Resize(size, antialias=True),
    # T.Resize(args.train_resize, antialias=True),
    # T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1], antialias=True),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
    T.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

basic_transform = T.Compose([
    T.Resize(size, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class HEDataset(Dataset):
    def __init__(self,
                 foldernames=['2013', '2017', '2019', '2020', '2022'],
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH,
                 random_transform = True,
                 ):
        super(HEDataset, self).__init__()
        self.base_path = base_path
        self.foldernames = foldernames

        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.dataframes = self.__getdataframes()
        # self.heights = list(self.dataframes['flight_height'].values)
        # self.heights_tensor = torch.tensor(self.heights, dtype=torch.float16).unsqueeze(1)
        self.year = list(self.dataframes['year'])
        self.flight_height = list(self.dataframes['flight_height'])
        self.heights_tensor = torch.tensor(self.flight_height).unsqueeze(1)
        self.alpha = list(self.dataframes['rotation_angle'])
        self.loc_x = list(self.dataframes['loc_x'])
        self.loc_y = list(self.dataframes['loc_y'])
        self.random_transform = random_transform
        
        # get all unique place ids
        self.total_nb_images = len(self.dataframes)
        
    def __getdataframes(self) -> pd.DataFrame:
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path + 'Dataframes/'+f'{self.foldernames[0]}.csv', encoding='utf-8', converters = {'year':str})
        df = df.sample(frac=1)  # shuffle the city dataframe
        

        # append other cities one by one
        for i in range(1, len(self.foldernames)):
            tmp_df = pd.read_csv(
                self.base_path+'Dataframes/'+f'{self.foldernames[i]}.csv', encoding='utf-8', converters = {'year':str})

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            
            df = pd.concat([df, tmp_df], ignore_index=True)

        if self.random_sample_from_each_place:
            df = df.sample(frac=1)

        # keep only places depicted by at least min_img_per_place images
        # res = df[df.groupby('place_id')['place_id'].transform('size') >= self.min_img_per_place]
        # return res.set_index('place_id')

        return df.reset_index(drop=True)
    
    def __getitem__(self, index):
        
        # height = self.heights[index]
        height = self.heights_tensor[index]
        # height = scale_down(height) # NOTE: 后期加入归一化，方便收敛
        image_name = f'@{self.year[index]}@{self.flight_height[index]:.2f}@{self.alpha[index]:.2f}@{self.loc_x[index]}@{self.loc_y[index]}@.png'
        # photo_meters_w = height_to_width(self.flight_height[index])
        # image_name = f'@{self.year[index]}@{photo_meters_w:.7f}@{self.flight_height[index]:.2f}@{self.alpha[index]:.2f}@{self.loc_x[index]}@{self.loc_y[index]}@.png'
        # f'{year}@{flight_height}@{alpha}@{loc_w}@{loc_h}.png'
        image_path = self.base_path + f'Images/{self.year[index]}/' + image_name
        image = Image.open(image_path).convert('RGB')
        
        
        if self.transform:
            image = self.transform(image)

        if self.random_transform:
            image_new, height_new = random_transform(image, height)
            image_out = [image, image_new]
            height_out = [height, height_new]
            return image_out, height_out
        return image, height
    
    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.flight_height)



class realHEDataset(Dataset):
    def __init__(self, base_path=real_BASE_PATH, transform=basic_transform):
        super().__init__()
        self.base_path = base_path
        self.transform = transform
        images_paths = sorted(glob(f"{real_BASE_PATH}/**/*.png", recursive=True))
        
        self.images_paths = images_paths
        self.heights = self.get_heights(images_paths)

    def __getitem__(self, index):
        
        height = self.heights[index]
        # height = scale_down(height) # NOTE: 后期加入归一化，方便收敛
        image_path = self.images_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, height
    
    def __len__(self):
        return len(self.images_paths)
    
    @staticmethod
    def get_heights(image_paths):
        info_list = [image_path.split('/')[-1].split('@') for image_path in image_paths]
        # heights = np.array([info[-4] for info in info_list]).astype(np.float16)
        # heights = torch.tensor(heights, dtype=torch.float16).unsqueeze(1)
        heights = torch.tensor([float(info[-4]) for info in info_list])
        return heights



class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
    