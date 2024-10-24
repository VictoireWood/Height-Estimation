import torch
from tqdm import tqdm
import numpy as np
from commons import scale_up

from torch.utils.data import DataLoader

def inference(model:torch.nn.Module, test_dl:DataLoader, range_threshold, test_img_num, device):
    # TODO 这里我要返回两个量，一个是平均MSE误差（开方），一个是按照各个高度真实值大小排列的误差平均值、最大误差、最小误差、中间值，以后可以画图看不同高度上的结果
    tqdm_bar = tqdm(range(test_img_num), ncols=100, desc="")
    model.eval()

    count_valid_recall = torch.zeros(len(range_threshold))
    valid_recall_percentage = torch.zeros(len(range_threshold))
    range_threshold_tensor = torch.tensor(range_threshold)

    valid_heigths = torch.zeros(test_img_num, len(range_threshold),dtype=torch.bool)

    with torch.no_grad():
        # tqdm_bar = tqdm(range(), ncols=100, desc="")
        # for images, heights_gt in test_dl:
        for query_i, (images,heights_gt) in enumerate(test_dl):
            images = images.to(device)
            heights_gt = torch.tensor(heights_gt).to(device)
            recall_heights_range = torch.zeros(test_img_num, len(range_threshold))
            # with torch.autocast(device):
            heights_pred = model(images)
            # NOTE 归一化后加入
            # heights_gt = scale_up(heights_gt)
            # heights_pred = scale_up(heights_pred)

            distances = abs(heights_pred - heights_gt)
            range_threshold_tensor = range_threshold_tensor.to(distances.device)
            valid_heigths[query_i,:] = distances < range_threshold_tensor

            tqdm_bar.set_description(f"{query_i:5d}")
            _ = tqdm_bar.refresh()
            _ = tqdm_bar.update()

        count_valid_recall = torch.count_nonzero(valid_heigths, dim=0)
        valid_recall_percentage = 100 * count_valid_recall / test_img_num

    val_recall_str = ", ".join([f'LR@{N}: {acc:.2f}' for N, acc in zip(range_threshold, valid_recall_percentage)])
    return val_recall_str, valid_recall_percentage

def inference_statistics(model:torch.nn.Module, test_dl:DataLoader, test_img_num, device):
    # TODO 这里我要返回两个量，一个是平均MSE误差（开方），一个是按照各个高度真实值大小排列的误差平均值、最大误差、最小误差、中间值，以后可以画图看不同高度上的结果
    tqdm_bar = tqdm(range(test_img_num), ncols=100, desc="")
    model.eval()
    
    inference_info_collect = [] # NOTE 放字典，每个字典包含qurery_i, heights_gt, heights_pred, distances

    with torch.no_grad():
        # tqdm_bar = tqdm(range(), ncols=100, desc="")
        # for images, heights_gt in test_dl:
        for query_i, (images,heights_gt) in enumerate(test_dl):
            images = images.to(device)
            heights_gt = torch.tensor(heights_gt).to(device)
            # with torch.autocast(device):
            heights_pred = model(images)
            # NOTE 归一化后加入
            # heights_gt = scale_up(heights_gt)
            # heights_pred = scale_up(heights_pred)

            distances = abs(heights_pred - heights_gt)
            query_i_info = {'query_i':query_i, 'heights_gt':heights_gt, 'heights_pred':heights_pred, 'distances':distances}
            inference_info_collect.append(query_i_info)
            
            tqdm_bar.set_description(f"{query_i:5d}")
            _ = tqdm_bar.refresh()
            _ = tqdm_bar.update()

        inference_info_collect = sorted(inference_info_collect, key=lambda info_dict: info_dict['heights_gt'].item())

        distances_list = torch.Tensor([info_dict['distances'] for info_dict in inference_info_collect])
        
        average_distances = distances_list.mean().item()

        statistical_results = []

        last_height = 0
        equal_height_dustbin = []

        for idx in range(test_img_num):
        # for info_dict in inference_info_collect:
            info_dict = inference_info_collect[idx]
            h_gt = round(info_dict['heights_gt'].item(), 2)
            if (h_gt != last_height and idx != 0) or idx == (test_img_num-1):
                equal_height_dustbin = torch.Tensor(equal_height_dustbin)
                distances_array = equal_height_dustbin[:,3]
                distances_max = distances_array.max().item()
                distances_min = distances_array.min().item()
                distances_mean = distances_array.mean().item()
                distances_median = distances_array.median().item()
                distances_std = distances_array.std().item()
                statistical_results.append([last_height, distances_mean, distances_median, distances_max, distances_min, distances_std])
                equal_height_dustbin = [[h_gt, info_dict['query_i'], info_dict['heights_gt'].item(), info_dict['distances'].item()]]
                last_height = h_gt
            elif equal_height_dustbin == []:

                equal_height_dustbin = [[h_gt, info_dict['query_i'], info_dict['heights_gt'].item(), info_dict['distances'].item()]]
                last_height = h_gt
            else:
                equal_height_dustbin.append([h_gt, info_dict['query_i'], info_dict['heights_gt'].item(), info_dict['distances'].item()])

        # statistical_results = np.array(statistical_results)
        statistical_results = list(enumerate(statistical_results, start=1))
    

    test_result_str = f'average estimation error: {average_distances:.2f}'
    statistical_results_str = f'estimation error statistical results:\n' + ', \n'.join(f'{result[0]:2d}. On height {result[1][0]:.2f}m, average: {result[1][1]:.2f}, median: {result[1][2]:.2f}, max: {result[1][3]:.2f}, min: {result[1][4]:.2f}, std: {result[1][5]:.2f}' for result in statistical_results)

    return test_result_str, statistical_results_str