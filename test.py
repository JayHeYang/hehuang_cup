
from tempfile import tempdir
import torch as t
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.config import Config
from models.my_net import EfficientNet_B2, EfficientNet_B3, SEResNet50, ResNet101
from models.other_model import HR_Net
from datasets.dataset import PAR_Train_Dataset, PAR_Val_Dataset, PAR_Test_Dataset


def save_resluts(img_names, hard_results, soft_results):
    all_name = ['name', 'upperLength', 'clothesStyles', 'hairStyles', 'upperBlack',
       'upperBrown', 'upperBlue', 'upperGreen', 'upperGray', 'upperOrange',
       'upperPink', 'upperPurple', 'upperRed', 'upperWhite', 'upperYellow']

    result = dict()

    length_map = np.array(['NoSleeve', 'ShortSleeve', 'LongSleeve'])
    length_idx = np.argmax(hard_results[:, :3], axis=1)
    print(length_idx[:10])
    length = length_map[length_idx]
     
    cloth_map = np.array(['lattice', 'multicolour', 'Solidcolor'])
    cloth_idx = np.argmax(hard_results[:, 3:6], axis=1)
    cloth = cloth_map[cloth_idx]

    hair_map = np.array(['Short', 'Long', 'middle', 'Bald'])
    hair_map = np.array(['Bald', 'middle', 'Long', 'Short'])
    hair_idx = np.argmax(hard_results[:, 6:], axis=1)
    hair = hair_map[hair_idx]

    soft_results = np.around(soft_results, 2)
    # print(soft_results[1])

    for ii, name in enumerate(all_name):
        if ii == 0:
            result[name] = img_names
        elif ii == 1:
            result[name] = length
        elif ii == 2:
            result[name] = cloth
        elif ii == 3:
            result[name] = hair
        else:
            result[name] = soft_results[:, ii-4]
    
    # print(result.keys())

    df = pd.DataFrame(result, index=None)
    print(df.head())
  
    df.to_csv('results/result_aug_mix_resample_v2.csv', index=None)

def vote_func(hard_list, soft_list):
    opt  = Config()

    final_hard_score, final_soft_score = torch.zeros(opt.bs, 10).cuda(), torch.zeros(opt.bs, 11).cuda()
    
    ## 第一个属性：
    # 获取三个模型分别的预测属性
    a = torch.cat([torch.argmax(hard_list[0][:, :3], dim=1).view(-1, 1), 
                         torch.argmax(hard_list[1][:, :3], dim=1).view(-1, 1), 
                         torch.argmax(hard_list[2][:, :3], dim=1).view(-1, 1)], dim=1)
    # 统计属性的得票, 得到最高票的索引
    b = torch.argmax(torch.cat([torch.sum(a ==0, dim=1).view(-1, 1), 
                     torch.sum(a ==1, dim=1).view(-1, 1), 
                     torch.sum(a ==2, dim=1).view(-1, 1)], dim=1), dim=1)
    # print(final_hard_score.shape, b.shape)
    final_hard_score[:, :3] = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[b]

    ## 第二个属性：
    # 获取三个模型分别的预测属性
    a = torch.cat([torch.argmax(hard_list[0][:, 3:6], dim=1).view(-1, 1), 
                   torch.argmax(hard_list[1][:, 3:6], dim=1).view(-1, 1), 
                   torch.argmax(hard_list[2][:, 3:6], dim=1).view(-1, 1)], dim=1)
    # 统计属性的得票, 得到最高票的索引
    b = torch.argmax(torch.cat([torch.sum(a ==0, dim=1).view(-1, 1), 
                     torch.sum(a ==1, dim=1).view(-1, 1), 
                     torch.sum(a ==2, dim=1).view(-1, 1)], dim=1), dim=1)
    final_hard_score[:, 3:6] = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[b]    

    ## 第三个属性：
    # 获取三个模型分别的预测属性
    a = torch.cat([torch.argmax(hard_list[0][:, 6:], dim=1).view(-1, 1), 
                   torch.argmax(hard_list[1][:, 6:], dim=1).view(-1, 1), 
                   torch.argmax(hard_list[2][:, 6:], dim=1).view(-1, 1)], dim=1)
    # 统计属性的得票, 得到最高票的索引
    b = torch.argmax(torch.cat([torch.sum(a ==0, dim=1).view(-1, 1), 
                     torch.sum(a ==1, dim=1).view(-1, 1), 
                     torch.sum(a ==2, dim=1).view(-1, 1),
                     torch.sum(a ==3, dim=1).view(-1, 1)], dim=1), dim=1)
    # print(a)
    final_hard_score[:, 6:] = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], 
                                             [0, 0, 1, 0], [0, 0, 0, 1]])[b]
    for ii in range(3):
        final_soft_score += soft_list[ii]
    # print(final_hard_score)
    return final_hard_score, final_soft_score / 3


def test():
    models = [EfficientNet_B2(), EfficientNet_B3(), SEResNet50()]
    # models = [ResNet101()]
    opt = Config()
    # 记录每个epoch产生的指标信息
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    opt = Config()

    # model2 = ResNet34()
    
    for ii, model in enumerate(models):
        model.to(device)
        model.load_state_dict(t.load(opt.pre_train_path[ii])[0])
        model.eval()


    test_data = PAR_Test_Dataset(opt.test_trans, opt.test_roots)
    val_loader = DataLoader(test_data, batch_size=opt.bs, shuffle=False,
                                 num_workers=opt.nw)

    hard_results = np.array([]) 
    soft_results = np.array([])

    img_names = []
    
    for i, (data, img_name) in enumerate(val_loader):
        data = data.to(device)
        temp_hard, temp_soft = [], []
        for model in models:
            hard_score, soft_score = model(data)
            temp_hard.append(hard_score), temp_soft.append(soft_score)
        
        hard_score, soft_score = vote_func(temp_hard, temp_soft) # 单模型时不需要
       

        if i == 0:
            hard_results = hard_score.data.cpu().numpy()
            soft_results = soft_score.data.cpu().numpy()  
        else:
            hard_results = np.vstack((hard_results, hard_score.data.cpu().numpy()))
            soft_results = np.vstack((soft_results, soft_score.data.cpu().numpy()))
        img_names.extend(img_name)
    save_resluts(img_names, hard_results, soft_results)


if __name__ == '__main__':
    test()