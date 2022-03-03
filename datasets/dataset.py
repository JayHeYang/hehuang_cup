import glob
import os
import sys
from PIL import Image
from sklearn import datasets
import torch
from torch.utils import data
import pandas as pd
import numpy as np
import random


# import sys
# sys.path.append('/home/hjy/hehuang_cup/pedestrian_attribute_recognition')  # 添加路径
from utils.config import Config


def generate_label(row_datas):
    label = list()
    # print(row_datas)

    length_name = {'LongSleeve':[0, 0, 1], 'ShortSleeve':[0 ,1, 0], 'NoSleeve':[1, 0, 0]}
    cloth_s_name = {'Solidcolor':[0, 0, 1], 'multicolour':[0, 1, 0], 'lattice':[1, 0, 0]}
    hair_s_name = {'Short':[0 ,0, 0, 1], 'Long':[0 ,0, 1, 0], 'middle':[0 ,1, 0, 0], 'Bald':[1 ,0, 0, 0]}

    label.extend(length_name[row_datas[0]])
    label.extend(cloth_s_name[row_datas[1]])
    label.extend(hair_s_name[row_datas[2]])
    label.extend(row_datas[3:])

    return label


def getRandomIndex(n=4000, x=1000):
	# 索引范围为[0, n), 随机选x个不重复
    random.seed(0) # 设置随机数种子
    test_index = random.sample(range(n), x)

    test_index = np.array(test_index)
    # 再将test_index从总的index中减去就得到了train_index
    train_index = np.delete(np.arange(n), test_index)
    return train_index.tolist(), test_index.tolist()


class PAR_Train_Dataset(data.Dataset):

    def __init__(self, transforms, roots):

        self.roots = roots
    
        # train_idx, _ = getRandomIndex()
        
        self.labels = pd.read_csv(self.roots).fillna(.0)
        # self.labels = pd.DataFrame(self.labels.iloc[train_idx])
        # self.labels = self.labels.reset_index(drop=True) # 删除原来的索引重排
       
        self.transforms = transforms
       
    def __getitem__(self, index):

        img_path = os.path.join(self.roots.split('.')[0], self.labels['name'][index])

        label = generate_label(self.labels.iloc[index, 1:].to_list())
        label = torch.from_numpy(np.array(label, dtype=np.float32))
        image = Image.open(img_path)
        image = self.transforms(image)
    
        return image, label

    def __len__(self):
        return len(self.labels['name'])



class PAR_Val_Dataset(data.Dataset):

    def __init__(self, transforms, roots):

        self.roots = roots

        _, test_idx = getRandomIndex()
        self.labels = pd.read_csv(self.roots).fillna(.0)
        self.labels = pd.DataFrame(self.labels.iloc[test_idx])
        self.labels = self.labels.reset_index(drop=True) # 删除原来的索引重排
    
        self.transforms = transforms

    def __getitem__(self, index):
        image_name = self.labels['name'][index]
        img_path = os.path.join(self.roots.split('.')[0], self.labels['name'][index])
        image = Image.open(img_path)
        image = self.transforms(image)
        return image, image_name

    def __len__(self):
        return len(self.labels['name'])



class PAR_Test_Dataset(data.Dataset):

    def __init__(self, transforms, roots):

        self.imgs = glob.glob(roots + '/*.jpg')
        self.transforms = transforms

    def __getitem__(self, index):

        image_name = os.path.split(self.imgs[index])[1]
        image = Image.open(self.imgs[index])
        image = self.transforms(image)
        return image, image_name

    def __len__(self):
        return len(self.imgs)




# opt = Config()
# datasets = PAR_Test_Dataset(opt.test_trans, opt.test_roots)
# print(datasets.__getitem__(0))
# print(datasets.__len__())
