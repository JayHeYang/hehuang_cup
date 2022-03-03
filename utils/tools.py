"""
一些工函数
"""
import numpy as np

import torch as t
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt




def compute_batch_attributes_weights(Target):

    counts = t.sum(Target, dim=0)

    N = Target.size()[0] # batchsize 的大小
    zero_idx = counts == 0
    counts[zero_idx] = 1

    weights = counts / N

    return weights


import random

def getRandomIndex(n=4000, x=1000):
	# 索引范围为[0, n), 随机选x个不重复
    random.seed(0) # 设置随机数种子
    test_index = random.sample(range(n), x)

    test_index = np.array(test_index)
    # 再将test_index从总的index中减去就得到了train_index
    train_index = np.delete(np.arange(n), test_index)
    return train_index.tolist(), test_index.tolist()

    
def compute_macro_f1():
    """
    根据result.csv文件计算MacroF1分数
    """
    # 
    predict = pd.read_csv('results/result_myresnet.csv')
    target = pd.read_csv('C:/code/hehuang_cup/train/train1A.csv').fillna(.0)
    _, test_idx = getRandomIndex()
    target = pd.DataFrame(target.iloc[test_idx])
    target = target.reset_index(drop=True) # 删除原来的索引重排
    

    # print(predict.head())
    # print(target.head())
    
    pred_info = {'LongSleeve': 1e-3, 'ShortSleeve':  1e-3, 'NoSleeve': 1e-3, 
    'Solidcolor':  1e-3, 'multicolour':  1e-3, 'lattice':  1e-3, 
    'Short':  1e-3, 'Long': 1e-3, 'middle': 1e-3, 'Bald': 1e-3} # 统计各个硬标签属性的数量
    targ_info = {'LongSleeve': 1e-3, 'ShortSleeve':  1e-3, 'NoSleeve': 1e-3, 
    'Solidcolor':  1e-3, 'multicolour':  1e-3, 'lattice':  1e-3, 
    'Short':  1e-3, 'Long': 1e-3, 'middle': 1e-3, 'Bald': 1e-3}

    for col_name in predict.columns.tolist()[:4]:
        if col_name == 'name':
            continue
        elif col_name in ['upperLength', 'clothesStyles', 'hairStyles']:
            pred_info.update(predict[col_name].value_counts().to_dict())
            targ_info.update(target[col_name].value_counts().to_dict())
        else:
            pass
    
    acc_count = {'LongSleeve': 1e-3, 'ShortSleeve':  1e-3, 'NoSleeve': 1e-3, 
    'Solidcolor':  1e-3, 'multicolour':  1e-3, 'lattice':  1e-3, 
    'Short':  1e-3, 'Long': 1e-3, 'middle': 1e-3, 'Bald': 1e-3}

    for col_name in predict.columns.tolist()[1:4]:
        for ii in range(len(target[col_name])):
            target_label = target[col_name][ii]
            pred_label = predict[col_name][ii]

            if target_label == pred_label:
                acc_count[target_label] = acc_count.get(target_label, 0) + 1
    

    
    
    F1 = 0.0
    count = 0
    for k, v in acc_count.items():
        if  pred_info[k] == 1e-3 or targ_info[k] == 1e-3:
            continue
        p =  v / pred_info[k]
        r = v / targ_info[k]
        F1 += 2 * p * r / (p + r)
        count += 1
        # print(F1)
    
    F1 /= count
    # print(count)
    # print(pred_info)
    # print(targ_info)
    # print(acc_count)
    print('marcoF1:', F1)


    # 获取软标签
    soft_pred = t.from_numpy(predict.iloc[:, 4:].to_numpy())
    soft_targ = t.from_numpy(target.iloc[:, 4:].to_numpy())
    loss = F.l1_loss(soft_pred, soft_targ, reduction='sum') / soft_targ.size(0)

    print(loss.data)
    


if __name__ == '__main__':
    compute_macro_f1()
    # score = np.random.randn(10, 8)
    # a = [0, 1, 1, 0, 0, 1, 0, 1]
    # b = [1, 1, 1, 0, 1, 0, 1, 1]
    # target = np.array([a, b, b, a, a, b, a, b, a, b]).reshape(10, 8)
    #
    # print(PR_curve(score, target, sigmoid=True))
    # compute_attributes_weights()
