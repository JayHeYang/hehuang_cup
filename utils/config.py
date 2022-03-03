
import torchvision.transforms as T


class Config(object):

    nw = 0 # 多线程加载数据所用的线程数
    # bs = 32  # batch_size
    bs = 4  # for test
    wd = 0.005
    lr = 0.001 # learning rate
    max_epoch = 10 # max epoch

    decay_epoch = 4  # 学习率衰减频率 
    gamma = 0.5
    models = ['My_Net', 'ResNet34']
    save_freq = 10 # 每10个epoch保存一次

    # 路径相关
    # train_roots = "train/train1A.csv"
    train_roots = 'train_new/train/train1A_new.csv'
    # val_roots = "train/train1A.csv"
    val_roots = 'train_new/train/train1A_new.csv'
    test_roots = 'testA'
    
    val_save_dir = 'val_results'
    test_save_dir = 'results'

    train_trans = T.Compose([
            T.RandomHorizontalFlip(),
            # T.Resize([int(256*1.1), int(192*1.1)]),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10), # 使用数据增强
            T.Resize((256, 192)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    test_trans = T.Compose([
            T.Resize([256, 192]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
    

    pre_train = False # 是否加载训练好的模型
    pre_train_path = ['model_b2_10.pth', 'model_b3_10.pth', 'model_se_10.pth']
    # pre_train_path = ['model_resnet101_10.pth']
    
    

# import torch.nn as nn
# import torch.optim as optim
# model = nn.Linear(10, 2)
# optimizer = optim.SGD(model.parameters(), lr=1.)
# steps = 10
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

# for epoch in range(5):
#     for idx in range(steps):
#         scheduler.step()
#         print(scheduler.get_lr())