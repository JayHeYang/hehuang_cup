
from unicodedata import name
import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader


from utils.config import Config
from models.my_net import EfficientNet_B2, EfficientNet_B3, SEResNet50, ResNet101
from datasets.dataset import PAR_Train_Dataset, PAR_Val_Dataset, PAR_Test_Dataset
from models.other_model import HR_Net



def train_func(model_name):
    # models = [EfficientNet_B2(), EfficientNet_B3(), SEResNet50()]
    if model_name =='resnet101':
        models = [ResNet101()]
    elif model_name == 'b3':
        models = [EfficientNet_B3()]
    else:
        models = [SEResNet50()]
    opt = Config()
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


    train_data = PAR_Train_Dataset(opt.train_trans, opt.train_roots)
    train_dataloader = DataLoader(train_data, opt.bs, shuffle=True, num_workers=opt.nw, drop_last=True)

    # criterions = [nn.BCEWithLogitsLoss(), nn.L1Loss(), nn.KLDivLoss()] # 损失集合
    criterions = [nn.BCEWithLogitsLoss(), nn.L1Loss()]
    # 先冻结参数
    for model in models:
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = True

    optimizers = [] # 优化器集合
    for model in models:
        optimizer = t.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=opt.wd)
        # optimizer = t.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.wd)
        optimizers.append(optimizer)
    
    lr_schedules = [] # 学习率调整器集合
    for optimizer in optimizers:
        lr_schedule = t.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_epoch, gamma=opt.gamma)
        # lr_schedule = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

        lr_schedules.append(lr_schedule)

    # 设置训练模式
    for model in models:
        model.train()
        model.to(device)
    
    for criterion in criterions:
        criterion.to(device)
   
    # 是否加预训练模型
    if opt.pre_train:
        pass

    # training
    step = 0
    for epoch in tqdm.tqdm(range(opt.max_epoch)):
        # 20个epoch之后解冻
        if epoch+1 == 20:
            for model in models:
                for name, param in model.named_parameters():
                    if 'backbone' in name:
                        param.requires_grad = True

        for ii, (data, label) in enumerate(train_dataloader):

            input = data.to(device)
            target =label.to(device)
   
            for optimizer in optimizers:
                optimizer.zero_grad()


            hard_score, soft_score  = [], []
            for kk, model in enumerate(models):
                t_hard_score, t_soft_score = model(input)
                if kk == 0:
                    bceloss = criterions[0](t_hard_score, target[:, :10])
                    l1loss = criterions[1](t_soft_score, target[:, 10:])
                    
                else:
                    bceloss += criterions[0](t_hard_score, target[:, :10])
                    l1loss += criterions[1](t_soft_score, target[:, 10:])
            
                hard_score.append(t_hard_score)
                soft_score.append(t_soft_score)
            
            # 计算klloss
            klloss = t.tensor(0.0).cuda()
            for ii in range(len(models)):
                for jj in range(ii+1, len(models)):
                    klloss += criterions[2](t.log_softmax(hard_score[ii], dim=1), t.softmax(hard_score[jj], dim=1)) * 5

            loss = bceloss + l1loss + klloss
                
            
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
        
            step += 1
            if step % 20 == 0:
        # print("\nEpoch:{}, Accuracy:{}".format(epoch + 1, acc))
                print("\n Epoch:{}, Loss:{}, BCELoss: {}, L1Loss: {}, KLLoss: {}"
                .format(epoch+1, loss.data.cpu().detach().numpy(),
                        bceloss.data.cpu().detach().numpy(),
                        l1loss.data.cpu().detach().numpy(),
                        klloss.data.cpu().detach().numpy()))
                
        lr_schedule.step()
        if (epoch + 1) % opt.save_freq == 0:
            all_models = dict()
            for ii, model in enumerate(models):
                all_models.update({ii: model.state_dict()})

            t.save(all_models, 'model_{}_{}.pth'.format(model_name, epoch+1))


if __name__ == '__main__':
    for ii, model_name in enumerate(['resnet101']):
        train_func(model_name)
