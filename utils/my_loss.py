import torch.nn as nn
import torch as t
import torch


class My_Loss(nn.Module):
    def __init__(self):
        super(My_Loss, self).__init__()

        # 35个PETA行人属性的权重
        self.weights = torch.Tensor([0.750,
                                     0.247,
                                     0.727,
                                     0.246,
                                     0.00225,
                                     
                                        ])
        if t.cuda.is_available():
            self.weights = self.weights.cuda()
        self.EPS = 1e-12


    def forward(self, Score, Target):

        # self.weights = weights
        EPS = self.EPS
        Score = t.sigmoid(Score)

        cur_weights = torch.exp(Target + (1 - Target * 2) * self.weights)
        loss = cur_weights * (Target * torch.log(Score + EPS)) + ((1 - Target) * torch.log(1 - Score + EPS))

        return torch.neg(torch.mean(loss))




if __name__ == '__main__':

    w = t.ones(35)

    criterion1 = My_Loss()
    criterion2 = nn.BCEWithLogitsLoss()

    score = t.randn(10, 35)
    target = t.ones(10, 35)

    # 随机设置一些索引为0
    index = t.randn(10, 35) < 1
    target[index] = 0

    print(criterion1(score.cuda(), target.cuda()))
    print(criterion2(score, target))
