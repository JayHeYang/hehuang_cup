from pyexpat import model
from select import select
from turtle import forward
from numpy import reshape
import torchvision as tv
import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x)


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self, x):
        return x.view((x.size(0), x.size(1)))


class HR_Net(nn.Module):
	def __init__(self):
		super(HR_Net, self).__init__()
		googlenet = tv.models.googlenet()

		# 预处理卷积
		self.pre_layers = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            BasicConv2d(64, 64, kernel_size=1, stride=1),

            BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

        )
		
		# low_level layer
		self.low_layer = nn.Sequential(
			googlenet.inception3a,
			googlenet.inception3b,
			nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
			googlenet.inception4a,
		)

		self.low_pred = nn.Sequential(
			BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1),
			nn.AdaptiveAvgPool2d((1, 1)),
			Reshape(),
			nn.Linear(256, 256),
			nn.ReLU(inplace=True),
			# nn.Dropout(0.3),
			nn.Linear(256, 11),
			# nn.Sigmoid(),

		)


		self.mid_layer = nn.Sequential(
			googlenet.inception4b,
			googlenet.inception4c,
			googlenet.inception4d,
		)

		self.mid_pred = nn.Sequential(
			BasicConv2d(528, 256, kernel_size=3, stride=1, padding=1),
			nn.AdaptiveAvgPool2d((1, 1)),
			Reshape(),
			nn.Linear(256, 256),
			nn.ReLU(inplace=True),
			# nn.Dropout(0.3),
			nn.Linear(256, 4),
			# nn.Sigmoid()
		)
	
		self.high_layer = nn.Sequential(
			googlenet.inception4e,
			nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
			googlenet.inception5a,
			googlenet.inception5b
		)

		self.high_pred = nn.Sequential(
			BasicConv2d(1024, 256, kernel_size=3, stride=1, padding=1),
			nn.AdaptiveAvgPool2d((1, 1)),
			Reshape(),
			nn.Linear(256, 256),
			nn.ReLU(inplace=True),
			# nn.Dropout(0.3),
			nn.Linear(256, 6),
			# nn.Sigmoid()
		)

		self.last_layer = nn.Sequential(
			nn.Linear(15, 256),
			nn.ReLU(inplace=True),
			# nn.Dropout(0.3),
			nn.Linear(256, 6),
			# nn.Sigmoid()

		)

	def forward(self, x):
		x = self.pre_layers(x)
		x = self.low_layer(x)
		low_fea = x

		x = self.mid_layer(x)
		mid_fea = x

		x = self.high_layer(x)
		high_fea = x

		low_pred = self.low_pred(low_fea)
		mid_pred = self.mid_pred(mid_fea)
		high_pred = self.high_pred(high_fea)

		recons_fea = torch.cat([low_pred, mid_pred], dim=1)
		weights = self.last_layer(recons_fea)
		high_pred = weights * high_pred

		hard_label = torch.cat([high_pred, mid_pred], dim=1)
		soft_label = F.softmax(low_pred, dim=1)
		return hard_label, soft_label



from torchsummary import summary
# model = HR_Net().cuda()

# reshape = Reshape(1).cuda()

# print(tv.models.vgg11_bn())

# fake_data = torch.randn(10, 1, 1, 1).cuda()
# print(reshape(fake_data).size())
# print(model(fake_data).size())
# print(summary(model, (3, 160, 75)))