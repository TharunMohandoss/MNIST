import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.custom_layers import EqualizedConv2d, EqualizedLinear, MinibatchStdDev

#in,out,kernel_size,stride,padding
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.no_of_downscales = 3

        self.from_rgb = EqualizedConv2d(1,25,1,0)

        self.minibatch_stddev = MinibatchStdDev()

        self.conv1 = EqualizedConv2d(26,25,3,1)
        self.conv2 = EqualizedConv2d(25,25,3,1)

        self.avg_pool = nn.AvgPool2d(2, stride=2)

        self.conv_list = nn.ModuleList([])
        in_channels = [27,50,52,50]
        out_channels = [50,50,50,50]

        for i in range(self.no_of_downscales-1):
            conv_1 = EqualizedConv2d(in_channels[2*i],out_channels[2*i],3,1)
            conv_2 = EqualizedConv2d(in_channels[2*i+1],out_channels[2*i+1],3,1)
            self.conv_list.append(conv_1)
            self.conv_list.append(conv_2)

        self.conv_last3x3 = EqualizedConv2d(52,50,3,1)

        #bring down to 512
        self.fc1 = EqualizedLinear(50*4*4,50)

        #linear
        self.fc2 = EqualizedLinear(50,1)

        #linear
        self.fc3 = EqualizedLinear(50,10)

        #leaky relu
        self.lr   = nn.LeakyReLU(negative_slope=0.2)


    def forward(self, img_list,classifier_only=False): 
        img_list = list(reversed(img_list))
        img = img_list[0]
        # print('img.shape : ',img.shape)
        # for x in img_list:
        #     print('x.shape : ',x.shape)

        x = self.from_rgb(img)

        x = self.minibatch_stddev(x)
        x = self.lr(self.conv1(x))
        x = self.lr(self.conv2(x))
        x = self.avg_pool(x)

        for i in range(self.no_of_downscales-1):
            # print(x.shape)
            # print(img_list[i+1].shape)
            x = torch.cat((x,img_list[i+1]),1)
            x = self.minibatch_stddev(x)
            x = self.lr(self.conv_list[2*i](x))
            x = self.lr(self.conv_list[2*i+1](x))
            x = self.avg_pool(x)

        x = torch.cat((x,img_list[-1]),1)
        x = self.minibatch_stddev(x)
        x = self.lr(self.conv_last3x3(x))

        x = torch.reshape(x,(x.shape[0],50*4*4))
        x = self.lr(self.fc1(x))

        z = self.fc2(x)
        y = self.fc3(x)

        if not classifier_only:
        	return z,y
        else:
        	return z










