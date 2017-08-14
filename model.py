import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from args import get_opt
import h5py
import os

opt = get_opt()

std = 0.01


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class dataset_mnist:
    def __init__(self):
        filename = 'data.hdf5'
        file = os.path.join(os.getcwd(), filename)

        try:
            self.data = h5py.File(file, 'r+')
        except:
            raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
        print("Reading Done: %s", file)


class G(nn.Module):
    def __init__(self, d=16):
        super(G, self).__init__()
        self.deconv0 = nn.ConvTranspose2d(10, 100, 1, 1, 0)
        self.deconv0_bn = nn.BatchNorm2d(100)
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)

        self.weight_init(mean=0.0, std=0.02)

    def forward(self, input):
        x = F.relu(self.deconv0_bn(self.deconv0(input)))
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(x)))
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)))
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)))
        x = F.sigmoid(self.deconv4(x))
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)



class lap_G(nn.Module):
    def __init__(self):
        super(lap_G, self).__init__()

        self.bilinear_deconv1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(10, 100, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(100)

        self.bilinear_deconv2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Conv2d(100, 50, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(50)

        self.bilinear_deconv3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3 = nn.Conv2d(50, 25, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(25)

        self.bilinear_deconv4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4 = nn.Conv2d(25, 6, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(6)

        self.bilinear_deconv5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv5 = nn.Conv2d(6, 1, 3, 1, 1)

        # self.weight_init(mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, std)
                m.bias.data.zero_()

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.bn1((self.conv1(self.bilinear_deconv1(input)))))
        x = F.leaky_relu(self.bn2((self.conv2(self.bilinear_deconv2(x)))))
        x = F.leaky_relu(self.bn3((self.conv3(self.bilinear_deconv3(x)))))
        x = F.leaky_relu(self.bn4((self.conv4(self.bilinear_deconv4(x)))))
        x = F.tanh(self.conv5(self.bilinear_deconv5(x)))
        return x


'''
shape Test
'''

if __name__ == "__main__":
    batch = opt.batch_size
    x_dim = opt.x_dim
    z_dim = opt.z_dim
    z = torch.zeros([batch, z_dim])
    z = z.view([-1, z_dim, 1, 1])
    z = Variable(z)

    x = torch.zeros([batch, 1, x_dim, x_dim])
    x = Variable(x)

    # g = lap_G()
    g = G()
    out = g.print_forward(z)

    r1 = out.data.numpy()

    print(r1.shape)
