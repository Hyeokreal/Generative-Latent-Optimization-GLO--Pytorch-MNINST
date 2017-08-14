import torchvision.datasets as dataset
# from data_loader import MNIST
import torchvision.transforms as transforms
import torch
import torchvision
import os
from torch import optim
from torch.autograd import Variable
from model import G, lap_G
from args import get_opt, _print
from util import setup, pca_feature, to_variable
import numpy as np

opt = get_opt()
_print()

# hyper parameters
n_epochs = opt.epochs
batch_size = opt.batch_size
z_dim = opt.z_dim
x_dim = opt.x_dim
sample_size = opt.sample_size
lr = opt.lr
log_step = 100
sample_step = 500

sample_path, model_path = setup()

img_size = opt.x_dim
transform = transforms.Compose([
    transforms.Scale(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = dataset.MNIST(root='./data/',
                              train=True,
                              transform=transform,
                              download=True)

# choose generator type laplacian generator or common dcgan generator
if opt.gen_type == "LAPGAN":
    generator = lap_G()
else:
    generator = G()

if opt.gpu:
    generator.cuda()

g_optimizer = optim.Adam(generator.parameters(), lr)

# initialize z with pca
data_for_z = train_dataset.train_data.clone()
z = pca_feature(data_for_z.numpy(), 10)

for i in range(z.shape[0]):
    z[i] = z[i, :] / np.linalg.norm(z[i, :], 2)

z_in_ball = torch.FloatTensor(z).view(-1, batch_size, z_dim, 1, 1)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

if opt.gpu:
    learnable_z = z_in_ball.view(-1, batch_size, z_dim, 1, 1).cuda()
else:
    learnable_z = z_in_ball.view(-1, batch_size, z_dim, 1, 1)

total_step = len(train_loader)

for epoch in range(n_epochs):
    for i, (x, _) in enumerate(train_loader):
        x = to_variable(x)
        z = to_variable(learnable_z[i], requires_grad=True)
        x_hat = generator.forward(z)

        # use L1 loss
        loss = torch.mean(torch.abs(x - x_hat))

        g_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()

        if opt.gpu:
            grad = z.grad.data.cuda()
        else:
            grad = z.grad.data

        z_update = learnable_z[i] - 5 * grad
        z_update = z_update.cpu().numpy()
        norm = np.sqrt(np.sum(z_update ** 2, axis=1))
        z_update_norm = z_update / norm[:, np.newaxis]

        if opt.gpu:
            learnable_z[i] = torch.from_numpy(z_update_norm).cuda()
        else:
            learnable_z[i] = torch.from_numpy(z_update_norm).cpu()

        if (i + 1) % log_step == 0:
            print('Epoch [%d/%d], Step[%d/%d], loss: %f'
                  % (epoch + 1, n_epochs, i + 1, total_step, loss.data[0]
                     ))

        # save the real images
        if (i + 1) % sample_step == 0:
            torchvision.utils.save_image(x.data,
                                         os.path.join(sample_path,
                                                      'real_samples-%d-%d.png' % (
                                                          epoch + 1, i + 1)), nrow=5)
        # save the generated images
        if (i + 1) % sample_step == 0:
            torchvision.utils.save_image(x_hat.data,
                                         os.path.join(sample_path,
                                                      'fake_samples-%d-%d.png' % (
                                                          epoch + 1, i + 1)), nrow=5)

    g_path = os.path.join(model_path, 'generator-%d.pkl' % (epoch + 1))
    z_path = os.path.join(model_path, 'latent-%d.pkl' % (epoch + 1))
    torch.save(learnable_z, z_path)
    torch.save(generator.state_dict(), g_path)
