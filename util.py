import torch
from torch.autograd import Variable

from time import localtime, strftime
import os
from args import get_opt
import numpy as np


opt = get_opt()

def z_init():
    pass


def pca_feature(X, d):
    X = X / 255.
    from sklearn.decomposition import PCA
    X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
    pca = PCA(n_components=d)
    return pca.fit_transform(X)

def setup():

    model_path = opt.model_path

    # check if dir exists
    if not os.path.isdir(opt.sample_path):
        os.mkdir(opt.sample_path)

    if not os.path.isdir(opt.model_path):
        os.mkdir(opt.model_path)

    # mkdir folder with name by date,time
    folder = strftime("%y-%m-%d %H-%M-%S", localtime())
    sample_path = os.path.join(os.getcwd(), 'samples', folder)
    os.mkdir(sample_path)

    # comment text write
    f = open(os.path.join(sample_path, '1-comment.txt'), 'w')
    comment = (""" experimental comment here """)
    for s in comment:
        f.write(s)
    f.close()

    # comment arg info write
    f = open(os.path.join(sample_path, '1-info.txt'), 'w')
    tuples = vars(opt).items()
    for x in tuples:
        f.write(str(x))
        f.write('\n')
    f.close()

    return sample_path, model_path

def to_variable(x, requires_grad=False):
    if opt.gpu:
        x = x.cuda()

    if requires_grad:
        return Variable(x, requires_grad=requires_grad)
    return Variable(x)
