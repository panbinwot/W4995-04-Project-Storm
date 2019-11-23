import os
import numpy as np
import _pickle as pickle

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models

import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm_notebook
from BatchLoader import BatchLoader
from storm import Storm
from train import train
from storm import Storm
from helper import read_data, plot

def main():
    if not os.path.exists('./data/'):
        wget_cmd = 'wget https://s3-eu-west-1.amazonaws.com/bsopenbucket/e6040/data.zip'
        unzip_cmd = 'unzip data.zip'
        os.system(wget_cmd)
        os.system(unzip_cmd)
    images, labels = read_data('data/cifar10')

    batch_size = 64

    print('Test data set shape:', images['test'].shape)
    data_train = BatchLoader(np.moveaxis(images['train'], 3, 1), labels['train'])
    data_test = BatchLoader(np.moveaxis(images['test'], 3, 1), labels['test'])
    trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss().cuda()

    def search(low, high, epsilon, call_func):
        results = [(call_func(param), param) for param in tqdm_notebook(range(low, high, epsilon))]
        return min(results, key=lambda x: x[0])[1]

    def call_storm(c):
        net = models.resnet18(pretrained=False).cuda()
        optim = Storm(net.parameters(), c=c)
        res = train(trainloader, testloader, net, optim, criterion, epochs=1, max_iter_per_epoch=500,
                    test_freq=10, no_test=True, verbose=False)
        return sum(res[0]) / len(res[0])

    best_c = search(0, 1000, 5, call_storm)
    best_c = 70
    print('best c is: ', best_c)

    net = models.resnet18(pretrained=False).cuda()
    opts = [Storm(net.parameters(), c= best_c), torch.optim.Adam(net.parameters()), torch.optim.Adagrad(net.parameters())]
    file_name = ['./res_Storm_64', './res_Adam_64', './res_Adagrad_64']
    
    for i, optim in enumerate(opts):
        res = train(trainloader, testloader, net, optim, criterion, epochs=10)
        np.save(file_name[i], res)

    optim_results = []
    optim_labels = ["Adam", "Adagrad", "STORM"]
    for name in file_name:
        name += '.npy'
        optim_results.append(np.load(name, allow_pickle = True))

    plot(optim_results, optim_labels, 0, "Loss")
    plot(optim_results, optim_labels, 1, "Train_Accuracy")
    plot(optim_results, optim_labels, 2, "Test_Accuracy")

if __name__ == "__main__":
    main()