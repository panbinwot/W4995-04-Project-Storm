import os
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# the followings are self-implemented
from LoadBatch import LoadBatch
from helper import *

def train_test(trainloader, testloader, net, optimizer, criterion, epochs):
  train_loss_lst = []
  train_acc_lst = []
  test_acc_lst = []
  i = 0
  for epoch in range(epochs):
    
    for i,data in enumerate(trainloader, 0):
        net.train()s
        batch_x = data[0]
        batch_y = data[1]
        optimizer.zero_grad()
        output = net(batch_x)
        loss = criterion(output, batch_y.long())
        train_loss_lst.append(loss)
        loss.backward()
        optimizer.step()

      # Evaluate at this batch

        net.eval()
        with torch.no_grad():
            optimizer.zero_grad()
            outputs = net(batch_x)
            _ , predicted = torch.max(outputs.data, 1)

            train_pred_batch, train_target_batch, = predicted.cpu().detach().numpy(),data[1].cpu().detach().numpy()

            train_acc = np.mean(train_pred_batch.astype(np.int)==train_target_batch )
            train_acc_lst.append(train_acc)

      # Test the function on the set
        test_acc = test(testloader, net, optimizer, criterion)
        test_acc_lst.append( test_acc)
    
        print("Epoch: {}, Batch: {}, Train Accuracy: {}, Test Accuracy: {}".format(epoch+1, i+1, round(train_acc,3), round(test_acc,3)))

        i += 1

    return train_loss_lst, train_acc_lst, test_acc_lst

def test(testloader, net, optimizer, criterion):
    test_pred, test_target = [], []
    net.eval()
    with torch.no_grad():
        for data in testloader:
      
        batch_x =  data[0]
        optimizer.zero_grad()
        outputs = net(batch_x)
        _, predicted = torch.max(outputs.data, 1)

        test_pred.append( predicted.cpu().detach().numpy() )
        test_target.append(data[1].cpu().detach().numpy())
    test_pred = np.concatenate(test_pred).ravel()
    test_target = np.concatenate(test_target).ravel()
    return np.mean(test_pred.astype(np.int)==test_target ) 

def main():
    if not os.path.exists('./data/'):
        wget_cmd = 'wget https://s3-eu-west-1.amazonaws.com/bsopenbucket/e6040/data.zip'
        unzip_cmd = 'unzip data.zip'
        os.system(wget_cmd)
        os.system(unzip_cmd)
    images, labels, mean, std = read_data('data/')
    
    batch_size = 64

    data_train = LoadBatch(np.moveaxis(images['train'], 3, 1), labels['train'])
    data_test = LoadBatch(np.moveaxis(images['test'], 3, 1), labels['test'])
    trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2)

    net = models.resnet34(pretrained=False)

    adam_optim = torch.optim.Adam(net.parameters())
    adagrad_optim = torch.optim.Adagrad(net.parameters())
    criterion = nn.CrossEntropyLoss()
    res_Adam = train_test(trainloader, testloader, net, adam_optim, criterion, epochs = 1)
    res_Adagrad = train_test(trainloader, testloader ,net, adagrad_optim, criterion, epochs = 1)

    print("working")
    plt.plot(res_Adam[0], label = "Adam")
    plt.plot(res_Adagrad[0], label = "Adagrad")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig('./image_output/1.png')

    plt.plot(res_Adam[1], label = "Adam")
    plt.plot(res_Adagrad[1], label = "Adagrad")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.savefig('./image_output/2.png')

    plt.plot(res_Adam[2], label = "Adam")
    plt.plot(res_Adagrad[2], label = "Adagrad")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.savefig('./image_output/3.png')

if __name__ == "__main__":
    main()