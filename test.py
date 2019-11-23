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

from LoadBatch import LoadBatch
from helper import read_data
from storm import Storm

def train(train_loader, test_loader, net, optimizer, criterion, epochs, max_iter_per_epoch=None,
		  test_freq=100, no_test=False, verbose=True):
	train_loss_lst = []
	train_acc_lst = []
	test_acc_lst = []

	i = 0
	for epoch in range(epochs):
		for data in train_loader:
			net.train()
			batch_x = data[0]
			batch_y = data[1]
			optimizer.zero_grad()
			output = net(batch_x)
			loss = criterion(output, batch_y)
			loss.backward()
			optimizer.step()

			if i % test_freq == 0:
				train_loss_lst.append(loss)
				net.eval()
				with torch.no_grad():
					optimizer.zero_grad()
					train_acc = test(train_loader, net, optimizer, 10)      # TODO 10 -> 100
					train_acc_lst.append(train_acc)

				if not no_test:
					test_acc = test(test_loader, net, optimizer, 10)        # TODO 10 -> 100
					test_acc_lst.append(test_acc)
					if verbose:
						print("Epoch: {}, Batch: {}, Train Accuracy: {}, Test Accuracy: {}".format(epoch + 1, i + 1,
																								   round(float(train_acc), 3),
																								   round(float(test_acc), 3)))
				elif verbose:
					print("Epoch: {}, Batch: {}, Train Accuracy: {}".format(epoch + 1, i + 1,
																			round(float(train_acc), 3)))
	
			if max_iter_per_epoch is not None and i > max_iter_per_epoch:
				break
			i += 1

	return train_loss_lst, train_acc_lst, test_acc_lst

def test(testloader, net, optimizer, max_nb_batches):
	assert max_nb_batches > 0
	test_pred, test_target = [], []
	net.eval()
	i = 0
	with torch.no_grad():
		for data in testloader:
			batch_x = data[0]
			optimizer.zero_grad()
			outputs = net(batch_x)
			_, predicted = torch.max(outputs.data, 1)

			test_pred.append(predicted.cpu().detach().numpy())
			test_target.append(data[1].cpu().detach().numpy())
			i += 1
			if i > max_nb_batches:
				break
		test_pred = np.concatenate(test_pred).ravel()
		test_target = np.concatenate(test_target).ravel()
	return np.mean(test_pred.astype(np.int) == test_target)

def main():
	if not os.path.exists('./data/'):
		wget_cmd = 'wget https://s3-eu-west-1.amazonaws.com/bsopenbucket/e6040/data.zip'
		unzip_cmd = 'unzip data.zip'
		os.system(wget_cmd)
		os.system(unzip_cmd)
	images, labels, _, __ = read_data('data/')
	
	batch_size = 64

	data_train = LoadBatch(np.moveaxis(images['train'], 3, 1), labels['train'])
	data_test = LoadBatch(np.moveaxis(images['test'], 3, 1), labels['test'])
	trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0)
	testloader = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=0)

	net = models.resnet18(pretrained=False)
	storm_optim = Storm(net.parameters())
	criterion = nn.CrossEntropyLoss()
	res_Storm =  train(trainloader, testloader ,net, storm_optim, criterion, epochs = 1)

if __name__ == "__main__":
	main()
