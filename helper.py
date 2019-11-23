# helper functions for the data loading 
import _pickle as pickle
import numpy as np
import os
from matplotlib import pyplot as plt

def read_data(data_path):
	print("I am reading data,...please wait for it")

	images, labels = {}, {}

	train_files = [
		"data_batch_1",
		"data_batch_2",
		"data_batch_3",
		"data_batch_4",
		"data_batch_5",
	]
	test_file = [
		"test_batch",
	]
	images["train"], labels["train"] = _read_data(data_path, train_files)
	images["test"], labels["test"] = _read_data(data_path, test_file)

	print("Prepropcess: Normalization")
	mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
	std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

	images["train"] = (images["train"] - mean) / std
	images["test"] = (images["test"] - mean) / std
	
	labels['train'] = labels['train'].astype(np.long)

	return images, labels

def _read_data(data_path, train_files):
	images, labels = [], []
	for file_name in train_files:
		full_name = os.path.join(data_path, file_name)
		print(full_name)
		with open(full_name, mode='rb') as finp:
			data = pickle.load(finp, encoding='bytes')
			data = convert(data)
			batch_images = data["data"].astype(np.float32) / 255.0
			batch_labels = np.array(data["labels"], dtype=np.int32)
			images.append(batch_images)
			labels.append(batch_labels)
	images,labels = np.concatenate(images, axis=0), np.concatenate(labels, axis=0)
	images = np.reshape(images, [-1, 3, 32, 32])
	images = np.transpose(images, [0, 2, 3, 1])

	return images, labels

def convert(data):
		if isinstance(data, bytes):  return data.decode('ascii')
		if isinstance(data, dict):   return dict(map(convert, data.items()))
		if isinstance(data, tuple):  return map(convert, data)
		return data

def plot(arrays, labels, dim, y_label):
    for arr, label in zip(arrays, labels):
        plt.plot(arr[dim], label=label)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel(y_label)
    plt.show()
    plt.savefig('./image_output/{}.png'.format(y_label))
    return