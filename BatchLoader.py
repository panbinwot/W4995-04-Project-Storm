from torch.autograd import Variable
from torch.utils.data import Dataset

class BatchLoader(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y
  
	def __len__(self):
		dataset_size = len(self.X)
		return dataset_size
  
	def __getitem__(self, idx):
		X_batch = self.X[idx]
		y_batch = self.y[idx]
		return X_batch, y_batch