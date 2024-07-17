import torch
import numpy as np

test = torch.tensor([[1,2,3,4]])
a = torch.tensor(0)
print(test, test.shape)
test[0,0] = a
print(test, test.shape)