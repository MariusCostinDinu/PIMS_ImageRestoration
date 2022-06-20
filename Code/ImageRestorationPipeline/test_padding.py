import torch

a = torch.tensor([[5, 17, 32], [921, 21, 61], [624, -43, 12]])
b = torch.zeros(3, 5)

b[:, 1:4] += a

print('b = ', b)