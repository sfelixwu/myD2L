import torch
x = torch.arange(12, dtype=torch.float32)
print(x)
print(x.shape)
print(x.numel())

X = x.reshape(3,4)
print(X)

X = x.reshape(2,6)
print(X)

X = x.reshape(6,2)
print(X)

Y = torch.zeros(2,3,4)
print(Y)

Y = torch.ones(2,3,4)
print(Y)

Z = torch.randn(3, 4)
print(Z)

Z = torch.randn(2, 6)
print(Z)

Z = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(Z)

z = Z.reshape(3,4)
print(z)

z = Z.reshape(4,3)
print(z)

x2 = torch.tensor([1.0, 2, 4, 8])
y2 = torch.tensor([2, 2, 2, 2])
print(x2+y2, x2-y2, x2*y2, x2/y2, x2**y2)
print(torch.exp(x2))

z3 = torch.arange(12, dtype=torch.float32).reshape((3,4))
print(torch.cat((z3,Z), dim=0))
print(torch.cat((z3,Z), dim=1))
print(id(z3))

z3 = torch.arange(12, dtype=torch.float32).reshape(3,4)
print(torch.cat((z3,Z), dim=0))
print(torch.cat((z3,Z), dim=1))
print(id(z3))

print(z3 == Z)
print(z3.sum())
print(z3[-1])
z3[-1] = torch.tensor([2,2,2,2])
print(z3[-1])
print(z3[1:3])
print(z3[0:2, :])
print(id(z3))

A = z3.numpy()
B = torch.from_numpy(A)
print(type(A), type(B))
print(z3)
print(A)
print(B)

aa = torch.tensor([3.555])
print(aa, aa.item(), float(aa), int(aa))

import os

os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd
data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs2 = pd.to_numeric(inputs["NumRooms"])
inputs3 = inputs2.fillna(inputs2.mean())
print(inputs)
print(inputs2)
print(inputs3)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

