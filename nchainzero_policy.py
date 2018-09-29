# implementing policy gradient using pytorch

import torch
import torch.nn as nn

N, D_in, H, D_out = 1, 5, 10, 1

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.Sigmoid(),
    nn.Linear(H, D_out), # probability of going right
)

loss_fn = nn.MSELoss(reduction='sum')

x = torch.eye(5)[2:3]

y = model(x)
print(y)
