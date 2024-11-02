import torch

without_coat_xyz = torch.load('without_coat_before_param.pt')
with_coat_xyz = torch.load('with_coat_before_param.pt')

if torch.equal(without_coat_xyz, with_coat_xyz):
    print('Two models have the same xyz.')
else:
    print('Two models have different xyz.')