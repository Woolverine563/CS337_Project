import torch

# Load the .pt file
data = torch.load('SleepEEG/test.pt')

# Get the dimensions of the loaded data
dimensions = data['samples'].shape

print(dimensions)
