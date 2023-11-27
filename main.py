
from dataloader import generate_dataloaders

data_path = "datasets/SleepEEG/"
mode = "pre_training"
batch_size = 128 # For Sleep EEG, batch_size = 128

trian_dl, valid_dl, test_dl = generate_dataloaders(data_path, mode, batch_size = batch_size)

