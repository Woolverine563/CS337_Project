import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations1 import DataTransform_FD, DataTransform_TD
import torch.fft as fft

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = mode
        X_train = dataset["samples"][:10*128]
        y_train = dataset["labels"][:10*128]
        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        self.x_data = X_train
        self.y_data = y_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        self.x_data_f = fft.fft(self.x_data).abs()
        self.len = X_train.shape[0]

        """Augmentation"""
        if self.training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, config)
            self.aug1_f = DataTransform_FD(self.x_data_f, config) # [7360, 1, 90]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


def generate_dataloaders(data_path, mode, batch_size):
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    val_dataset = torch.load(os.path.join(data_path, "val.pt")) 
    test_dataset = torch.load(os.path.join(data_path, "test.pt")) 

    train_dataset = Load_Dataset(train_dataset, mode)
    val_dataset = Load_Dataset(val_dataset, mode)
    test_dataset = Load_Dataset(test_dataset, mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=True, drop_last=True,
                                              num_workers=0)

    return train_loader, val_loader, test_loader

    
    

    
    
    
    