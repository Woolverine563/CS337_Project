import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations import DataTransform_FD, DataTransform_TD
import torch.fft as fft

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, mode, target_dataset_size, subset = True):
        super(Load_Dataset, self).__init__()
        self.training_mode = mode
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :1, :178] # take the first 178 samples
        
        if subset == True:
            subset_size = target_dataset_size * 30
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]
            # print('Using subset for debugging, the datasize is:', y_train.shape[0])
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train
        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""
        
        self.x_data_f = fft.fft(self.x_data).abs()
        self.len = X_train.shape[0]
        """Augmentation"""
        if self.training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, jitter_ratio = 2)
            self.aug1_f = DataTransform_FD(self.x_data_f) # [7360, 1, 90]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


def generate_dataloaders(data_path, mode, batch_size, target_batch_size, subset = True):
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    val_dataset = torch.load(os.path.join(data_path, "train.pt")) 
    test_dataset = torch.load(os.path.join(data_path, "test.pt")) 

    train_dataset = Load_Dataset(train_dataset, mode, target_dataset_size=batch_size, subset=subset)
    val_dataset = Load_Dataset(val_dataset, mode, target_dataset_size=target_batch_size, subset=subset)
    test_dataset = Load_Dataset(test_dataset, mode, target_dataset_size=target_batch_size, subset=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=target_batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=target_batch_size,
                                              shuffle=True, drop_last=True,
                                              num_workers=0)
    return train_loader, val_loader, test_loader

    
    

    
    
    
    