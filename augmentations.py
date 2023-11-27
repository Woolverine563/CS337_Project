import torch
import numpy as np
def Time_Augmentation(data):
    sigma = 0.5
    segments = 5
    augmented_data = torch.zeros(data.shape)
    for i, x in enumerate(data):
        aug_choice = np.random.randint(0, 3)
        if aug_choice == 0:
            augmented_data[i] = x + torch.random.normal(loc=0., scale=sigma, shape=data.shape)
        elif aug_choice == 1:
            splits = np.array_split(np.arange(x.shape[2]), segments)
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            x[i] = x[0, warp]
        else:
            
        