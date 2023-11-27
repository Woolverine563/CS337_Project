import torch 
from torch import nn
import numpy as np

class ContrastiveLoss(nn.module):
    def __init__(self, batch_size, tau):

        self.batch_size = batch_size
        self.tau = tau
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="sum")

    def get_mask(self):
        diagonal = np.eye(2*self.batch_size)
        upperdiag = np.eye(2*self.batch_size, k = self.batch_size)
        lowerdiag = np.eye(2*self.batch_size, k = -self.batch_size)
        return torch.from_numpy((1-diagonal-upperdiag-lowerdiag)).type(torch.bool)

    def forward(self, x, y):
        #concat as a whole
        x_y = torch.cat((x, y), dim = 0)
        
        #issues
        #get the similarities
        similarity_matrix = nn.CosineSimilarity(dim = -1)(x_y, x_y)#Shayad se class banao, dekhna padega agar issue hue
        #issue over
        
        #get the numerators
        xi_yi = torch.diag(similarity_matrix, self.batch_size)
        numerators = torch.cat((xi_yi, xi_yi)).view(2*self.batch_size, 1)
        #get the denominators
        mask = self.get_mask()
        denominators = similarity_matrix[mask].view(2*self.batch_size, -1)
        #get logits
        logits = torch.cat((numerators, denominators), dim=1)/self.tau

        #improvable -logx -x
        #cross_entropy_loss
        loss = self.cross_entropy_loss(logits, torch.zeros(2*self.batch_size).long())/2*self.batch_size
        #improvable over
        return loss



