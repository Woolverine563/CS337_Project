import torch 
from torch import nn
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, tau):
        super(ContrastiveLoss, self).__init__()

        self.batch_size = batch_size
        self.tau = tau
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="sum")
        self.cosine_similarity = nn.CosineSimilarity(dim = -1)

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

        similarity_matrix = self.cosine_similarity(x_y.unsqueeze(1), x_y.unsqueeze(0))#Shayad se class banao, dekhna padega agar issue hue
        #issue over
        # print(similarity_matrix.shape)
        #get the numerators
        xi_yi = torch.diag(similarity_matrix, self.batch_size)

        numerators = torch.cat((xi_yi, xi_yi)).view(2*self.batch_size, 1)
        #get the denominators
        mask = self.get_mask()
        denominators = similarity_matrix[mask].view(2*self.batch_size, -1)
        #get logits
        logits = torch.cat((numerators, denominators), dim=1)/self.tau

        #improvable -logx -x

        onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),torch.zeros(2 * self.batch_size, denominators.shape[-1])),dim=-1).long()
        # Add poly loss
        pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))
        epsilon = self.batch_size
        loss = self.cross_entropy_loss(logits, torch.zeros(2*self.batch_size).long())/(2*self.batch_size)
        # loss = self.cross_entropy_loss(logits, torch.zeros(2*self.batch_size).long())/(2*self.batch_size) + epsilon * (1/self.batch_size - pt)
        #improvable over
        return loss
