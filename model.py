import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class EncoderParams:
    def __init__(self, input_dim, hidden_dim=None, n_heads=2, num_layers=2):
        self.input_dim = input_dim
        if(hidden_dim != None):
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = 2*self.input_dim
        self.n_heads = n_heads
        self.num_layers = num_layers

class Encoder(nn.Module):
    def __init__(self, params):

        self.encoder = TransformerEncoder(
            encoder_layer = TransformerEncoderLayer(
                d_model = params.input_dim,
                dim_feedforward = params.hidden_dim,
                nhead= params.n_heads,
            ),
            num_layers=params.num_layers
        )

    def forward(self, input):
        return self.encoder(input).reshape(input.shape[0], -1)

class ProjectorParams:
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

class Projector(nn.Module):
    def __init__(self, params):
        self.projector = nn.Sequential(
            nn.Linear(params.input_dim, params.hidden_dim),
            nn.BatchNorm1d(params.hidden_dim),
            nn.ReLU(),
            nn.Linear(params.hidden_dim,params.output_dim)
        )

    def forward(self, input):
        return self.projector(input)

class TFC(nn.Module):
    
    def __init__(self, encoderParams, projectorparams):
        super(TFC, self).__init__()

        self.time_encoder = Encoder(encoderParams)
        self.time_projector = Projector(projectorparams)

        self.frq_encoder = Encoder(encoderParams)
        self.frq_projector = Projector(projectorparams)

    def forward(self, x_time, x_frq):
        #encoding
        h_time = self.time_encoder(x_time)
        h_frq = self.frq_encoder(x_frq)

        #getting to same dim
        z_time = self.time_projector(h_time)
        z_frq = self.frq_projector(h_frq)

        return h_time, z_time, h_frq, z_frq
    

class target_classifier(nn.Module):
    def __init__(self, targets):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, targets)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
