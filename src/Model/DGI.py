import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Model.VGAE import Encoder

def corrupt_features(x):
    perm = torch.randperm(x.size(0))
    return x[perm]

def corrupt(data):
    x_omics1_corrupt = corrupt_features(data.x_omics1)
    x_omics2_corrupt = corrupt_features(data.x_omics2)
    return x_omics1_corrupt, x_omics2_corrupt, data.edge_index

class DGI(nn.Module):
    def __init__(self, in_omics1, in_omics2, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()

        # encoders 
        self.omics1_encoder = Encoder(in_omics1, hidden_dims, dropout)
        self.omics2_encoder = Encoder(in_omics2, hidden_dims, dropout)

        fused_dim = hidden_dims[-1] * 2

        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )

        # discriminator
        self.weight = nn.Parameter(torch.Tensor(fused_dim, fused_dim))
        nn.init.xavier_uniform_(self.weight)

    def encode(self, x_omics1, x_omics2, edge_index):
        h_omics1 = self.omics1_encoder(x_omics1, edge_index)
        h_omics2 = self.omics2_encoder(x_omics2, edge_index)
        h = torch.cat([h_omics1, h_omics2], dim=-1)
        z = self.fuse(h)
        return z

    def summary(self, z):
        # global summary vector
        return torch.sigmoid(z.mean(dim=0))

    def discriminate(self, z, s):
        # bilinear scoring
        s = torch.matmul(s, self.weight)
        return torch.matmul(z, s)

    def forward(self, data):
        # positive
        z_pos = self.encode(data.x_omics1, data.x_omics2, data.edge_index)

        # negative (corrupted)
        x_omics1_neg, x_omics2_neg, edge_neg = corrupt(data)
        z_neg = self.encode(x_omics1_neg, x_omics2_neg, edge_neg)

        # summary
        s = self.summary(z_pos)

        # scores
        pos_score = self.discriminate(z_pos, s)
        neg_score = self.discriminate(z_neg, s)

        return pos_score, neg_score, z_pos

def dgi_loss(pos_score, neg_score):
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_score, torch.zeros_like(neg_score)
    )
    return pos_loss + neg_loss

def train_dgi(
    model,
    data,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-5,
    device=None,
    verbose=True
):
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []

    for epoch in range(epochs):
        model.train()

        pos_score, neg_score, _ = model(data)

        loss = dgi_loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

    return model, loss_history

@torch.no_grad()
def get_embedding_dgi(model, data, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    data = data.to(device)

    z = model.encode(data.x_omics1, data.x_omics2, data.edge_index)
    return z.cpu()