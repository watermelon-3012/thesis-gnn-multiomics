import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.VGAE import Encoder

class BGRL(nn.Module):
    def __init__(
        self,
        in_omics1,
        in_omics2,
        hidden_dims=(128, 64),
        proj_dim=64,
        dropout=0.2,
        tau=0.99
    ):
        super().__init__()

        # ---- Online encoders ----
        self.omics1_encoder = Encoder(in_omics1, hidden_dims, dropout)
        self.omics2_encoder = Encoder(in_omics2, hidden_dims, dropout)

        fused_dim = hidden_dims[-1] * 2

        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.ReLU(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # ---- Target encoders (EMA) ----
        self.omics1_encoder_target = copy.deepcopy(self.omics1_encoder)
        self.omics2_encoder_target = copy.deepcopy(self.omics2_encoder)
        self.fuse_target = copy.deepcopy(self.fuse)

        for p in self.omics1_encoder_target.parameters():
            p.requires_grad = False
        for p in self.omics2_encoder_target.parameters():
            p.requires_grad = False
        for p in self.fuse_target.parameters():
            p.requires_grad = False

        self.tau = tau

    def encode_online(self, x_omics1, x_omics2, edge_index):
        h_omics1 = self.omics1_encoder(x_omics1, edge_index)
        h_omics2 = self.omics2_encoder(x_omics2, edge_index)
        h = torch.cat([h_omics1, h_omics2], dim=-1)
        z = self.fuse(h)
        p = self.predictor(z)
        return z, p

    @torch.no_grad()
    def encode_target(self, x_omics1, x_omics2, edge_index):
        h_omics1 = self.omics1_encoder_target(x_omics1, edge_index)
        h_omics2 = self.omics2_encoder_target(x_omics2, edge_index)
        h = torch.cat([h_omics1, h_omics2], dim=-1)
        z = self.fuse_target(h)
        return z

    def forward(self, view1, view2):
        x_omics1_1, x_omics2_1, edge_1 = view1
        x_omics1_2, x_omics2_2, edge_2 = view2

        # online (view1)
        z1, p1 = self.encode_online(x_omics1_1, x_omics2_1, edge_1)

        # target (view2)
        z2 = self.encode_target(x_omics1_2, x_omics2_2, edge_2)

        return z1, p1, z2

    @torch.no_grad()
    def update_target(self):
        for online, target in zip(self.omics1_encoder.parameters(), self.omics1_encoder_target.parameters()):
            target.data = self.tau * target.data + (1 - self.tau) * online.data

        for online, target in zip(self.omics2_encoder.parameters(), self.omics2_encoder_target.parameters()):
            target.data = self.tau * target.data + (1 - self.tau) * online.data

        for online, target in zip(self.fuse.parameters(), self.fuse_target.parameters()):
            target.data = self.tau * target.data + (1 - self.tau) * online.data
    
def drop_edges(edge_index, drop_prob=0.2):
    mask = torch.rand(edge_index.size(1)) > drop_prob
    return edge_index[:, mask]

def drop_features(x, drop_prob=0.2):
    mask = torch.rand_like(x) > drop_prob
    return x * mask
def augment(data):
    edge_index = drop_edges(data.edge_index, 0.2)
    x_omics1 = drop_features(data.x_omics1, 0.2)
    x_omics2 = drop_features(data.x_omics2, 0.2)
    return x_omics1, x_omics2, edge_index

# LOSS
def bgrl_loss(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z.detach(), dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()

def train_bgrl(
    model,
    data,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-5,
    device=None,
    verbose=True
):

    # ---- device setup ----
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []

    for epoch in range(epochs):
        model.train()

        # ---- create two augmented views ----
        view1 = augment(data)
        view2 = augment(data)

        # ---- move to device ----
        view1 = tuple(v.to(device) for v in view1)
        view2 = tuple(v.to(device) for v in view2)

        # ---- forward (both directions) ----
        z1, p1, z2 = model(view1, view2)
        z2_, p2, z1_ = model(view2, view1)

        # ---- loss ----
        loss = bgrl_loss(p1, z2) + bgrl_loss(p2, z1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- EMA update ----
        model.update_target()

        loss_history.append(loss.item())

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

    return model, loss_history

@torch.no_grad()
def get_embedding_bgrl(model, data, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    data = data.to(device)

    h_omics1 = model.omics1_encoder(data.x_omics1, data.edge_index)
    h_omics2 = model.omics2_encoder(data.x_omics2, data.edge_index)

    z = model.fuse(torch.cat([h_omics1, h_omics2], dim=-1))

    return z.cpu()