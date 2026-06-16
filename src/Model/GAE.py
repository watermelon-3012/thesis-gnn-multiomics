import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.VGAE import Encoder
from torch_geometric.utils import negative_sampling

class GAE(nn.Module):
    def __init__(
        self,
        dropout,
        in_omics1, in_omics2,
        branch_dims=(128, 64),
        fusion_dim=128,
        z_dim=32,
        recon_omics1_dim=None,
        recon_omics2_dim=None,
        recon_spatial_dim=None,
    ):
        super().__init__()

        # ===== Encoders (same as VGAE) =====
        self.omics1_branch = Encoder(in_omics1, branch_dims, dropout=dropout)
        self.omics2_branch = Encoder(in_omics2, branch_dims, dropout=dropout)

        fused_in = branch_dims[-1] * 2
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, z_dim)   # directly output z
        )

        # ===== Reconstruction heads =====
        self.recon_omics1 = None
        self.recon_omics2 = None
        self.recon_spatial = None

        if recon_omics1_dim is not None:
            self.recon_omics1 = nn.Sequential(
                nn.Linear(z_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, recon_omics1_dim),
            )
        if recon_omics2_dim is not None:
            self.recon_omics2 = nn.Sequential(
                nn.Linear(z_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, recon_omics2_dim),
            )
        if recon_spatial_dim is not None:
            self.recon_spatial = nn.Sequential(
                nn.Linear(z_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, recon_spatial_dim),
            )

    def encode(self, x_omics1, x_omics2, edge_index):
        h_omics1 = self.omics1_branch(x_omics1, edge_index)
        h_omics2 = self.omics2_branch(x_omics2, edge_index)
        h = torch.cat([h_omics1, h_omics2], dim=-1)
        z = self.fuse(h)
        return z

    def decode_graph(self, z, edge_index):
        # same as VGAE
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1))

    def forward(self, x_omics1, x_omics2, edge_index):
        z = self.encode(x_omics1, x_omics2, edge_index)

        xhat_omics1 = self.recon_omics1(z) if self.recon_omics1 else None
        xhat_omics2 = self.recon_omics2(z) if self.recon_omics2 else None
        xhat_spatial = self.recon_spatial(z) if self.recon_spatial else None

        adj_pred = self.decode_graph(z, edge_index)

        return z, xhat_omics1, xhat_omics2, xhat_spatial, adj_pred
    
def train_gae(model, data, epochs, device,
              lr=1e-3, weight_decay=1e-5,
              lambda_omics1=1.0, lambda_omics2=1.0):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    edge_index = data.edge_index.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z, xhat_omics1, xhat_omics2, _, _ = model(
            data.x_omics1.to(device),
            data.x_omics2.to(device),
            edge_index
        )

        # ===== Graph reconstruction =====
        pos_pred = model.decode_graph(z, edge_index)
        pos_label = torch.ones_like(pos_pred)

        neg_edge_index = negative_sampling(
            edge_index, num_nodes=z.size(0), num_neg_samples=edge_index.size(1)
        )
        neg_pred = model.decode_graph(z, neg_edge_index)
        neg_label = torch.zeros_like(neg_pred)

        preds = torch.cat([pos_pred, neg_pred])
        labels = torch.cat([pos_label, neg_label])

        loss_edges = F.binary_cross_entropy(preds, labels)

        # ===== Feature reconstruction =====
        loss_omics1 = F.mse_loss(xhat_omics1, data.x_omics1.to(device)) if xhat_omics1 is not None else 0
        loss_omics2 = F.mse_loss(xhat_omics2, data.x_omics2.to(device)) if xhat_omics2 is not None else 0

        # ===== Total loss =====
        loss = loss_edges + lambda_omics1 * loss_omics1 + lambda_omics2 * loss_omics2

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:03d} | "
              f"Total: {loss.item():.4f} | "
              f"Graph: {loss_edges.item():.4f} | "
              f"Omics1: {float(loss_omics1):.4f} | "
              f"Omics2: {float(loss_omics2):.4f}")

    # ===== Final embedding =====
    model.eval()
    with torch.no_grad():
        z, _, _, _, _ = model(
            data.x_omics1.to(device),
            data.x_omics2.to(device),
            edge_index
        )

    return z