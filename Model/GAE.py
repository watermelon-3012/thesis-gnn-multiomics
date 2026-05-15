import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.VGAE import Encoder

class GAE(nn.Module):
    def __init__(
        self,
        dropout,
        in_rna, in_adt,
        branch_dims=(128, 64),
        fusion_dim=128,
        z_dim=32,
        recon_rna_dim=None,
        recon_adt_dim=None,
        recon_spatial_dim=None,
    ):
        super().__init__()

        # ===== Encoders (same as VGAE) =====
        self.rna_branch = Encoder(in_rna, branch_dims, dropout=dropout)
        self.adt_branch = Encoder(in_adt, branch_dims, dropout=dropout)

        fused_in = branch_dims[-1] * 2
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, z_dim)   # directly output z
        )

        # ===== Reconstruction heads =====
        self.recon_rna = None
        self.recon_adt = None
        self.recon_spatial = None

        if recon_rna_dim is not None:
            self.recon_rna = nn.Sequential(
                nn.Linear(z_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, recon_rna_dim),
            )
        if recon_adt_dim is not None:
            self.recon_adt = nn.Sequential(
                nn.Linear(z_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, recon_adt_dim),
            )
        if recon_spatial_dim is not None:
            self.recon_spatial = nn.Sequential(
                nn.Linear(z_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, recon_spatial_dim),
            )

    def encode(self, x_rna, x_adt, edge_index):
        h_rna = self.rna_branch(x_rna, edge_index)
        h_adt = self.adt_branch(x_adt, edge_index)
        h = torch.cat([h_rna, h_adt], dim=-1)
        z = self.fuse(h)
        return z

    def decode_graph(self, z, edge_index):
        # same as VGAE
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1))

    def forward(self, x_rna, x_adt, edge_index):
        z = self.encode(x_rna, x_adt, edge_index)

        xhat_rna = self.recon_rna(z) if self.recon_rna else None
        xhat_adt = self.recon_adt(z) if self.recon_adt else None
        xhat_spatial = self.recon_spatial(z) if self.recon_spatial else None

        adj_pred = self.decode_graph(z, edge_index)

        return z, xhat_rna, xhat_adt, xhat_spatial, adj_pred
    
def train_gae(model, data, epochs, device,
              lr=1e-3, weight_decay=1e-5,
              lambda_rna=1.0, lambda_adt=1.0):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    edge_index = data.edge_index.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z, xhat_rna, xhat_adt, _, _ = model(
            data.x_rna.to(device),
            data.x_adt.to(device),
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
        loss_rna = F.mse_loss(xhat_rna, data.x_rna.to(device)) if xhat_rna is not None else 0
        loss_adt = F.mse_loss(xhat_adt, data.x_adt.to(device)) if xhat_adt is not None else 0

        # ===== Total loss =====
        loss = loss_edges + lambda_rna * loss_rna + lambda_adt * loss_adt

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:03d} | "
              f"Total: {loss.item():.4f} | "
              f"Graph: {loss_edges.item():.4f} | "
              f"RNA: {float(loss_rna):.4f} | "
              f"ADT: {float(loss_adt):.4f}")

    # ===== Final embedding =====
    model.eval()
    with torch.no_grad():
        z, _, _, _, _ = model(
            data.x_rna.to(device),
            data.x_adt.to(device),
            edge_index
        )

    return z