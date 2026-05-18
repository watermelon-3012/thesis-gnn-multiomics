import torch
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        last_dim = in_dim

        for i, h in enumerate(hidden_dims):
            concat = (i != len(hidden_dims) - 1)

            out_dim = h // heads if concat else h

            self.layers.append(
                GATConv(
                    in_channels=last_dim,
                    out_channels=out_dim,
                    heads=heads,
                    concat=concat,
                    dropout=dropout
                )
            )

            last_dim = h

    def forward(self, x, edge_index):
        h = x
        for gat in self.layers:
            h = gat(h, edge_index)
            h = F.elu(h)  # better than ReLU for GAT
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
class GATE(nn.Module):
    def __init__(
        self,
        dropout,
        in_omics1, in_omics2,
        branch_dims=(128, 64),
        fusion_dim=128,
        z_dim=32,
        heads=4
    ):
        super().__init__()

        # ===== GAT encoders =====
        self.omics1_branch = GATEncoder(in_omics1, branch_dims, heads=heads, dropout=dropout)
        self.omics2_branch = GATEncoder(in_omics2, branch_dims, heads=heads, dropout=dropout)

        fused_in = branch_dims[-1] * 2

        self.encoder = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, z_dim)
        )

        # ===== Decoders (IMPORTANT PART of GATE) =====
        self.decoder_omics1 = nn.Sequential(
            nn.Linear(z_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, in_omics1)
        )

        self.decoder_omics2 = nn.Sequential(
            nn.Linear(z_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, in_omics2)
        )

    def forward(self, x_omics1, x_omics2, edge_index):
        h_omics1 = self.omics1_branch(x_omics1, edge_index)
        h_omics2 = self.omics2_branch(x_omics2, edge_index)

        h = torch.cat([h_omics1, h_omics2], dim=-1)
        z = self.encoder(h)

        xhat_omics1 = self.decoder_omics1(z)
        xhat_omics2 = self.decoder_omics2(z)

        return z, xhat_omics1, xhat_omics2
    
def train_gate(model, data, epochs, device,
               lr=1e-3, weight_decay=1e-5,
               lambda_omics1=1.0, lambda_omics2=1.0,
               lambda_smooth=0.1,
               lambda_reg=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_omics1 = data.x_omics1.to(device)
    x_omics2 = data.x_omics2.to(device)
    edge_index = data.edge_index.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z, xhat_omics1, xhat_omics2 = model(x_omics1, x_omics2, edge_index)

        # ===== 1. Feature reconstruction =====
        loss_omics1 = F.mse_loss(xhat_omics1, x_omics1)
        loss_omics2 = F.mse_loss(xhat_omics2, x_omics2)

        # ===== 2. Spatial smoothness =====
        row, col = edge_index
        loss_smooth = ((z[row] - z[col])**2).mean()

        # ===== 3. Latent regularization =====
        loss_reg = torch.mean(z**2)

        # ===== Total loss =====
        loss = (
            lambda_omics1 * loss_omics1 +
            lambda_omics2 * loss_omics2 +
            lambda_smooth * loss_smooth +
            lambda_reg * loss_reg
        )

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:03d} | "
              f"Total: {loss.item():.4f} | "
              f"Omics1: {loss_omics1.item():.4f} | "
              f"Omics2: {loss_omics2.item():.4f} | "
              f"Smooth: {loss_smooth.item():.4f} | "
              f"Reg: {loss_reg.item():.4f}")

    # ===== Final embedding =====
    model.eval()
    with torch.no_grad():
        z, _, _ = model(x_omics1, x_omics2, edge_index)

    return z