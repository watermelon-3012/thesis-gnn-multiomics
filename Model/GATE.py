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
        in_rna, in_adt,
        branch_dims=(128, 64),
        fusion_dim=128,
        z_dim=32,
        heads=4
    ):
        super().__init__()

        # ===== GAT encoders =====
        self.rna_branch = GATEncoder(in_rna, branch_dims, heads=heads, dropout=dropout)
        self.adt_branch = GATEncoder(in_adt, branch_dims, heads=heads, dropout=dropout)

        fused_in = branch_dims[-1] * 2

        self.encoder = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, z_dim)
        )

        # ===== Decoders (IMPORTANT PART of GATE) =====
        self.decoder_rna = nn.Sequential(
            nn.Linear(z_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, in_rna)
        )

        self.decoder_adt = nn.Sequential(
            nn.Linear(z_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, in_adt)
        )

    def forward(self, x_rna, x_adt, edge_index):
        h_rna = self.rna_branch(x_rna, edge_index)
        h_adt = self.adt_branch(x_adt, edge_index)

        h = torch.cat([h_rna, h_adt], dim=-1)
        z = self.encoder(h)

        xhat_rna = self.decoder_rna(z)
        xhat_adt = self.decoder_adt(z)

        return z, xhat_rna, xhat_adt
    
def train_gate(model, data, epochs, device,
               lr=1e-3, weight_decay=1e-5,
               lambda_rna=1.0, lambda_adt=1.0,
               lambda_smooth=0.1,
               lambda_reg=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_rna = data.x_rna.to(device)
    x_adt = data.x_adt.to(device)
    edge_index = data.edge_index.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z, xhat_rna, xhat_adt = model(x_rna, x_adt, edge_index)

        # ===== 1. Feature reconstruction =====
        loss_rna = F.mse_loss(xhat_rna, x_rna)
        loss_adt = F.mse_loss(xhat_adt, x_adt)

        # ===== 2. Spatial smoothness =====
        row, col = edge_index
        loss_smooth = ((z[row] - z[col])**2).mean()

        # ===== 3. Latent regularization =====
        loss_reg = torch.mean(z**2)

        # ===== Total loss =====
        loss = (
            lambda_rna * loss_rna +
            lambda_adt * loss_adt +
            lambda_smooth * loss_smooth +
            lambda_reg * loss_reg
        )

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:03d} | "
              f"Total: {loss.item():.4f} | "
              f"RNA: {loss_rna.item():.4f} | "
              f"ADT: {loss_adt.item():.4f} | "
              f"Smooth: {loss_smooth.item():.4f} | "
              f"Reg: {loss_reg.item():.4f}")

    # ===== Final embedding =====
    model.eval()
    with torch.no_grad():
        z, _, _ = model(x_rna, x_adt, edge_index)

    return z