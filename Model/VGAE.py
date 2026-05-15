import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import negative_sampling, to_undirected

class Encoder(nn.Module): 
    def __init__(self, in_dim, hidden_dims, dropout): 
        super().__init__() 
        self.gcn_layers = nn.ModuleList()
        last = in_dim
        for h in hidden_dims: 
            self.gcn_layers.append(GCNConv(last, h)) 
            last = h 
        self.dropout = dropout
            
    def forward(self, x, edge_index):
        h = x
        for conv in self.gcn_layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
class VGAE(nn.Module):
    def __init__(
        self, dropout,
        in_omics1, in_omics2,
        branch_dims=(128, 64),
        fusion_dim=128,
        z_dim=32,
        recon_omics1_dim=None,
        recon_omics2_dim=None,
        recon_spatial_dim=None,   
    ):
        super().__init__()
        # Separate 2 omics encoders
        self.omics1_branch = Encoder(in_omics1, branch_dims, dropout=dropout)
        self.omics2_branch = Encoder(in_omics2, branch_dims, dropout=dropout)

        # Fusion
        fused_in = branch_dims[-1] * 2
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(fusion_dim, z_dim)
        self.logvar = nn.Linear(fusion_dim, z_dim)

        # Optional reconstruction heads (feature decoders)
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
        h = self.fuse(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_graph(self, z, edge_index):
        # Inner-product VGAE decoder
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1))

    def forward(self, x_omics1, x_omics2, edge_index):
        mu, logvar = self.encode(x_omics1, x_omics2, edge_index)
        z = self.reparam(mu, logvar)

        # feature reconstructions
        xhat_omics1 = self.recon_omics1(z) if self.recon_omics1 else None
        xhat_omics2 = self.recon_omics2(z) if self.recon_omics2 else None
        xhat_spatial = self.recon_spatial(z) if self.recon_spatial else None

        # graph reconstruction (edge probabilities)
        adj_pred = self.decode_graph(z, edge_index)

        return z, mu, logvar, xhat_omics1, xhat_omics2, xhat_spatial, adj_pred
    
#LOSS

def edge_recon_loss(z, edge_index, num_nodes, neg_ratio=1.0):
    pos_edge_index = edge_index
    num_pos = pos_edge_index.size(1)
    num_neg = int(neg_ratio * num_pos)

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method="sparse"
    )

    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss

def kl_loss(mu, logvar, eps=1e-9):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp() + eps)

def train_vgae(model, data, epochs, device,
               lr=1e-3, weight_decay=1e-5, dropout=0.2,
               lambda_omics1=1.0, lambda_omics2=1.0):


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    edge_index = data.edge_index.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z, mu, logvar, xhat_omics1, xhat_omics2, _, adj_pred = model(
            data.x_omics1.to(device),
            data.x_omics2.to(device),
            edge_index
        )

        # Graph reconstruction loss (pos + neg edges)
        pos_pred = model.decode_graph(z, edge_index)
        pos_label = torch.ones_like(pos_pred)

        # sample negative edges
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=z.size(0), num_neg_samples=edge_index.size(1)
        )
        neg_pred = model.decode_graph(z, neg_edge_index)
        neg_label = torch.zeros_like(neg_pred)

        # combine pos & neg
        preds = torch.cat([pos_pred, neg_pred])
        labels = torch.cat([pos_label, neg_label])
        loss_edges = F.binary_cross_entropy(preds, labels)

        # KL loss 
        loss_kld = kl_loss(mu, logvar)

        # Feature reconstruction losses 
        loss_omics1 = F.mse_loss(xhat_omics1, data.x_omics1.to(device)) if xhat_omics1 is not None else 0
        loss_omics2 = F.mse_loss(xhat_omics2, data.x_omics2.to(device)) if xhat_omics2 is not None else 0

        # Total loss
        loss = loss_edges + loss_kld + lambda_omics1 * loss_omics1 + lambda_omics2 * loss_omics2

        # Backprop
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:03d} | "
              f"Total: {loss.item():.4f} | "
              f"Graph: {loss_edges.item():.4f} | "
              f"KLD: {loss_kld.item():.4f} | "
              f"Omics1: {float(loss_omics1):.4f} | "
              f"Omics2: {float(loss_omics2):.4f}")

    # Final embeddings
    model.eval()
    with torch.no_grad():
        z, _, _, _, _, _, _ = model(
            data.x_omics1.to(device),
            data.x_omics2.to(device),
            edge_index
        )

    return z