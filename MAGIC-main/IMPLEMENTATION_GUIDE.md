# MAGIC: Graph Representation Layer - Implementation Guide

## Quick Reference

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **GATConv** | Single attention layer | (graph, node_features) | node_embeddings |
| **GAT** | Multi-layer encoder | (graph, node_features) | hierarchical_embeddings |
| **GMAEModel** | Masked autoencoder | (masked_graph) | reconstruction_loss |
| **Decoder** | Feature reconstruction | (encoder_output, graph) | reconstructed_features |
| **SCE Loss** | Training objective | (reconstructed, original) | loss_value |

---

## 1. Code Structure Walkthrough

### Model Building Pipeline

```python
# From train.py
from model.autoencoder import build_model

def main(main_args):
    # 1. Load dataset
    dataset = load_batch_level_dataset(dataset_name)
    graphs = dataset['dataset']  # List of DGL graphs
    
    # 2. Build model
    model = build_model(main_args)  # Creates GMAEModel
    
    # 3. Create dataloaders
    train_loader = extract_dataloaders(graphs, batch_size=12)
    
    # 4. Training loop
    for epoch in range(max_epochs):
        for batch_graphs in train_loader:
            loss = model.forward(batch_graphs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Model Initialization

```python
# From model/autoencoder.py - build_model function
def build_model(args):
    model = GMAEModel(
        n_dim=args.n_dim,              # Input node feature dimension
        e_dim=args.e_dim,              # Edge feature dimension
        hidden_dim=args.num_hidden,    # 64-256 depending on dataset
        n_layers=args.num_layers,      # 3-4 layers
        n_heads=4,                     # 4 attention heads
        activation="prelu",
        feat_drop=0.1,
        negative_slope=0.2,
        residual=True,                 # Residual connections
        mask_rate=args.mask_rate,      # 0.3-0.5
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=args.alpha_l           # SCE loss exponent
    )
    return model
```

### GMAEModel Structure

```python
# From model/autoencoder.py
class GMAEModel(nn.Module):
    def __init__(self, ...):
        # 1. ENCODER: Multi-layer GAT
        self.encoder = GAT(
            n_dim=n_dim,               # Input: node feature dim
            e_dim=e_dim,               # Edge features
            hidden_dim=enc_num_hidden, # Internal dim
            out_dim=enc_num_hidden,
            n_layers=n_layers,         # 3-4 layers
            n_heads=enc_nhead,         # 4 heads
            n_heads_out=enc_nhead,
            concat_out=True,           # Concat multi-head output
            encoding=True
        )
        
        # 2. DECODER: Single-layer GAT
        self.decoder = GAT(
            n_dim=dec_in_dim,          # Input: encoder output
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,             # Output: reconstruct original
            n_layers=1,                # Only 1 layer
            n_heads=n_heads,
            n_heads_out=1,             # Single head output
            encoding=False
        )
        
        # 3. Masking
        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        
        # 4. Projection
        self.encoder_to_decoder = nn.Linear(
            dec_in_dim * n_layers,    # Concat all encoder layers
            dec_in_dim,
            bias=False
        )
        
        # 5. Loss function
        self.criterion = partial(sce_loss, alpha=alpha_l)
```

---

## 2. Forward Pass Detailed

### Step-by-Step Data Flow

```python
# From model/autoencoder.py - compute_loss method
def compute_loss(self, g):
    # ========== MASKING PHASE ==========
    pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
        g, self._mask_rate
    )
    # pre_use_g: Graph with some node features replaced by [MASK] token
    # mask_nodes: Indices of masked nodes
    # keep_nodes: Indices of visible nodes
    
    # ========== ENCODER PHASE ==========
    # Input: Masked graph, masked node features
    pre_use_x = pre_use_g.ndata['attr']  # Shape: [num_nodes, n_dim]
    
    # Forward through encoder
    enc_rep, all_hidden = self.encoder(
        pre_use_g, 
        pre_use_x, 
        return_hidden=True
    )
    # all_hidden: List of outputs from each GAT layer
    # - Layer 1: [num_nodes, hidden_dim]
    # - Layer 2: [num_nodes, hidden_dim]
    # - Layer 3: [num_nodes, hidden_dim]
    
    # Concatenate all layer outputs
    enc_rep = torch.cat(all_hidden, dim=1)
    # Shape: [num_nodes, hidden_dim * num_layers]
    
    # ========== DECODER PROJECTION PHASE ==========
    rep = self.encoder_to_decoder(enc_rep)
    # rep: [num_nodes, hidden_dim]
    # Projects concatenated representation back to hidden dimension
    
    # ========== DECODER PHASE ==========
    recon = self.decoder(pre_use_g, rep)
    # recon: [num_nodes, n_dim]
    # Reconstructs original node feature dimension
    
    # ========== LOSS COMPUTATION PHASE ==========
    # Extract original features of masked nodes
    x_init = g.ndata['attr'][mask_nodes]      # [num_mask_nodes, n_dim]
    
    # Extract reconstructed features of masked nodes
    x_recon = recon[mask_nodes]               # [num_mask_nodes, n_dim]
    
    # Compute reconstruction loss using SCE
    loss = self.criterion(x_recon, x_init)
    
    return loss
```

### Masking Function

```python
def encoding_mask_noise(self, g, mask_rate=0.3):
    """
    Randomly mask a portion of nodes and replace with learnable token
    
    Args:
        g: DGL Graph object
        mask_rate: Fraction of nodes to mask (0.0 - 1.0)
    
    Returns:
        new_g: Graph with masked nodes
        (mask_nodes, keep_nodes): Indices of masked and visible nodes
    """
    new_g = g.clone()
    num_nodes = g.num_nodes()
    
    # Generate random permutation
    perm = torch.randperm(num_nodes, device=g.device)
    
    # Calculate number of nodes to mask
    num_mask_nodes = int(mask_rate * num_nodes)
    
    # Split into mask and keep indices
    mask_nodes = perm[: num_mask_nodes]
    keep_nodes = perm[num_mask_nodes:]
    
    # Replace masked node features with learnable [MASK] token
    new_g.ndata["attr"][mask_nodes] = self.enc_mask_token
    
    return new_g, (mask_nodes, keep_nodes)
```

---

## 3. GAT Layer Implementation

### GATConv Layer Details

```python
# From model/gat.py - GATConv class
class GATConv(nn.Module):
    def __init__(self, in_dim, e_dim, out_dim, n_heads, ...):
        """
        Single GAT Convolutional Layer with multi-head attention
        
        Args:
            in_dim: Input node feature dimension
            e_dim: Edge feature dimension
            out_dim: Output feature dimension per head
            n_heads: Number of attention heads
        """
        self.n_heads = n_heads
        self.out_feat = out_dim
        
        # Linear transformations for attention
        self.fc_q = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.fc_k = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.fc_v = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        
        # Attention parameters
        self.fc_e = nn.Linear(e_dim, n_heads, bias=False)
        self.attn_weight = nn.Parameter(torch.Tensor(1, n_heads, out_dim))
        
        # Output projection
        self.fc_out = nn.Linear(out_dim * n_heads, out_dim * n_heads, bias=False)
    
    def forward(self, g, feat):
        """
        Args:
            g: DGL Graph
            feat: Node features [num_nodes, in_dim]
        
        Returns:
            out: Node embeddings [num_nodes, out_dim * n_heads]
        """
        # Linear transformations
        q = self.fc_q(feat)  # [num_nodes, out_dim * n_heads]
        k = self.fc_k(feat)
        v = self.fc_v(feat)
        
        # Reshape for multi-head attention
        q = q.view(-1, self.n_heads, self.out_feat)  # [num_nodes, n_heads, out_dim]
        k = k.view(-1, self.n_heads, self.out_feat)
        v = v.view(-1, self.n_heads, self.out_feat)
        
        # Compute attention scores
        g.ndata['q'] = q
        g.ndata['k'] = k
        g.ndata['v'] = v
        
        # Message passing with attention
        g.apply_edges(self.edge_attention)
        attn = edge_softmax(g, g.edata['a'])  # Softmax attention weights
        
        # Aggregate messages
        g.edata['w'] = attn
        g.update_all(
            fn.u_mul_e('v', 'w', 'm'),
            fn.mean('m', 'agg')
        )
        
        # Gather aggregated messages
        rst = g.ndata['agg']  # [num_nodes, n_heads, out_dim]
        rst = rst.view(-1, self.n_heads * self.out_feat)  # Flatten heads
        
        return rst
```

---

## 4. Attention Weight Analysis

### Visualizing What the Model Learns

```python
# Code to extract and visualize attention weights
def visualize_attention_weights(model, graph, feat):
    """
    Extract attention weights to understand what edges model focuses on
    """
    with torch.no_grad():
        # Forward pass through first layer
        layer_0 = model.encoder.gats[0]
        
        # Get attention scores before softmax
        q = layer_0.fc_q(feat)  # [num_nodes, out_dim * n_heads]
        k = layer_0.fc_k(feat)
        
        # Reshape for multi-head
        q = q.view(-1, layer_0.n_heads, layer_0.out_feat)
        k = k.view(-1, layer_0.n_heads, layer_0.out_feat)
        
        # Compute attention: sim(q_i, k_j) for all edges
        graph.ndata['q'] = q
        graph.ndata['k'] = k
        
        # Per-edge attention computation
        def edge_attention_fn(edges):
            # Dot product attention
            return {'a': torch.sum(
                edges.src['q'] * edges.dst['k'], 
                dim=-1
            )}
        
        graph.apply_edges(edge_attention_fn)
        
        # Get attention weights per edge
        attention_scores = graph.edata['a']
        
    return attention_scores, graph.edges()
```

---

## 5. Reconstruction Error Analysis

### Computing Anomaly Scores

```python
def compute_anomaly_scores(model, test_graphs, method='reconstruction'):
    """
    Compute anomaly scores for test graphs
    
    Args:
        model: Trained GMAEModel
        test_graphs: List of DGL graphs to evaluate
        method: 'reconstruction' or 'embedding_distance'
    
    Returns:
        anomaly_scores: Per-node and per-graph scores
    """
    model.eval()
    anomaly_scores = {
        'node_scores': [],
        'graph_scores': []
    }
    
    with torch.no_grad():
        for g in test_graphs:
            x = g.ndata['attr']
            
            # ===== METHOD 1: Reconstruction Error =====
            if method == 'reconstruction':
                # Forward through encoder
                enc_rep, all_hidden = model.encoder(g, x, return_hidden=True)
                enc_rep = torch.cat(all_hidden, dim=1)
                
                # Project and decode
                rep = model.encoder_to_decoder(enc_rep)
                x_recon = model.decoder(g, rep)
                
                # Compute per-node reconstruction error
                node_errors = torch.norm(x_recon - x, dim=1)  # L2 norm
                anomaly_scores['node_scores'].extend(node_errors.cpu().numpy())
                
                # Aggregate to graph-level score
                graph_score = torch.mean(node_errors)
                anomaly_scores['graph_scores'].append(graph_score.item())
            
            # ===== METHOD 2: Embedding Distance (KNN) =====
            elif method == 'embedding_distance':
                # Get node embeddings
                embeddings = model.encoder(g, x, return_hidden=False)
                # Use KNN outlier detection (implemented separately)
                # Higher distance to k-nearest neighbors = more anomalous
    
    return anomaly_scores
```

---

## 6. Feature Dimension Reference

### Common Dataset Configurations

**StreamSpot:**
```
n_dim = 15          # Node feature dimension
e_dim = 2           # Edge feature dimension
num_hidden = 256
num_layers = 4
batch_size = 12
```

**DARPA Trace:**
```
n_dim = 40
e_dim = 5
num_hidden = 64
num_layers = 3
batch_size = 1
```

**Wget:**
```
n_dim = 12
e_dim = 2
num_hidden = 256
num_layers = 4
batch_size = 1
```

---

## 7. Debugging & Monitoring

### Print Model Architecture

```python
model = build_model(args)
print(model)

# Output structure:
# GMAEModel(
#   (encoder): GAT(
#     (gats): ModuleList(
#       (0): GATConv(...)
#       (1): GATConv(...)
#       (2): GATConv(...)
#     )
#     (head): Identity()
#   )
#   (decoder): GAT(
#     (gats): ModuleList(
#       (0): GATConv(...)
#     )
#   )
#   (edge_recon_fc): Sequential(...)
#   (encoder_to_decoder): Linear(...)
# )
```

### Monitor Training

```python
# During training loop
for epoch in range(max_epochs):
    epoch_loss = 0
    for batch_graphs in train_loader:
        loss = model.forward(batch_graphs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    # Monitor:
    # - Loss should decrease over epochs
    # - Gradient norms should be stable
    # - Model weights should be updating
```

### Inspect Hidden Representations

```python
# Check what encoder learns at each layer
g = test_graphs[0]
x = g.ndata['attr']

with torch.no_grad():
    _, hidden_list = model.encoder(g, x, return_hidden=True)
    
    for layer_idx, h in enumerate(hidden_list):
        print(f"Layer {layer_idx}:")
        print(f"  Shape: {h.shape}")
        print(f"  Mean: {h.mean():.4f}, Std: {h.std():.4f}")
        print(f"  Min: {h.min():.4f}, Max: {h.max():.4f}")
        
        # Analyze by head (if multi-head)
        # h_per_head = h.view(-1, model.encoder.n_heads, -1)
```

---

## 8. Key Parameters Explained

```
n_dim: Input node feature dimension
  - Number of attributes per node
  - Fixed by dataset
  - Example: Process ID, command, IP address features

e_dim: Edge feature dimension
  - Attributes of edges/interactions
  - Example: syscall type, timestamp delta
  
hidden_dim: Internal representation dimension
  - Learned feature space
  - Larger = more capacity, longer training
  - Typical: 64-256

num_layers: Number of GAT layers
  - Deeper = larger receptive field
  - 3-4 layers common in MAGIC
  - Each layer adds ~1-2 hop neighborhood context

mask_rate: Fraction of nodes to mask
  - 0.3-0.5 typical
  - Higher = more challenging prediction
  - Forces robust feature learning

n_heads: Attention heads
  - 4 typical in MAGIC
  - Each head learns different aspects
  - Concatenated for output

alpha_l: SCE loss exponent
  - Controls loss sensitivity
  - 2-3 typical
  - Higher = stronger penalties for large errors
```

---

## 9. Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss doesn't decrease | Learning rate too low | Increase lr or use warmup |
| Diverging loss | Learning rate too high | Reduce lr or add gradient clipping |
| Out of memory | Batch too large | Reduce batch_size or hidden_dim |
| Poor anomaly detection | Underfitting encoder | Increase num_layers or hidden_dim |
| Overfitting | Model too powerful | Increase mask_rate or feat_drop |
| Attention weights all uniform | No useful structure | Check input graphs and edges |

---

## 10. Performance Metrics to Track

```python
# Training metrics
- Reconstruction Loss (target: decreasing)
- Per-epoch average loss
- Gradient norms (should be stable)
- Learning rate schedule

# Evaluation metrics
- ROC-AUC (detection performance)
- Precision, Recall, F1-score
- Per-node vs per-graph detection
- False positive rate on benign data
- Detection latency per graph
```

---

## Summary

The MAGIC graph representation layer is built on:

1. **Input**: Provenance graphs with node and edge features
2. **Masking**: Random node masking to create self-supervised task
3. **Encoding**: Multi-layer GAT with attention mechanisms
4. **Decoding**: Single-layer GAT to reconstruct original features
5. **Loss**: SCE loss to optimize reconstruction quality
6. **Output**: Anomaly scores based on reconstruction error

This design enables both efficient learning of graph patterns and effective APT detection.
