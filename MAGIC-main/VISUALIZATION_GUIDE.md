# MAGIC: Interactive Graph Analysis & Visualization Guide

## How to Visualize Graph Representations

### 1. Simple Graph Visualization

```python
import matplotlib.pyplot as plt
import networkx as nx
import dgl

def visualize_graph(g, title="Graph Visualization"):
    """
    Convert DGL graph to NetworkX and visualize
    """
    # Convert DGL graph to NetworkX
    g_nx = dgl.to_networkx(g).to_undirected()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layout algorithm (can use spring, circular, etc.)
    pos = nx.spring_layout(g_nx, k=2, iterations=50)
    
    # Draw graph
    nx.draw_networkx_nodes(g_nx, pos, node_color='lightblue', 
                          node_size=500, ax=ax)
    nx.draw_networkx_edges(g_nx, pos, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(g_nx, pos, font_size=8, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Usage
from utils.loaddata import load_batch_level_dataset
dataset = load_batch_level_dataset('streamspot')
graphs = dataset['dataset']
visualize_graph(graphs[0], "StreamSpot Example Graph")
```

### 2. Attention Weight Heatmap

```python
import seaborn as sns
import numpy as np

def visualize_attention_heatmap(model, g, feat, layer_idx=0):
    """
    Visualize attention weights as heatmap
    Shows which edges are important
    """
    model.eval()
    with torch.no_grad():
        # Get first GAT layer
        gat_layer = model.encoder.gats[layer_idx]
        
        # Extract node features and compute attention
        q = gat_layer.fc_q(feat)
        k = gat_layer.fc_k(feat)
        
        # For visualization, use first attention head
        q_h0 = q[:, :gat_layer.out_feat]  # First head
        k_h0 = k[:, :gat_layer.out_feat]
        
        # Compute attention matrix (all-pairs)
        attn_matrix = torch.matmul(q_h0, k_h0.t())
        attn_matrix = attn_matrix.cpu().numpy()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn_matrix, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Attention'})
    ax.set_title(f'GAT Layer {layer_idx} - Attention Head 0')
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    plt.tight_layout()
    plt.show()

# Usage
visualize_attention_heatmap(model, graphs[0], graphs[0].ndata['attr'], layer_idx=0)
```

### 3. Node Embedding Visualization (t-SNE)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings_tsne(model, graphs, labels, sample_size=1000):
    """
    Visualize learned node embeddings using t-SNE
    Shows separation between normal and anomalous nodes
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    count = 0
    with torch.no_grad():
        for g, label in zip(graphs, labels):
            x = g.ndata['attr']
            
            # Get encoder output (last layer)
            embeddings = model.encoder(g, x, return_hidden=False)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend([label] * len(embeddings))
            
            count += len(embeddings)
            if count >= sample_size:
                break
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)[:sample_size]
    all_labels = np.array(all_labels)[:sample_size]
    
    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for label in np.unique(all_labels):
        mask = all_labels == label
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  label=f'Label {label}', alpha=0.6, s=20)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Node Embeddings Visualization (t-SNE)')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Usage (requires labels)
# visualize_embeddings_tsne(model, graphs[:10], labels[:10])
```

### 4. Reconstruction Error Distribution

```python
def plot_reconstruction_errors(model, graphs, labels):
    """
    Show distribution of reconstruction errors
    Normal vs Anomalous samples should be separated
    """
    model.eval()
    errors_normal = []
    errors_anomaly = []
    
    with torch.no_grad():
        for g, label in zip(graphs, labels):
            x = g.ndata['attr']
            
            # Encode
            enc_rep, all_hidden = model.encoder(g, x, return_hidden=True)
            enc_rep = torch.cat(all_hidden, dim=1)
            
            # Decode
            rep = model.encoder_to_decoder(enc_rep)
            x_recon = model.decoder(g, rep)
            
            # Compute error per node
            errors = torch.norm(x_recon - x, dim=1).cpu().numpy()
            graph_error = np.mean(errors)
            
            if label == 0:  # Normal
                errors_normal.append(graph_error)
            else:  # Anomaly
                errors_anomaly.append(graph_error)
    
    # Plot distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors_normal, bins=20, alpha=0.6, label='Normal', color='blue')
    ax.hist(errors_anomaly, bins=20, alpha=0.6, label='Anomalous', color='red')
    
    ax.set_xlabel('Reconstruction Error (Mean L2)')
    ax.set_ylabel('Frequency')
    ax.set_title('Reconstruction Error Distribution by Class')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
# plot_reconstruction_errors(model, graphs, labels)
```

### 5. Per-Layer Feature Analysis

```python
def analyze_per_layer_features(model, g):
    """
    Show how node features transform through each encoder layer
    """
    model.eval()
    x = g.ndata['attr']
    
    with torch.no_grad():
        # Get output from each layer
        _, hidden_list = model.encoder(g, x, return_hidden=True)
    
    # Analyze statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    stats = {
        'layer': [],
        'mean': [],
        'std': [],
        'min': [],
        'max': []
    }
    
    for layer_idx, h in enumerate(hidden_list):
        h_np = h.cpu().numpy()
        stats['layer'].append(layer_idx)
        stats['mean'].append(h_np.mean())
        stats['std'].append(h_np.std())
        stats['min'].append(h_np.min())
        stats['max'].append(h_np.max())
    
    # Plot statistics
    ax = axes[0]
    ax.plot(stats['layer'], stats['mean'], marker='o', label='Mean')
    ax.set_ylabel('Mean Activation')
    ax.set_xlabel('Layer')
    ax.set_title('Mean Activation per Layer')
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    ax.plot(stats['layer'], stats['std'], marker='o', color='orange', label='Std')
    ax.set_ylabel('Std Dev')
    ax.set_xlabel('Layer')
    ax.set_title('Activation Std Dev per Layer')
    ax.grid(alpha=0.3)
    
    ax = axes[2]
    ax.plot(stats['layer'], stats['min'], marker='o', color='red', label='Min')
    ax.plot(stats['layer'], stats['max'], marker='s', color='green', label='Max')
    ax.set_ylabel('Value')
    ax.set_xlabel('Layer')
    ax.set_title('Min/Max Activations per Layer')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[3]
    ax.bar(stats['layer'], [s**2 for s in stats['std']], color='purple', alpha=0.7)
    ax.set_ylabel('Variance')
    ax.set_xlabel('Layer')
    ax.set_title('Variance per Layer')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
# analyze_per_layer_features(model, graphs[0])
```

### 6. Edge Importance Analysis

```python
def analyze_edge_importance(model, g, feat):
    """
    Extract and visualize which edges contribute most to predictions
    """
    model.eval()
    
    # Get all attention weights from first layer
    layer_0 = model.encoder.gats[0]
    
    with torch.no_grad():
        # Compute attention
        q = layer_0.fc_q(feat).view(-1, layer_0.n_heads, layer_0.out_feat)
        k = layer_0.fc_k(feat).view(-1, layer_0.n_heads, layer_0.out_feat)
        
        # For each edge, compute attention
        edge_importance = {}
        
        for src, dst in zip(g.edges()[0], g.edges()[1]):
            src, dst = int(src), int(dst)
            
            # Dot product attention across heads
            attn = (q[src] * k[dst]).sum(dim=-1)  # [n_heads]
            attn = torch.softmax(attn, dim=0)
            
            edge_importance[(src, dst)] = attn.cpu().numpy()
    
    # Get top important edges
    sorted_edges = sorted(
        edge_importance.items(),
        key=lambda x: x[1].mean(),
        reverse=True
    )
    
    print("Top 20 Most Important Edges (by attention):")
    for idx, (edge, attn) in enumerate(sorted_edges[:20]):
        print(f"{idx+1}. Edge {edge}: avg_attn={attn.mean():.4f}")
    
    return edge_importance

# Usage
# edge_imp = analyze_edge_importance(model, graphs[0], graphs[0].ndata['attr'])
```

---

## Comparison with Other Graph Anomaly Detection Methods

```
┌─────────────────────────┬──────────────┬─────────────┬──────────────────┐
│ Method                  │ Learning     │ Graph       │ Interpretability │
├─────────────────────────┼──────────────┼─────────────┼──────────────────┤
│ MAGIC (GAT + Masking)   │ Self-Sup     │ ✓ Full      │ High (attention) │
│ GCN-based               │ Supervised   │ ✓ Full      │ Medium           │
│ GraphSAGE               │ Supervised   │ ✓ Sampling  │ Medium           │
│ Weisfeiler-Lehman       │ Feature Eng  │ ✗ Limited   │ Low              │
│ Random Walk             │ Feature Eng  │ ✓ Full      │ Low              │
│ Statistical (Degrees)   │ None         │ ✗ Limited   │ High             │
└─────────────────────────┴──────────────┴─────────────┴──────────────────┘

MAGIC Advantages:
- Self-supervised: no labeled data needed for pretraining
- Handles graphs of any size
- Interpretable attention weights
- Multi-granularity: node-level and graph-level detection
- Adapts to concept drift
```

---

## Detailed Architecture Flow

### Example: StreamSpot Dataset

```
StreamSpot Provenance Graph:
  - Nodes: 50-500 (system entities)
  - Edges: 100-2000 (interactions)
  - Node features: 15 dimensions
  - Edge features: 2 dimensions

MAGIC Processing:

[Step 1] Masking (30% of nodes)
  - Original: 100 nodes
  - Masked: 30 nodes (replaced with [MASK] token)
  - Visible: 70 nodes (preserved)

[Step 2] GAT Layer 1 (4 heads)
  - Input: [100, 15]
  - Each head learns: [100, 64]
  - Output: [100, 256]

[Step 3] GAT Layer 2 (4 heads)
  - Input: [100, 256]
  - Each head learns: [100, 64]
  - Output: [100, 256]

[Step 4] GAT Layer 3 (4 heads)
  - Input: [100, 256]
  - Each head learns: [100, 64]
  - Output: [100, 256]

[Step 5] Concatenate All Layers
  - [100, 256 * 3] = [100, 768]

[Step 6] Encoder-to-Decoder Projection
  - [100, 768] → [100, 256]

[Step 7] GAT Decoder (1 layer, 4 heads)
  - [100, 256] → [100, 15]

[Step 8] Compute SCE Loss
  - Compare reconstructed[30 masked nodes]
  - vs original[30 masked nodes]
  - Loss = mean SCE over 30 nodes

[Result] Training Signal
  - Backprop through all layers
  - Update attention parameters
  - Update projection matrices
  - Update mask token
```

---

## Multi-Head Attention Breakdown

For StreamSpot example with 4 heads:

```
Query: [100, 256] → Linear(256, 64*4) → [100, 256]
  ├─ Head 1: [100, 64]
  ├─ Head 2: [100, 64]
  ├─ Head 3: [100, 64]
  └─ Head 4: [100, 64]

Key: [100, 256] → Linear(256, 64*4) → [100, 256]
  ├─ Head 1: [100, 64]
  ├─ Head 2: [100, 64]
  ├─ Head 3: [100, 64]
  └─ Head 4: [100, 64]

Value: [100, 256] → Linear(256, 64*4) → [100, 256]
  ├─ Head 1: [100, 64]
  ├─ Head 2: [100, 64]
  ├─ Head 3: [100, 64]
  └─ Head 4: [100, 64]

Attention (per head):
  - Compute: Q @ K^T → [100, 100] attention scores
  - Softmax over neighbors → attention weights
  - Apply to Values: weights @ V → [100, 64] output
  
Concatenate: [100, 64] || [100, 64] || [100, 64] || [100, 64]
           = [100, 256]
```

Each head can learn different patterns:
- Head 1: Process relationships
- Head 2: File access patterns
- Head 3: Network connections
- Head 4: Temporal sequences

---

## Debugging Checklist

```
□ Check data loading
  - Verify graphs loaded correctly
  - Confirm node/edge feature dimensions match config
  - Check for NaN/Inf values in features

□ Verify model initialization
  - Print model architecture
  - Check parameter count
  - Verify device placement (GPU/CPU)

□ Monitor training
  - Loss decreases over epochs
  - Gradients are stable (not NaN/Inf)
  - Weight updates are happening
  - Attention weights are diverse (not all uniform)

□ Evaluate representations
  - Embeddings should separate normal/anomaly
  - Reconstruction error should correlate with labels
  - Per-layer activations should be reasonable
  - Attention weights should focus on relevant edges

□ Check detection quality
  - ROC curve shows discrimination
  - Confusion matrix reasonable
  - False positive rate acceptable
  - Detection latency within bounds
```

---

## Performance Optimization Tips

```python
# Tip 1: Use mixed precision training
from torch.cuda.amp import autocast
with autocast():
    loss = model.compute_loss(graphs)

# Tip 2: Use gradient accumulation for larger effective batches
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model.forward(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Tip 3: Use DGL's GraphDataLoader with workers
from dgl.dataloading import GraphDataLoader
loader = GraphDataLoader(
    graphs, 
    batch_size=32, 
    shuffle=True,
    num_workers=4  # Parallel data loading
)

# Tip 4: Pin memory for faster GPU transfer
loader = GraphDataLoader(
    graphs,
    batch_size=32,
    shuffle=True,
    pin_memory=True
)
```

---

## Summary: What to Visualize

| Visualization | Purpose | How Often |
|---------------|---------|-----------|
| Training loss curve | Monitor convergence | Every epoch |
| Attention heatmaps | Understand edge importance | After training |
| Embedding t-SNE | Verify class separation | After training |
| Reconstruction errors | Check detection signal | After training |
| Per-layer statistics | Debug representation learning | During development |
| Edge importance | Explain model decisions | For interpretability |

MAGIC provides a rich set of learnable components (attention, embeddings, reconstruction) that can be visualized and analyzed to understand what patterns the model discovers in provenance graphs.
