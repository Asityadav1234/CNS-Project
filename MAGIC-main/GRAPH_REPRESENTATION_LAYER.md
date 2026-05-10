# MAGIC: Graph Representation Layer Analysis

## Paper: Detecting Advanced Persistent Threats via Masked Graph Representation Learning
**USENIX Security 2024**

---

## 1. Model Architecture Overview

### Core Components:

```
Input Provenance Graph
        ↓
   [Graph Masking]
        ↓
   [GAT Encoder] → Hidden Representations
        ↓
   [Encoder-to-Decoder Projection]
        ↓
   [GAT Decoder] → Reconstructed Node Features
        ↓
   [SCE Loss] → Self-supervised Learning Signal
```

---

## 2. Graph Representation Layer (Encoder)

### Architecture:
- **Type**: Multi-layer Graph Attention Network (GAT)
- **Input Dimensions**:
  - Node features: `n_dim` (varies by dataset)
  - Edge features: `e_dim` (varies by dataset)
- **Configuration**:
  - Number of layers: 3-4 (dataset dependent)
  - Hidden dimension: 64-256 nodes
  - Attention heads: 4 heads per layer
  - Activation: PReLU
  - Normalization: BatchNorm
  - Dropout: Feature dropout 0.1

### Key Features:

1. **Multi-head Attention Mechanism**:
   - Each head learns different aspects of the graph structure
   - Attention weights computed between neighboring nodes
   - Edge features integrated into attention computation

2. **Residual Connections**:
   - Skip connections between layers
   - Helps preserve important features during deep propagation

3. **Hierarchical Feature Learning**:
   - Layer 1: Local neighborhood aggregation
   - Layer 2: 2-hop neighborhood patterns
   - Layer 3+: Global graph structure

### Output:
- **Node embeddings**: `[num_nodes, hidden_dim]`
- **Multi-level representations**: Concatenated from all layers

---

## 3. Masked Graph Representation Learning

### Masking Strategy:

```
Original Node Features
        ↓
[Randomly mask k% of nodes] ← mask_rate = 0.3-0.5
        ↓
Replace with [MASK] token
        ↓
Pass to Encoder
```

### Implementation Details:

```python
def encoding_mask_noise(g, mask_rate=0.3):
    # Randomly select mask_rate % of nodes
    num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]
    
    # Replace masked node features with learnable mask token
    new_g.ndata["attr"][mask_nodes] = enc_mask_token
    
    return masked_graph, (mask_nodes, keep_nodes)
```

### Why Masking?
- Forces the encoder to learn robust representations
- Encoder learns to predict masked node features from neighbors
- Encourages relational learning rather than memorization
- Detects anomalies by identifying nodes with poor reconstruction

---

## 4. Decoder & Reconstruction

### Decoder Architecture:
- **Type**: Single-layer GAT
- **Purpose**: Reconstruct original node features from hidden representations
- **Input**: Encoder hidden states (concatenated from all encoder layers)
- **Output**: Reconstructed node features matching original dimensions

### Process:

```
Encoder Output [hidden_dim * num_layers]
        ↓
[Linear Projection] → [hidden_dim]
        ↓
[Single-layer GAT Decoder]
        ↓
Reconstructed Features [n_dim]
```

### Edge Reconstruction:
- Additional FC layer for edge reconstruction
- Input: Concatenated source and destination node embeddings
- Output: Edge presence probability [0, 1]

---

## 5. Loss Functions

### SCE Loss (Symmetric Cross Entropy):

```python
def sce_loss(x, y, alpha=3):
    # Normalize embeddings
    x = normalize(x, p=2)
    y = normalize(y, p=2)
    
    # Compute similarity-based loss
    loss = (1 - (x * y).sum(dim=-1)).pow(alpha)
    return loss.mean()
```

### Key Properties:
- **Symmetric**: Treats both x and y equally
- **Smooth gradient**: Exponential term (alpha=3) provides stable learning
- **Contrastive**: Maximizes similarity between original and reconstructed features
- **Robustness**: Reduces impact of outliers

### Training Objective:
- Minimize reconstruction error on masked nodes
- Learn representations that preserve graph structure and semantics

---

## 6. Implementation Details by Dataset

### StreamSpot & Wget (Batch-level Detection):
```
num_hidden = 256
num_layers = 4
max_epoch = 2-5
batch_size = 1-12
```

### DARPA TC (Trace, Theia, Cadets - Entity-level Detection):
```
num_hidden = 64
num_layers = 3
max_epoch = 50
batch_size = 1
```

---

## 7. Anomaly Detection Pipeline

### After Training:

```
Unseen Provenance Graph
        ↓
[GAT Encoder] → Hidden representations
        ↓
[GAT Decoder] → Reconstructed features
        ↓
[Compute Reconstruction Error]
        ↓
[Outlier Detection (KNN)] → Anomaly Score
        ↓
Compare with Threshold → Normal / Anomalous
```

### Detection Methods:
1. **Node-level**: Individual node reconstruction errors
2. **Graph-level**: Aggregated graph reconstruction errors
3. **KNN-based outlier detection**: Uses learned representations in feature space

---

## 8. Key Innovations in MAGIC

### 1. **Multi-Granularity Representation**
   - Captures both local and global graph patterns
   - Hierarchical feature extraction through multiple GAT layers

### 2. **Self-Supervised Learning**
   - No labeled anomalies required for pre-training
   - Learns from natural graph structure

### 3. **Masked Learning Strategy**
   - Inspired by BERT-style masking in NLP
   - Forces encoder to learn from context
   - Robust feature learning

### 4. **Graph-Aware Architecture**
   - Edge features influence node representations
   - Attention weights capture interaction importance
   - Structure-preserving reconstruction

### 5. **Concept Drift Handling**
   - Model adaptation mechanism for evolving benign behaviors
   - Online learning capabilities

---

## 9. Mathematical Formulation

### Graph Attention Mechanism:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

### Node Representation Update:
$$h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W^{(l)} h_j^{(l)}\right)$$

Where:
- $\alpha_{ij}$ = attention weight from node i to node j
- $W^{(l)}$ = learnable weight matrix at layer l
- $\mathcal{N}(i)$ = neighbors of node i

### Reconstruction Loss:
$$\mathcal{L} = \text{SCE}(\text{Decoder}(\text{Encoder}(G_{masked})), \text{Original}(G))$$

---

## 10. Performance Characteristics

### Advantages:
- **Flexible**: Works with varying graph sizes and structures
- **Efficient**: DGL optimizations for large-scale graphs
- **Interpretable**: Attention weights show important edges
- **Unsupervised**: Doesn't require labeled APT examples

### Computational Complexity:
- **Forward pass**: O(E) where E = number of edges
- **Attention computation**: O(N²) in worst case, O(E) with sparse graphs
- **Memory**: O(N × hidden_dim)

---

## 11. Dataset-Specific Graphs

### StreamSpot:
- **Nodes**: System entities (processes, files, network connections)
- **Edges**: System calls and their sequences
- **Features**: Entity type, attributes, behavior patterns

### DARPA TC:
- **Nodes**: Subject, Object, Properties
- **Edges**: Events (read, write, execute, connect)
- **Features**: Detailed forensic information

### Detection Scenario:
- **Entity-level**: Detect individual compromised entities
- **Batch-level**: Detect entire attack sessions

---

## 12. Model Training Flow

```python
# 1. Load and preprocess graphs
graphs = load_batch_level_dataset(dataset_name)

# 2. Initialize model
model = build_model(args)  # GMAEModel with GAT encoder/decoder

# 3. Training loop
for epoch in range(max_epochs):
    for batch_graphs in dataloader:
        # Apply masking
        masked_graphs, (mask_nodes, keep_nodes) = model.encoding_mask_noise(batch_graphs)
        
        # Forward pass
        loss = model.compute_loss(masked_graphs)
        
        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Inference and anomaly detection
encodings = []
for test_graph in test_dataset:
    hidden_rep = model.encoder(test_graph, test_graph.ndata['attr'])
    # Compute reconstruction error
    # Apply outlier detection
    anomaly_score = compute_anomaly_score(hidden_rep)
```

---

## Summary

MAGIC implements a sophisticated **masked graph auto-encoder** architecture that:

1. **Encodes** provenance graphs using multi-layer GAT with attention mechanisms
2. **Learns** robust representations through masked node prediction
3. **Decodes** to reconstruct original node features and edges
4. **Detects** anomalies by identifying nodes/graphs with poor reconstruction
5. **Adapts** to concept drift in benign system behaviors

The graph representation layer is the core innovation, enabling both accurate APT detection and adaptability to evolving threat landscapes.
