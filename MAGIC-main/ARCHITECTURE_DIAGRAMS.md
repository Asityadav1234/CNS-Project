# MAGIC Architecture Visualization

## Complete Pipeline Architecture

```mermaid
graph TD
    A["Input Provenance Graph<br/>(Nodes + Edges + Features)"] --> B["Feature Extraction<br/>Node: n_dim, Edge: e_dim"]
    B --> C["Random Node Masking<br/>mask_rate = 0.3-0.5"]
    C --> D["Masked Graph<br/>(Some nodes replaced with [MASK] token)"]
    
    D --> E["GAT Encoder<br/>Multi-layer Graph Attention"]
    E --> F["Layer 1: Local Aggregation<br/>hidden_dim × 4 heads"]
    F --> G["Layer 2: 2-hop Patterns<br/>hidden_dim × 4 heads"]
    G --> H["Layer 3: Global Structure<br/>hidden_dim × 4 heads"]
    H --> I["Encoder Output<br/>Concatenated Representations"]
    
    I --> J["Encoder-to-Decoder<br/>Linear Projection"]
    J --> K["GAT Decoder<br/>1-layer attention"]
    K --> L["Reconstructed Features<br/>Shape: [n_nodes, n_dim]"]
    
    A --> M["Original Features<br/>(Masked nodes)"]
    L --> N["Compute SCE Loss<br/>Similarity-based reconstruction loss"]
    M --> N
    
    N --> O["Backpropagation<br/>Update Model Parameters"]
    O --> P["Learned Graph<br/>Representation Model"]
```

## Encoder Architecture (GAT Stack)

```mermaid
graph LR
    A["Input Features<br/>n_dim"] --> B["GATConv Layer 1<br/>4 heads<br/>→ hidden_dim"]
    B --> C["Attention Aggregation<br/>Multi-head"]
    C --> D["BatchNorm + PReLU"]
    D --> E["Residual Connection"]
    E --> F["GATConv Layer 2<br/>4 heads<br/>→ hidden_dim"]
    F --> G["Attention Aggregation<br/>Multi-head"]
    G --> H["BatchNorm + PReLU"]
    H --> I["Residual Connection"]
    I --> J["GATConv Layer 3<br/>4 heads<br/>→ hidden_dim"]
    J --> K["Attention Aggregation<br/>Multi-head"]
    K --> L["Output Embeddings<br/>hidden_dim × 4"]
    L --> M["Concatenate All Layers<br/>hidden_dim × num_layers"]
```

## Attention Head Mechanism

```mermaid
graph TD
    A["Query, Key, Value<br/>from Node Features"] --> B["Multi-Head Split<br/>4 parallel heads"]
    B --> C1["Head 1: Attention"]
    B --> C2["Head 2: Attention"]
    B --> C3["Head 3: Attention"]
    B --> C4["Head 4: Attention"]
    C1 --> D["Compute Attention Weights<br/>softmax(QK^T / √d)"]
    C2 --> D
    C3 --> D
    C4 --> D
    D --> E["Apply to Values<br/>Attention × V"]
    E --> F["Concatenate Heads<br/>[head1 || head2 || head3 || head4]"]
    F --> G["Output Projection<br/>W_out × concatenated"]
    G --> H["Head Output<br/>hidden_dim"]
```

## Masking & Reconstruction Loss

```mermaid
graph TD
    A["Original Graph<br/>All nodes visible"] --> B["Masking Layer<br/>Random 30-50%"]
    B --> C["Masked Nodes"]
    B --> D["Visible Nodes"]
    C --> E["Replace with [MASK] Token<br/>Learnable Parameter"]
    E --> F["Masked Graph"]
    D --> F
    
    F --> G["Encoder<br/>Process entire graph"]
    G --> H["Hidden Representation"]
    
    H --> I["Decoder<br/>Reconstruct features"]
    I --> J["Reconstructed Features<br/>All nodes"]
    
    J --> K["Extract Masked Node Predictions"]
    
    A --> L["Original Masked Node Features"]
    K --> M["SCE Loss Function<br/>loss = ||1 - sim||^α<br/>α=3"]
    L --> M
    
    M --> N["Loss Signal<br/>Drives learning"]
```

## Dataset-Specific Configurations

```mermaid
graph TD
    A["MAGIC Model"] --> B{Dataset Type}
    B -->|StreamSpot| C["hidden_dim: 256<br/>num_layers: 4<br/>max_epoch: 5<br/>batch_size: 12"]
    B -->|Wget| D["hidden_dim: 256<br/>num_layers: 4<br/>max_epoch: 2<br/>batch_size: 1"]
    B -->|DARPA TC| E["hidden_dim: 64<br/>num_layers: 3<br/>max_epoch: 50<br/>batch_size: 1"]
    
    C --> F1["Batch-Level Detection<br/>Entire attack sessions"]
    D --> F1
    E --> F2["Entity-Level Detection<br/>Individual compromised entities"]
```

## Training vs Inference

```mermaid
graph TD
    subgraph Training
        A["Masked Graphs"] --> B["Encoder<br/>Learn representations"]
        B --> C["Decoder<br/>Reconstruct features"]
        C --> D["SCE Loss<br/>Optimization"]
        D --> E["Updated Model<br/>Saved checkpoint"]
    end
    
    subgraph Inference
        F["Test Graphs<br/>(Full, no masking)"] --> G["Encoder<br/>Generate embeddings"]
        G --> H["Decoder<br/>Reconstruct"]
        H --> I["Reconstruction Error"]
        I --> J["KNN Outlier Detection"]
        J --> K["Anomaly Score"]
        K --> L{Score > Threshold?}
        L -->|Yes| M["Anomalous"]
        L -->|No| N["Benign"]
    end
```

## Message Passing in GATConv

```mermaid
graph TD
    A["Neighbor Nodes<br/>h_j @ Layer l"] --> B["Transform by Weight Matrix<br/>W × h_j"]
    B --> C["Compute Attention Score<br/>softmax(attention_weight)"]
    C --> D["Weight Aggregation<br/>α_ij × (W × h_j)"]
    E["Center Node<br/>i"] --> D
    D --> F["Sum Weighted Messages<br/>Σ α_ij(W × h_j)"]
    F --> G["Apply Activation<br/>ReLU(·)"]
    G --> H["Output Node Embedding<br/>h_i @ Layer l+1"]
    
    I["Edge Features"] -.-> C
    I -.-> B
```

## Anomaly Detection Pipeline

```mermaid
graph TD
    A["Benign System Logs<br/>Training Phase"] --> B["Train MAGIC Model<br/>Learn normal behavior patterns"]
    B --> C["Model Checkpoint<br/>Saved representations"]
    
    C --> D["Test System Logs<br/>Unknown or test data"]
    D --> E["Encode with Trained Model<br/>Get node embeddings"]
    E --> F["Measure Reconstruction Error<br/>per node and graph"]
    F --> G["KNN in Embedding Space<br/>Find similar nodes"]
    G --> H["Compute Anomaly Score<br/>Distance to k-nearest neighbors"]
    
    H --> I{Multi-Granularity<br/>Scoring}
    I -->|Node Level| J["Entity Anomaly Score"]
    I -->|Graph Level| K["Batch/Session Anomaly Score"]
    
    J --> L["Outlier Detection<br/>Statistical threshold"]
    K --> L
    
    L --> M{Detection Result}
    M -->|High Score| N["🚨 APT Detected<br/>Compromised Entity/Session"]
    M -->|Low Score| O["✓ Benign<br/>Normal behavior"]
```

## Feature Flow Through Model

```mermaid
graph LR
    A["Node Features<br/>Dimension: n_dim"] --> B["GAT Layer 1<br/>linear: n_dim → hidden_dim/4<br/>attention: 4 heads"]
    B --> C["hidden_dim<br/>4 heads"]
    C --> D["Concat heads<br/>→ hidden_dim"]
    D --> E["GAT Layer 2<br/>linear: hidden_dim → hidden_dim/4<br/>attention: 4 heads"]
    E --> F["hidden_dim<br/>4 heads"]
    F --> G["Concat heads<br/>→ hidden_dim"]
    G --> H["GAT Layer 3<br/>linear: hidden_dim → hidden_dim/4<br/>attention: 4 heads"]
    H --> I["hidden_dim<br/>4 heads"]
    I --> J["Concat: All Layers<br/>hidden_dim × num_layers"]
    J --> K["Decoder Projection<br/>linear: hidden_dim×num_layers → hidden_dim"]
    K --> L["Decoder Layer<br/>hidden_dim → n_dim"]
    L --> M["Reconstructed Features<br/>Dimension: n_dim"]
```

## Key Design Patterns

```mermaid
graph TD
    A["Design Decision"] --> B{Pattern}
    B -->|Multi-layer GAT| C["Multiple layers for<br/>different receptive fields"]
    C --> C1["Layer 1: 1-hop neighbors"]
    C --> C2["Layer 2: 2-hop patterns"]
    C --> C3["Layer 3+: Global structure"]
    
    B -->|Masked Learning| D["Hide information<br/>Learn from context"]
    D --> D1["Prevents memorization"]
    D --> D2["Builds robust features"]
    
    B -->|Residual Connections| E["Skip connections<br/>Preserve information"]
    E --> E1["Gradient flow"]
    E --> E2["Feature reuse"]
    
    B -->|Attention Heads| F["Multiple attention subspaces<br/>Diverse interactions"]
    F --> F1["Head 1: edge type patterns"]
    F --> F2["Head 2: frequency patterns"]
    F --> F3["Head 3-4: other aspects"]
    
    B -->|SCE Loss| G["Normalized similarity loss<br/>Robustness to magnitude"]
    G --> G1["Handles outliers well"]
    G --> G2["Stable gradients"]
```
