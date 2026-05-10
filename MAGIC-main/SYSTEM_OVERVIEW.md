# MAGIC: Complete System Architecture Diagram

## 1. Full Pipeline: From Raw Data to Anomaly Detection

```mermaid
graph TD
    subgraph Input["Input Layer"]
        A["Provenance Graph<br/>System Logs"]
        B["Node Features<br/>n_dim dimensions"]
        C["Edge Features<br/>e_dim dimensions"]
    end
    
    subgraph Preprocessing["Preprocessing"]
        D["Graph Parsing<br/>Create DGL Graph"]
        E["Feature Extraction<br/>Normalize Features"]
        F["Train/Test Split<br/>Batch Creation"]
    end
    
    subgraph Training["Training Phase"]
        G["Random Masking<br/>mask_rate %"]
        H["GAT Encoder<br/>Multi-layer"]
        I["Hidden Representation<br/>Multi-level features"]
        J["Projection Layer<br/>Decoder Preparation"]
        K["GAT Decoder<br/>1 layer"]
        L["Reconstruction<br/>Original dimensions"]
        M["SCE Loss<br/>Contrastive Learning"]
        N["Backpropagation<br/>Update Parameters"]
    end
    
    subgraph Inference["Inference Phase"]
        O["Test Graph<br/>Full no masking"]
        P["Encode<br/>Get embeddings"]
        Q["Decode<br/>Reconstruct"]
        R["Reconstruction Error<br/>Per-node metric"]
        S["KNN Outlier Detection<br/>Embedding space"]
        T["Anomaly Score<br/>Composite metric"]
    end
    
    subgraph Output["Output Layer"]
        U{Score > Threshold?}
        V["Normal<br/>Benign Behavior"]
        W["Anomalous<br/>APT Detected"]
    end
    
    A --> D
    B --> E
    C --> E
    D --> F
    E --> F
    
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N -->|Update| H
    N -->|Update| K
    N -->|Update| J
    
    F --> O
    O --> P
    P --> Q
    Q --> R
    R --> S
    S --> T
    
    T --> U
    U -->|Low| V
    U -->|High| W
```

---

## 2. Graph Attention Network (GAT) Layer Details

```mermaid
graph LR
    subgraph Input["Input"]
        A["Node Features<br/>[N, d_in]"]
    end
    
    subgraph Transform["Linear Transform"]
        B["Project to Q<br/>Linear(d_in, d_out × h)"]
        C["Project to K<br/>Linear(d_in, d_out × h)"]
        D["Project to V<br/>Linear(d_in, d_out × h)"]
    end
    
    subgraph MultiHead["Multi-Head Attention"]
        E1["Head 1"]
        E2["Head 2"]
        E3["Head 3"]
        E4["Head 4"]
    end
    
    subgraph Attention["Attention Computation"]
        F["Compute Q·K^T<br/>Attention scores"]
        G["Softmax<br/>Normalize weights"]
        H["Apply to Values<br/>weighted sum"]
    end
    
    subgraph Combine["Combine Heads"]
        I["Concatenate<br/>Head outputs"]
        J["Project Output<br/>W_out"]
    end
    
    subgraph Output["Output"]
        K["Node Embeddings<br/>[N, d_out × h]"]
    end
    
    A --> B
    A --> C
    A --> D
    
    B --> E1
    B --> E2
    B --> E3
    B --> E4
    
    C --> F
    D --> H
    
    E1 --> F
    E2 --> F
    E3 --> F
    E4 --> F
    
    F --> G
    G --> H
    
    E1 --> I
    E2 --> I
    E3 --> I
    E4 --> I
    
    I --> J
    J --> K
```

---

## 3. Masked Autoencoder Architecture

```mermaid
graph TB
    A["Input Graph<br/>All nodes visible"] --> B["Masking Layer<br/>Mask k% nodes"]
    B --> C["Masked Graph<br/>Some [MASK] tokens"]
    
    C --> D["GAT Encoder Layer 1"]
    D --> E1["[N, hidden_dim]"]
    
    E1 --> F["GAT Encoder Layer 2"]
    F --> E2["[N, hidden_dim]"]
    
    E2 --> G["GAT Encoder Layer 3"]
    G --> E3["[N, hidden_dim]"]
    
    E1 --> H["Concatenate<br/>All Layers"]
    E2 --> H
    E3 --> H
    
    H --> I["[N, hidden_dim × n_layers]"]
    I --> J["Linear Projection<br/>→ hidden_dim"]
    J --> K["[N, hidden_dim]"]
    
    K --> L["GAT Decoder"]
    L --> M["[N, n_dim]"]
    
    M --> N["Extract Masked Nodes<br/>Predictions"]
    
    A --> O["Extract Masked Nodes<br/>Original Features"]
    
    N --> P["SCE Loss<br/>Compare"]
    O --> P
    
    P --> Q["Loss Signal"]
    Q --> R["Backpropagation"]
    R --> S["Update All<br/>Parameters"]
```

---

## 4. Loss Function: SCE (Symmetric Cross Entropy)

```mermaid
graph LR
    A["Predicted Features<br/>x_recon"] --> B["Normalize<br/>||x|| = 1"]
    C["Target Features<br/>x_original"] --> D["Normalize<br/>||x|| = 1"]
    
    B --> E["Compute Similarity<br/>x_norm · y_norm"]
    D --> E
    
    E --> F["1 - similarity"]
    F --> G["Power by alpha<br/>(1-sim)^α"]
    
    G --> H["Mean over<br/>masked nodes"]
    H --> I["Final Loss"]
```

---

## 5. Complete Layer-by-Layer Feature Dimensions

### StreamSpot Example (num_layers=4, num_heads=4, hidden_dim=256)

```
INPUT: Provenance Graph
├─ Nodes: 100-500
├─ Node Features: [N, 15]
├─ Edge Features: [E, 2]
└─ Edges: 200-2000

MASKING (30%)
├─ Mask Tokens: Create [1, 15] learnable parameter
├─ Replace 30 nodes with [MASK] token
└─ Result: Modified graph for encoder

GAT ENCODER - LAYER 1
├─ Input: [N, 15]
├─ Linear projections: 15 → 64×4 (4 heads)
├─ Per-head computations: [N, 64] per head
├─ Output attention: softmax over edges
├─ Value aggregation: [N, 64] per head
├─ Concatenate heads: [N, 256]
└─ Output: [N, 256]

GAT ENCODER - LAYER 2
├─ Input: [N, 256]
├─ Linear projections: 256 → 64×4
├─ Multi-head attention: same mechanism
└─ Output: [N, 256]

GAT ENCODER - LAYER 3
├─ Input: [N, 256]
├─ Linear projections: 256 → 64×4
├─ Multi-head attention: same mechanism
└─ Output: [N, 256]

GAT ENCODER - LAYER 4
├─ Input: [N, 256]
├─ Linear projections: 256 → 64×4
├─ Multi-head attention: same mechanism
└─ Output: [N, 256]

CONCATENATE ENCODER LAYERS
├─ Cat([Layer1, Layer2, Layer3, Layer4])
├─ [N, 256] || [N, 256] || [N, 256] || [N, 256]
└─ Output: [N, 1024]

ENCODER-TO-DECODER PROJECTION
├─ Linear(1024, 256)
└─ Output: [N, 256]

GAT DECODER - LAYER 1
├─ Input: [N, 256]
├─ Linear projections: 256 → hidden_dim
├─ Attention computation (1 layer)
├─ Output projection: → 15
└─ Output: [N, 15]

LOSS COMPUTATION
├─ Extract predictions[masked_indices]: [M, 15]
├─ Extract targets[masked_indices]: [M, 15]
├─ SCE Loss: normalize & compute (1-sim)^3
└─ Final Loss: scalar value
```

---

## 6. Training Loop State Machine

```mermaid
stateDiagram-v2
    [*] --> Initialize
    
    Initialize --> LoadData
    LoadData --> BuildModel
    
    BuildModel --> EpochStart
    
    EpochStart --> BatchLoop
    
    BatchLoop --> MaskNodes: For each batch
    MaskNodes --> Encode: Forward pass
    Encode --> Concat: Combine layers
    Concat --> Project: Linear projection
    Project --> Decode: Decoder forward
    Decode --> ComputeLoss: Compare predictions
    ComputeLoss --> Backward: Backpropagation
    Backward --> UpdateParams: Optimizer step
    
    UpdateParams --> BatchComplete{More batches?}
    BatchComplete -->|Yes| BatchLoop
    BatchComplete -->|No| EpochComplete
    
    EpochComplete --> CheckConvergence{Converged?}
    CheckConvergence -->|No| EpochStart
    CheckConvergence -->|Yes| SaveModel
    
    SaveModel --> [*]
```

---

## 7. Inference Pipeline

```mermaid
graph LR
    A["Test Graph<br/>Complete no masking"] --> B["Encoder"]
    B --> C["Multi-layer Hidden<br/>Representations"]
    C --> D["Concatenate<br/>All layers"]
    D --> E["Project to<br/>Decoder space"]
    E --> F["Decoder"]
    F --> G["Reconstructed<br/>Features"]
    
    A --> H["Original<br/>Features"]
    G --> I["Reconstruction<br/>Error"]
    H --> I
    
    I --> J["Per-node<br/>errors"]
    
    J --> K["Aggregate to<br/>graph-level"]
    
    K --> L["KNN Search<br/>in embedding space"]
    
    L --> M["Distance to<br/>k-nearest"]
    
    M --> N["Anomaly<br/>Score"]
    
    N --> O{Threshold}
    O -->|High| P["ALERT:<br/>Anomaly"]
    O -->|Low| Q["OK:<br/>Normal"]
```

---

## 8. Dataset Configuration Matrix

```
┌──────────────┬─────────┬─────────┬──────────┬────────────┬──────────┐
│ Dataset      │ n_dim   │ e_dim   │ n_layers │ hidden_dim │ batch_sz │
├──────────────┼─────────┼─────────┼──────────┼────────────┼──────────┤
│ StreamSpot   │ 15      │ 2       │ 4        │ 256        │ 12       │
│ Wget         │ 12      │ 2       │ 4        │ 256        │ 1        │
│ Trace        │ 40      │ 5       │ 3        │ 64         │ 1        │
│ Theia        │ 40      │ 5       │ 3        │ 64         │ 1        │
│ Cadets       │ 40      │ 5       │ 3        │ 64         │ 1        │
└──────────────┴─────────┴─────────┴──────────┴────────────┴──────────┘
```

---

## 9. Model Parameter Count Estimation

```
For StreamSpot Configuration:
  n_dim=15, e_dim=2, hidden_dim=256, n_layers=4, n_heads=4

GAT Encoder (each layer):
  ├─ Q projection: 256 × 256 + 256 = 65,792
  ├─ K projection: 256 × 256 + 256 = 65,792
  ├─ V projection: 256 × 256 + 256 = 65,792
  ├─ Edge projection: 2 × 4 + 4 = 12
  ├─ Attention params: 1 × 4 × 64 = 256
  └─ Output projection: (256×4) × (256×4) = 1,048,576

Per GAT layer: ~1.25M parameters
Encoder (4 layers): ~5M parameters

GAT Decoder (1 layer):
  ├─ Q, K, V projections: ~66K each = 198K
  ├─ Output projection: 256×256 + bias = 65,792
  └─ Total: ~265K parameters

Encoder-to-Decoder: 1024 × 256 = 262,144 parameters
Mask token: 1 × 15 = 15 parameters

Total: ~5.5M parameters
```

---

## 10. Computational Complexity Analysis

```
For a graph with N nodes and E edges:

Forward Pass:
├─ Per GAT layer: O(E + N)
│  ├─ Linear transforms: O(N × d)
│  ├─ Attention: O(E + N log N) with softmax
│  └─ Aggregation: O(E)
├─ All encoder layers: O(num_layers × (E + N))
├─ Decoder layer: O(E + N)
└─ Total: O(num_layers × E)

Memory Usage:
├─ Node features: O(N × hidden_dim)
├─ All layer outputs: O(N × hidden_dim × num_layers)
├─ Attention weights: O(E)
└─ Total: O(N × hidden_dim × num_layers)

For StreamSpot: N~500, E~2000, hidden_dim=256
├─ Forward: ~24K operations
├─ Memory: ~512KB activations
└─ GPU memory: ~10GB for batch of 12 graphs
```

---

## 11. Anomaly Detection Decision Boundary

```
Reconstruction Error Threshold Determination:

1. Train on Benign Data
   ├─ Compute reconstruction errors
   ├─ Distribution: typically log-normal or exponential
   └─ Mean ≈ 0.1-0.5 depending on dataset

2. Analyze Error Distribution
   ├─ Normal data: low errors (μ ≈ 0.2, σ ≈ 0.1)
   ├─ Anomalous data: high errors (μ ≈ 0.8, σ ≈ 0.3)
   └─ Overlap region: 0.4-0.6

3. Set Detection Threshold
   ├─ Option 1: μ_normal + 3σ (99.7% detection)
   ├─ Option 2: ROC curve knee point (maximize TPR-FPR)
   └─ Option 3: Percentile cutoff (e.g., 95th percentile)

4. Multi-Granularity Scoring
   ├─ Node-level: individual node errors
   ├─ Edge-level: edge reconstruction (optional)
   └─ Graph-level: mean/max/weighted aggregate
```

---

## 12. Key Formulas Reference

### Attention Mechanism
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$

### Multi-head Attention
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h) \cdot W^O$$
where $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$

### SCE Loss
$$\mathcal{L}_{SCE} = \text{mean}\left[(1 - \cos(x, y))^{\alpha}\right]$$
where $\cos(x,y) = \frac{x \cdot y}{||x|| \cdot ||y||}$

### Node Embedding Update (GAT)
$$h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(l)} W^{(l)} h_j^{(l)}\right)$$

### Reconstruction Error
$$E = \frac{1}{N} \sum_{i=1}^{N} ||x_i^{\text{recon}} - x_i^{\text{orig}}||_2$$

### KNN Anomaly Score
$$A_i = \frac{1}{k} \sum_{j \in kNN(i)} ||e_i - e_j||_2$$

---

## 13. Debugging Decision Tree

```
                    Training Not Converging?
                           |
                  _____ ___|___ _____
                 |              |
            Loss oscillates   Loss stuck
                 |              |
              Check:         Check:
            • Learning      • Learning
              rate          rate too small
            • Batch size    • Data quality
            • Gradients     • Model capacity
                 |              |
            Reduce LR       Increase LR
            or add LR       or add 
            scheduling      regularization
```

---

## Summary Table: Model Component Roles

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| Masking Layer | Graph | Masked Graph | Create self-supervised task |
| GAT Encoder L1 | [N, n_dim] | [N, h] | Local feature learning |
| GAT Encoder L2 | [N, h] | [N, h] | 2-hop patterns |
| GAT Encoder L3 | [N, h] | [N, h] | Global structure |
| Concatenation | 3×[N, h] | [N, 3h] | Multi-level features |
| Encoder-to-Decoder | [N, 3h] | [N, h] | Dimensionality match |
| GAT Decoder | [N, h] | [N, n_dim] | Feature reconstruction |
| SCE Loss | [M, n_dim] pairs | scalar | Training signal |

MAGIC processes provenance graphs through a sequence of learned transformations that balance depth (multiple layers), breadth (multi-head attention), and interpretability (attention weights).
