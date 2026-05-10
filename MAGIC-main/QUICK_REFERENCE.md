# MAGIC: Quick Reference Card

## Paper Info
- **Title**: MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning
- **Conference**: USENIX Security 2024
- **Core Innovation**: Self-supervised masked graph learning for APT detection

---

## Quick Architecture Summary

```
Graph Input → Mask 30-50% Nodes → GAT Encoder (3-4 layers) 
→ Concatenate Representations → Project → GAT Decoder (1 layer)
→ Reconstruct Features → Compare with Original → SCE Loss → Train
```

---

## Dataset Quick Reference

| Dataset | Detection Level | Nodes | Features | Config |
|---------|-----------------|-------|----------|--------|
| StreamSpot | Batch | 50-500 | n=15, e=2 | L4, H256 |
| Wget | Batch | 100-1000 | n=12, e=2 | L4, H256 |
| Trace | Entity | 1000-10K | n=40, e=5 | L3, H64 |
| Theia | Entity | 1000-10K | n=40, e=5 | L3, H64 |
| Cadets | Entity | 1000-10K | n=40, e=5 | L3, H64 |

---

## Key Files

| File | Purpose |
|------|---------|
| `model/autoencoder.py` | GMAEModel, masking, loss |
| `model/gat.py` | GAT encoder/decoder layers |
| `model/loss_func.py` | SCE loss function |
| `train.py` | Training pipeline |
| `eval.py` | Evaluation & detection |
| `utils/loaddata.py` | Data loading & parsing |

---

## Model Parameters Cheat Sheet

```python
# Encoder Configuration
n_layers = 3 or 4           # 3 for TC, 4 for StreamSpot/Wget
n_heads = 4                 # Fixed in MAGIC
hidden_dim = 64 or 256      # 256 for batch-level, 64 for entity-level
mask_rate = 0.3-0.5         # Masking ratio

# Loss Function
alpha_l = 2-3               # SCE exponent (higher = stricter)

# Training
max_epochs = 2-50           # Depends on dataset size
feat_drop = 0.1             # Input feature dropout
negative_slope = 0.2        # PReLU slope
```

---

## Training Loop (3 lines)

```python
for epoch in range(max_epochs):
    for graphs in dataloader:
        loss = model(graphs); loss.backward(); optimizer.step()
```

---

## Inference Loop (5 lines)

```python
model.eval()
with torch.no_grad():
    embeddings = model.encoder(g, g.ndata['attr'])
    reconstructed = model.decoder(g, embeddings)
    error = (reconstructed - g.ndata['attr']).norm(dim=1)
```

---

## Hyperparameter Tuning Guide

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Learning rate | 0.001 | 0.0001-0.01 | Convergence speed |
| Mask rate | 0.4 | 0.2-0.7 | Learning difficulty |
| hidden_dim | 64-256 | 32-512 | Model capacity |
| n_layers | 3-4 | 2-6 | Receptive field |
| feat_drop | 0.1 | 0-0.5 | Regularization |

---

## Common Commands

```bash
# Quick evaluation (pre-trained)
python eval.py --dataset trace

# Standard evaluation (train detector only)
rm -rf eval_result/*
python eval.py --dataset trace

# Full training from scratch
rm -rf checkpoints/* eval_result/*
python train.py --dataset trace
python eval.py --dataset trace

# Check configurations
grep "def build_args" utils/config.py
```

---

## Dimension Flow (StreamSpot Example)

```
Input Graphs:
[batch_size, num_nodes] → features: [num_nodes, 15]

After Masking:
30% masked nodes → [MASK] token: [1, 15]

GAT Encoder:
[N, 15] → [N, 256] → [N, 256] → [N, 256] → [N, 256]

Concatenate:
[N, 1024]

Project:
[N, 256]

GAT Decoder:
[N, 15]

Loss:
Mean of 30 masked node errors
```

---

## Performance Targets (Expected Results)

| Metric | StreamSpot | Wget | DARPA TC |
|--------|-----------|------|----------|
| ROC-AUC | ~0.95 | ~0.98 | ~0.90 |
| F1-Score | ~0.92 | ~0.95 | ~0.88 |
| Training time | ~5 min | ~2 min | ~2-4 hours |
| Detection latency | <100ms | <50ms | <500ms |

---

## Troubleshooting Checklist

- [ ] Data loads without errors (check dimensions)
- [ ] Model parameters match dataset config
- [ ] Training loss decreases smoothly
- [ ] No NaN/Inf gradients
- [ ] Attention weights are diverse (not uniform)
- [ ] Validation curves similar to training
- [ ] Detection ROC curve shows discrimination
- [ ] Memory usage reasonable for hardware

---

## Key Innovation: Masking

**Traditional Autoencoders**: Learn to reconstruct perfectly → memorization
**MAGIC**: Mask nodes → predict missing → force relational learning
**Result**: Robust representations, better anomaly detection

---

## Loss Function: Why SCE?

$$\mathcal{L} = \text{mean}\left[(1 - \cos(x, y))^{3}\right]$$

- **Normalized**: Scale-invariant
- **Smooth**: Stable gradients
- **Robust**: Outliers less impactful
- **Contrastive**: Maximizes similarity

---

## Attention Weight Interpretation

Each of 4 attention heads learns different patterns:
- **Head 1**: Main relationship types
- **Head 2**: Secondary patterns
- **Head 3**: Frequency/temporal info
- **Head 4**: Boundary/anomalous signals

Sum of attention across edges for a node = how much information it receives

---

## Detection Thresholds

```
Benign graphs:    reconstruction error ~ 0.2 (std 0.1)
Anomalous graphs: reconstruction error ~ 0.8 (std 0.3)

Typical threshold: 0.5
- Below: Normal (low false alarm)
- Above: Alert (high confidence)

Fine-tune using ROC curve or precision-recall tradeoff
```

---

## Memory Requirements

| Configuration | GPU Memory | CPU RAM |
|---------------|-----------|---------|
| Single graph evaluation | ~1 GB | ~2 GB |
| Batch of 12 graphs (StreamSpot) | ~8 GB | ~4 GB |
| Full training (DARPA TC) | ~12 GB | ~8 GB |

---

## Recommended Development Flow

1. Start with StreamSpot (smallest, fastest)
2. Verify training loss curves
3. Check detection performance on validation set
4. Tune hyperparameters if needed
5. Scale to DARPA TC datasets
6. Compare attention patterns across anomaly types
7. Deploy with appropriate threshold

---

## When to Use MAGIC

✓ Detection of system-level anomalies
✓ Provenance graph analysis
✓ Unknown attack detection
✓ Need interpretability (attention)
✗ Real-time streaming (batch-based)
✗ Very small graphs (<10 nodes)
✗ Graphs with >50K nodes per batch

---

## Research Paper Key Points

1. **Self-supervised learning**: No labeled APT data needed for pre-training
2. **Multi-granularity**: Node-level and graph-level detection
3. **Concept drift**: Model adaptation mechanism for evolving benign behavior
4. **Interpretability**: Attention weights show important edges
5. **Efficiency**: DGL optimizations for large-scale graphs
6. **Universality**: Works across different system monitoring formats

---

## Code Example: End-to-End

```python
# 1. Setup
from model.autoencoder import build_model
from utils.loaddata import load_batch_level_dataset
from utils.config import build_args

# 2. Load data
args = build_args()
dataset = load_batch_level_dataset(args.dataset)
graphs = dataset['dataset']

# 3. Build model
model = build_model(args)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Train
for epoch in range(50):
    for g in graphs:
        loss = model(g)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss={loss:.4f}")

# 5. Evaluate
model.eval()
with torch.no_grad():
    embeddings = model.encoder(test_graph, test_graph.ndata['attr'])
    recon = model.decoder(test_graph, embeddings)
    error = (recon - test_graph.ndata['attr']).norm(dim=1)
    anomaly_score = error.mean()
    if anomaly_score > threshold:
        print("ALERT: Anomaly detected!")
```

---

## Citation

```bibtex
@inproceedings{jia2024magic,
  title={MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning},
  author={Jia, Zian and Xiong, Yun and Nan, Yuhong and Zhang, Yao and Zhao, Jinjing and Wen, Mi},
  booktitle={33rd USENIX Security Symposium},
  year={2024}
}
```

---

## Resources

- **GAT Paper**: Graph Attention Networks (Veličković et al., 2018)
- **DGL**: https://docs.dgl.ai/
- **PyTorch**: https://pytorch.org/docs/
- **DARPA TC**: https://github.com/darpa-i2o/Transparent-Computing

---

## Next Steps

1. ✓ Understand architecture (see ARCHITECTURE_DIAGRAMS.md)
2. ✓ Read implementation details (see IMPLEMENTATION_GUIDE.md)
3. → Run experiments with different mask rates
4. → Visualize attention patterns (see VISUALIZATION_GUIDE.md)
5. → Test on your own provenance data
6. → Optimize hyperparameters for your use case

---

**Created for: MAGIC Research Implementation**
**Last Updated: 2024**
