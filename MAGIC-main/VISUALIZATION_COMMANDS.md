# MAGIC: Commands to Generate Visualizations

## Quick Answer
**Quick evaluation (`python eval.py`) does NOT automatically show visualizations** - it only prints AUC scores in the terminal.

To get **graphical visualizations**, use the dedicated visualization script.

---

## Commands to Run Visualizations

### 1. **Generate All Visualizations (Recommended)**

```bash
# For StreamSpot dataset
python visualize_graphs.py --dataset streamspot

# For Wget dataset
python visualize_graphs.py --dataset wget

# For DARPA TC datasets
python visualize_graphs.py --dataset trace
python visualize_graphs.py --dataset theia
python visualize_graphs.py --dataset cadets
```

**Output**: Saves 4-5 PNG images in `./figs/visualizations/`

---

### 2. **Generate Visualizations and Display Them**

```bash
# Display plots in interactive window
python visualize_graphs.py --dataset streamspot --show

# Works for all datasets
python visualize_graphs.py --dataset trace --show
```

---

### 3. **Save to Custom Directory**

```bash
# Save visualizations to specific folder
python visualize_graphs.py --dataset streamspot --output_dir ./my_viz

# Creates ./my_viz/ with all plots
```

---

### 4. **Use GPU for Faster Processing**

```bash
# Use GPU device 0
python visualize_graphs.py --dataset trace --device 0

# For batch-level datasets (faster processing)
python visualize_graphs.py --dataset streamspot --device 0
```

---

## What Visualizations You Get

Each dataset generates these plots automatically:

### **1. Graph Structure** 
- File: `{dataset}_graph_structure.png`
- Shows: Nodes as circles, edges as connections
- Use: Understand provenance graph topology

### **2. Attention Weights Heatmap**
- File: `{dataset}_attention_weights.png`
- Shows: Which node pairs the model focuses on
- Use: Interpret important edges in graph
- Red = high attention, Yellow = medium, Light = low attention

### **3. Reconstruction Errors** *(Batch-level only)*
- File: `{dataset}_reconstruction_errors.png`
- Shows: Distribution of reconstruction errors
- Use: See separation between normal and anomalous graphs

### **4. Embeddings t-SNE**
- File: `{dataset}_embeddings_tsne.png`
- Shows: 2D visualization of learned node embeddings
- Use: Verify if embeddings separate normal vs anomalous
- Different colors = different clusters

### **5. Layer Statistics**
- File: `{dataset}_layer_statistics.png`
- Shows: 4 subplots:
  - Mean activation per layer
  - Std dev per layer
  - Min/max values per layer
  - Variance per layer
- Use: Debug if layers are learning properly

---

## Expected Output Folder Structure

```
./figs/visualizations/
├── streamspot_graph_structure.png
├── streamspot_attention_weights.png
├── streamspot_reconstruction_errors.png
├── streamspot_embeddings_tsne.png
└── streamspot_layer_statistics.png
```

---

## Step-by-Step Workflow

### Step 1: Train or Load Model
```bash
# Already trained checkpoints exist in ./checkpoints/
# Just run visualization directly
```

### Step 2: Generate Visualizations
```bash
python visualize_graphs.py --dataset streamspot
# Wait 30-60 seconds...
```

### Step 3: View Results
```bash
# Option A: Open PNG files in image viewer
# ./figs/visualizations/streamspot_*.png

# Option B: View in Jupyter (if running notebook)
# from IPython.display import Image
# Image('./figs/visualizations/streamspot_graph_structure.png')
```

---

## Full Examples

### Example 1: Quick Visualization (No GPU)
```bash
cd C:\Users\asit3\OneDrive\Desktop\MAGIC-main\MAGIC-main
python visualize_graphs.py --dataset streamspot
# Output: 5 PNG files in ./figs/visualizations/
```

### Example 2: Fast Visualization (With GPU)
```bash
python visualize_graphs.py --dataset trace --device 0
# ~2x faster than CPU
```

### Example 3: View All Datasets
```bash
python visualize_graphs.py --dataset streamspot
python visualize_graphs.py --dataset wget
python visualize_graphs.py --dataset trace
python visualize_graphs.py --dataset theia
python visualize_graphs.py --dataset cadets
# Creates 25 total visualization files
```

### Example 4: Generate and Display
```bash
python visualize_graphs.py --dataset streamspot --show
# Opens interactive matplotlib window with plots
```

---

## Comparison: Eval vs Visualization

| Command | Output | Visualizations | Time |
|---------|--------|-----------------|------|
| `python eval.py --dataset trace` | Text: AUC ± std | ❌ None | 1-2 min |
| `python visualize_graphs.py --dataset trace` | 5 PNG files | ✅ Full | 2-3 min |
| Both combined | Text + images | ✅ Full | 3-5 min |

**Recommendation**: Run visualization AFTER eval to get both metrics and visualizations

---

## Interpretation Guide

### Graph Structure Visualization
- Larger graph = more complex system behavior
- Dense connections = many interactions
- Sparse connections = isolated entities
- **Anomaly insight**: APTs often show unusual connection patterns

### Attention Weights Heatmap
```
Red zones = Important edges that model learned
Yellow zones = Medium importance
Light zones = Minor importance

For anomaly detection:
- Attackers often use unusual edge combinations
- Attention weights should highlight suspicious patterns
```

### Reconstruction Errors
```
Low error = Learned pattern matches original
High error = Deviated from learned benign behavior
If high error → Mark as anomalous

Ideal: Bimodal distribution
- Left peak: Normal graphs (low error)
- Right peak: Anomalous graphs (high error)
```

### t-SNE Embeddings
```
Well-separated clusters = Model learned good representations
Overlapping clusters = Model may need tuning
Outliers = Single nodes/graphs very different from others

Colors show:
- Blue: Normal nodes
- Red: Anomalous nodes
- Closer = more similar nodes
```

### Layer Statistics
```
If Mean & Std increasing across layers:
  → Model capturing more complex patterns ✓
  
If Mean/Std staying flat:
  → Check if model is learning properly

Healthy pattern:
  L0: ~mean 0.5, std 0.2
  L1: ~mean 0.6, std 0.3
  L2: ~mean 0.7, std 0.35
```

---

## Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Install missing visualization dependencies
pip install matplotlib seaborn scikit-learn networkx
```

### Issue: CUDA out of memory
```bash
# Use CPU instead of GPU
python visualize_graphs.py --dataset trace --device -1

# Or reduce sample size by editing the script
```

### Issue: Takes too long
```bash
# For DARPA TC, reduce graphs loaded
# Edit visualize_graphs.py, line ~180:
# Change: for i in range(min(n_train, 30))
# To: for i in range(min(n_train, 10))
```

### Issue: PNG files very large/blurry
```bash
# Adjust DPI in visualize_graphs.py
# Change: fig.savefig(..., dpi=150, ...)
# To: fig.savefig(..., dpi=300, ...)  # Higher quality
```

---

## Batch Processing All Datasets

Create a `run_all_visualizations.sh`:

```bash
#!/bin/bash
echo "Generating visualizations for all datasets..."

python visualize_graphs.py --dataset streamspot --output_dir ./figs/viz_streamspot
python visualize_graphs.py --dataset wget --output_dir ./figs/viz_wget
python visualize_graphs.py --dataset trace --output_dir ./figs/viz_trace
python visualize_graphs.py --dataset theia --output_dir ./figs/viz_theia
python visualize_graphs.py --dataset cadets --output_dir ./figs/viz_cadets

echo "All visualizations complete!"
```

Run with:
```bash
bash run_all_visualizations.sh
```

---

## Python Script Usage (In Jupyter/Notebook)

```python
import torch
from visualize_graphs import (
    setup_model, 
    visualize_single_graph,
    visualize_attention_weights,
    visualize_embeddings_tsne
)
from utils.loaddata import load_batch_level_dataset

# Load model and data
model, config = setup_model('streamspot', device='cpu')
dataset = load_batch_level_dataset('streamspot')
graphs = dataset['dataset']

# Generate individual plots
fig1 = visualize_single_graph(graphs[0])
fig2 = visualize_attention_weights(model, graphs[0])
fig3 = visualize_embeddings_tsne(model, graphs[:20])

# Display
import matplotlib.pyplot as plt
plt.show()
```

---

## Node-Level Visualizations During Evaluation

Currently, **quick evaluation does NOT show node-level visualizations by default**.

However, you get:
- ✅ **Per-node reconstruction errors** (stored internally)
- ✅ **Per-node embeddings** (via visualization script)
- ✅ **Per-node anomaly scores** (computed but not visualized)

To enable node-level viz during eval, you can modify `eval.py` (optional):

```python
# Add after getting predictions in eval.py:
if visualize:
    # Plot per-node errors
    import matplotlib.pyplot as plt
    plt.hist(node_errors, bins=50)
    plt.title('Per-Node Reconstruction Errors')
    plt.savefig('node_errors.png')
    plt.show()
```

---

## Summary

| What You Want | Command |
|---------------|---------|
| Just AUC score | `python eval.py --dataset trace` |
| All visualizations (PNG) | `python visualize_graphs.py --dataset trace` |
| Visualizations + interactive | `python visualize_graphs.py --dataset trace --show` |
| GPU-accelerated viz | `python visualize_graphs.py --dataset trace --device 0` |
| Node embeddings | Run visualization → `_embeddings_tsne.png` |
| Attention patterns | Run visualization → `_attention_weights.png` |
| Graph topology | Run visualization → `_graph_structure.png` |

**Tip**: After each evaluation run, immediately run visualization to see what the model learned!
