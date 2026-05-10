"""
MAGIC: Visualization Scripts
Generate graph visualizations, embeddings, and analysis plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import dgl
import networkx as nx
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.config import build_args
import warnings
warnings.filterwarnings('ignore')


def get_node_features(graph):
    """Dynamically resolve the node feature key from available ndata keys."""
    for key in ['attr', 'feat', 'feature', 'x', 'type']:
        if key in graph.ndata:
            return graph.ndata[key].float()
    raise KeyError(f"No recognized node feature key found. Available keys: {list(graph.ndata.keys())}")


def prepare_graph(graph):
    """
    Ensure graph.ndata and graph.edata both have an 'attr' key,
    as expected by model/gat.py.
    """
    print(f"   ndata keys: {list(graph.ndata.keys())}")
    print(f"   edata keys: {list(graph.edata.keys())}")

    # --- Fix node features ---
    if 'attr' not in graph.ndata:
        for key in ['feat', 'feature', 'x', 'type']:
            if key in graph.ndata:
                graph.ndata['attr'] = graph.ndata[key].float()
                print(f"   Mapped ndata['{key}'] -> ndata['attr']")
                break
        else:
            # Absolute fallback: use the first available ndata tensor
            if graph.ndata:
                first_key = list(graph.ndata.keys())[0]
                graph.ndata['attr'] = graph.ndata[first_key].float()
                print(f"   Fallback: mapped ndata['{first_key}'] -> ndata['attr']")

    # --- Fix edge features ---
    if 'attr' not in graph.edata:
        n_edges = graph.num_edges()
        mapped = False

        # Try known keys first
        for key in ['feat', 'feature', 'type', 'w', 'weight', 'etype', 'efeat']:
            if key in graph.edata:
                graph.edata['attr'] = graph.edata[key].float()
                print(f"   Mapped edata['{key}'] -> edata['attr']")
                mapped = True
                break

        # Use first available edata tensor
        if not mapped and graph.edata:
            first_key = list(graph.edata.keys())[0]
            graph.edata['attr'] = graph.edata[first_key].float()
            print(f"   Fallback: mapped edata['{first_key}'] -> edata['attr']")
            mapped = True

        # Nothing at all — create zeros with correct dim (20 for streamspot)
        if not mapped:
            e_dim = 20
            graph.edata['attr'] = torch.zeros(n_edges, e_dim)
            print(f"   No edge features found — created zeros edata['attr'] ({n_edges} x {e_dim})")

    return graph


def setup_model(dataset_name, device='cpu'):
    """Load trained model"""
    args = build_args()
    args.dataset = dataset_name
    
    if dataset_name in ['streamspot', 'wget']:
        args.num_hidden = 256
        args.num_layers = 4
    else:
        args.num_hidden = 64
        args.num_layers = 3
    
    dataset = load_batch_level_dataset(dataset_name) if dataset_name in ['streamspot', 'wget'] \
              else load_metadata(dataset_name)
    
    if dataset_name in ['streamspot', 'wget']:
        args.n_dim = dataset['n_feat']
        args.e_dim = dataset['e_feat']
    else:
        args.n_dim = dataset['node_feature_dim']
        args.e_dim = dataset['edge_feature_dim']
    
    model = build_model(args)
    model.load_state_dict(torch.load(f"./checkpoints/checkpoint-{dataset_name}.pt", 
                                     map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, args


def visualize_single_graph(graph, title="Provenance Graph"):
    """Visualize a single graph structure (optimized for large graphs)"""
    g_nx = dgl.to_networkx(graph).to_undirected()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    num_nodes = g_nx.number_of_nodes()
    num_edges = g_nx.number_of_edges()
    print(f"   Graph has {num_nodes} nodes, {num_edges} edges")
    
    # For extremely large graphs, just show statistics instead of layout
    if num_nodes > 50000:
        print(f"   Graph too large for layout visualization, showing statistics instead...")
        ax.axis('off')
        
        degrees = [d for n, d in g_nx.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = np.max(degrees) if degrees else 0
        
        stats_text = f"""
Graph Statistics for {title}

Nodes: {num_nodes:,}
Edges: {num_edges:,}
Average Degree: {avg_degree:.2f}
Max Degree: {max_degree}
Density: {2*num_edges/(num_nodes*(num_nodes-1)):.6f}

(Graph too large for layout visualization)
        """
        
        ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    if num_nodes > 5000:
        pos = nx.random_layout(g_nx, seed=42)
    elif num_nodes > 1000:
        pos = nx.spectral_layout(g_nx)
    elif num_nodes > 300:
        pos = nx.kamada_kawai_layout(g_nx)
    else:
        pos = nx.spring_layout(g_nx, k=2, iterations=20, seed=42)
    
    print(f"   Drawing {num_nodes} nodes and {num_edges} edges...")
    nx.draw_networkx_nodes(g_nx, pos, node_color='lightblue', 
                          node_size=30, ax=ax, alpha=0.6)
    nx.draw_networkx_edges(g_nx, pos, alpha=0.1, ax=ax, width=0.2)
    
    if num_nodes <= 50:
        nx.draw_networkx_labels(g_nx, pos, font_size=5, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig



def visualize_reconstruction_errors(model, graphs, labels=None, num_graphs=15):
    """Visualize reconstruction errors (optimized)"""
    model.eval()
    errors = []
    
    print(f"Computing reconstruction errors for {len(graphs[:num_graphs])} graphs...")
    with torch.no_grad():
        for i, g in enumerate(graphs[:num_graphs]):
            g = prepare_graph(g)
            x = get_node_features(g)
            enc_rep, all_hidden = model.encoder(g, x, return_hidden=True)
            enc_rep = torch.cat(all_hidden, dim=1)
            rep = model.encoder_to_decoder(enc_rep)
            x_recon = model.decoder(g, rep)
            
            error = torch.norm(x_recon - x, dim=1).cpu().numpy()
            errors.append(np.mean(error))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if labels is not None:
        labels_subset = labels[:num_graphs]
        normal_errors = [errors[i] for i in range(len(errors)) if labels_subset[i] == 0]
        anomaly_errors = [errors[i] for i in range(len(errors)) if labels_subset[i] == 1]
        
        ax.hist(normal_errors, bins=10, alpha=0.6, label='Normal', color='blue')
        ax.hist(anomaly_errors, bins=10, alpha=0.6, label='Anomalous', color='red')
    else:
        ax.hist(errors, bins=10, alpha=0.7, color='green')
    
    ax.set_xlabel('Mean Reconstruction Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Reconstruction Error Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def visualize_embeddings_tsne(model, graphs, labels=None, sample_size=300):
    """Visualize learned embeddings using t-SNE (ultra-fast optimized)"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    count = 0
    print("Extracting embeddings (ultra-fast mode)...")
    with torch.no_grad():
        for i, g in enumerate(graphs):
            g = prepare_graph(g)
            x = get_node_features(g)
            embeddings = model.encoder(g, x, return_hidden=False)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if labels is not None:
                all_labels.extend([labels[i]] * len(embeddings))
            
            count += len(embeddings)
            if count >= sample_size:
                break
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)[:sample_size]
    
    print(f"Running t-SNE on {len(all_embeddings)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(all_embeddings)//10), 
                init='pca', max_iter=300, learning_rate=200, verbose=0)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if labels is not None and len(all_labels) > 0:
        all_labels = np.array(all_labels)[:sample_size]
        for label in np.unique(all_labels):
            mask = all_labels == label
            label_name = 'Normal' if label == 0 else 'Anomalous'
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      label=label_name, alpha=0.6, s=20)
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title(f'Node Embeddings Visualization (t-SNE)\n{len(all_embeddings)} nodes', 
                fontsize=13, fontweight='bold')
    if labels is not None:
        ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


def visualize_layer_statistics(model, graph):
    """Show activation statistics through layers"""
    model.eval()
    graph = prepare_graph(graph)
    x = get_node_features(graph)

    with torch.no_grad():
        _, hidden_list = model.encoder(graph, x, return_hidden=True)
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(stats['layer'], stats['mean'], marker='o', linewidth=2, markersize=8, color='blue')
    ax.set_ylabel('Mean Activation', fontsize=11)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_title('Mean Activation per Layer', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(stats['layer'], stats['std'], marker='o', linewidth=2, markersize=8, color='orange')
    ax.set_ylabel('Std Dev', fontsize=11)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_title('Activation Std Dev per Layer', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(stats['layer'], stats['min'], marker='o', linewidth=2, markersize=8, color='red', label='Min')
    ax.plot(stats['layer'], stats['max'], marker='s', linewidth=2, markersize=8, color='green', label='Max')
    ax.set_ylabel('Value', fontsize=11)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_title('Min/Max Activations per Layer', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    variance = [s**2 for s in stats['std']]
    ax.bar(stats['layer'], variance, color='purple', alpha=0.7)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_title('Variance per Layer', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize MAGIC model")
    parser.add_argument("--dataset", type=str, default="streamspot", 
                       choices=['streamspot', 'wget', 'trace', 'theia', 'cadets'])
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="./figs/visualizations")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    
    args = parser.parse_args()
    device = torch.device('cuda' if args.device >= 0 else 'cpu')
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading {args.dataset} model...")
    model, config = setup_model(args.dataset, device)
    
    print(f"Loading {args.dataset} data...")
    if args.dataset in ['streamspot', 'wget']:
        dataset = load_batch_level_dataset(args.dataset)

        raw_graphs = dataset['dataset']
        graphs = []

        for item in raw_graphs:
            if isinstance(item, tuple):
                graphs.append(item[0])
            else:
                graphs.append(item)

        labels = None
        
        print("\n1. Visualizing graph structure...")
        fig = visualize_single_graph(graphs[0], f"{args.dataset.upper()} - Graph Structure")
        fig.savefig(f"{args.output_dir}/{args.dataset}_graph_structure.png", dpi=150, bbox_inches='tight')
        print(f"   Saved: {args.output_dir}/{args.dataset}_graph_structure.png")
        
        print("2. Visualizing reconstruction errors...")
        fig = visualize_reconstruction_errors(model, graphs[:10])
        fig.savefig(f"{args.output_dir}/{args.dataset}_reconstruction_errors.png", dpi=100, bbox_inches='tight')
        print(f"   Saved: {args.output_dir}/{args.dataset}_reconstruction_errors.png")
        
        print("3. Visualizing embeddings (t-SNE - ultra-fast mode)...")
        fig = visualize_embeddings_tsne(model, graphs[:8], sample_size=250)
        fig.savefig(f"{args.output_dir}/{args.dataset}_embeddings_tsne.png", dpi=100, bbox_inches='tight')
        print(f"   Saved: {args.output_dir}/{args.dataset}_embeddings_tsne.png")
        
        print("4. Visualizing layer statistics...")
        fig = visualize_layer_statistics(model, graphs[0])
        fig.savefig(f"{args.output_dir}/{args.dataset}_layer_statistics.png", dpi=150, bbox_inches='tight')
        print(f"   Saved: {args.output_dir}/{args.dataset}_layer_statistics.png")
        
    else:  # Entity-level datasets (TRACE, THEIA, CADETS)
        metadata = load_metadata(args.dataset)
        n_train = metadata['n_train']
        
        num_graphs_to_load = min(n_train, 3)
        print(f"Loading {num_graphs_to_load} training graphs (out of {n_train})...")
        graphs = []
        for i in range(num_graphs_to_load):
            print(f"  Loading graph {i+1}/{num_graphs_to_load}...")
            g = load_entity_level_dataset(args.dataset, 'train', i)
            graphs.append(g)
        
        print("\n1. Visualizing graph structure...")
        fig = visualize_single_graph(graphs[0], f"{args.dataset.upper()} - Entity Graph")
        fig.savefig(f"{args.output_dir}/{args.dataset}_graph_structure.png", dpi=100, bbox_inches='tight')
        print(f"   Saved: {args.output_dir}/{args.dataset}_graph_structure.png")
        
        print("2. Skipping attention weights (graph too large)...")
        
        ##print("3. Visualizing layer statistics...")
        ##fig = visualize_layer_statistics(model, graphs[0])
        ##fig.savefig(f"{args.output_dir}/{args.dataset}_layer_statistics.png", dpi=100, bbox_inches='tight')
        ##print(f"   Saved: {args.output_dir}/{args.dataset}_layer_statistics.png")
        
        print("3. Visualizing embeddings (t-SNE - ultra-fast mode)...")
        fig = visualize_embeddings_tsne(model, graphs, sample_size=400)
        fig.savefig(f"{args.output_dir}/{args.dataset}_embeddings_tsne.png", dpi=100, bbox_inches='tight')
        print(f"   Saved: {args.output_dir}/{args.dataset}_embeddings_tsne.png")
    
    print(f"\nAll visualizations saved to: {args.output_dir}")
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()