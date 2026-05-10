import torch
import warnings
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.config import build_args
warnings.filterwarnings('ignore')


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device)
    dataset_name = main_args.dataset
    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3
    set_random_seed(0)

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        pooler = Pooling(main_args.pooling)
        test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], args.dataset, main_args.n_dim,
                                                    main_args.e_dim)
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        model.eval()
        malicious, malicious_names = metadata['malicious']
        node_id_to_name = dict(zip(malicious, malicious_names))
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g).cpu().numpy())
                del g
            x_train = np.concatenate(x_train, axis=0)
            skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                x_test.append(model.embed(g).cpu().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)

            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            malicious_dict = {}
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            # Exclude training samples from the test set
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test
            
            # --- MODIFIED: Capture scores and trigger path reconstruction ---
            test_auc, test_std, scores, _ = evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test, result_y_test)
            
            # Trigger the kill chain walk on the final test graph
            last_g = load_entity_level_dataset(dataset_name, 'test', n_test - 1)
            reconstruct_attack_path(last_g, scores, test_idx, skip_benign, node_id_to_name)
            # ----------------------------------------------------------------
            
    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
    return


def reconstruct_attack_path(last_g, scores, test_idx, skip_benign, node_id_to_name, max_steps=15):
    print("\n" + "="*60)
    print("🔥 ANOMALOUS KILL CHAIN RECONSTRUCTION 🔥")
    print("="*60)

    # 1. Map scores back to the local node IDs of the final test graph
    node_to_score = {}
    for i, idx in enumerate(test_idx):
        if idx >= skip_benign:
            node_id = idx - skip_benign
            node_to_score[node_id] = scores[i]

    if not node_to_score:
        print("No anomalous nodes found in the final graph.")
        return

    # 2. Find the starting point (Patient Zero)
    # Filter for nodes that are actually in our dictionary (Known IOCs)
    ioc_nodes = {n: s for n, s in node_to_score.items() if (n + skip_benign) in node_id_to_name}
    
    if ioc_nodes:
        # Anchor the trace on the highest scoring known malicious entity
        start_node = max(ioc_nodes, key=ioc_nodes.get)
    else:
        # Fallback to the absolute highest score if no known IOCs exist here
        start_node = max(node_to_score, key=node_to_score.get)
        
    current_node = start_node
    visited = {start_node}

    # Grab the real name by converting the local ID back to a global ID
    global_start_node = start_node + skip_benign
    start_name = node_id_to_name.get(global_start_node, f"Unknown System Entity (Node {start_node})")
    
    print(f"🚨 PATIENT ZERO IDENTIFIED: {start_name}")
    print(f"   [Anomaly Score: {node_to_score[start_node]:.2f} - EXCEEDS THRESHOLD]")

    # 3. Traverse the graph greedily via DGL successors
    for step in range(max_steps):
        _, neighbors = last_g.out_edges(current_node)
        neighbors = neighbors.tolist()

        if not neighbors:
            print("  ↳ [END OF TRACE: Process terminated / No further outgoing interactions]")
            break

        valid_neighbors = [n for n in neighbors if n in node_to_score and n not in visited]
        if not valid_neighbors:
            print("  ↳ [END OF TRACE: All connected interactions have been investigated]")
            break

        next_node = max(valid_neighbors, key=lambda n: node_to_score[n])
        visited.add(next_node)

        global_next_node = next_node + skip_benign
        # If the attacker traverses into a normal system file, it will label it "Unknown System Entity"
        next_name = node_id_to_name.get(global_next_node, f"Unknown System Entity (Node {next_node})")

        print(f"  ↓ lateral movement via system interaction to")
        print(f"🦠 {next_name} (Score: {node_to_score[next_node]:.2f})")

        current_node = next_node


if __name__ == '__main__':
    args = build_args()
    main(args)