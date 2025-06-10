import networkx as nx
import pandas as pd
import os

def build_graph_from_triplets(triplet_df, directed=False):
    """
    Build a NetworkX graph from triplet DataFrame.
    Returns:
        G: networkx.Graph or networkx.DiGraph
        node_to_id: dict mapping node name to integer id
        id_to_node: dict mapping integer id to node name
    """
    node_mapping_path = os.path.join("..", "data", "node_mapping_with_id.csv")

    # if os.path.exists(node_mapping_path):
    #     print(f"Node mapping file already exists at {node_mapping_path}. Loading existing mapping.")
    #     node_mapping_df = pd.read_csv(node_mapping_path)
    #     node_to_id = dict(zip(node_mapping_df['node_name'], node_mapping_df['node_id']))
    #     id_to_node = dict(zip(node_mapping_df['node_id'], node_mapping_df['node_name']))
    #     print(f"Loaded {len(node_to_id)} nodes from existing mapping.")
    # else: 
    print("Building graph from triplets...")

    nodes = set(triplet_df['subject']).union(set(triplet_df['object']))
    print(f"Found {len(nodes)} unique nodes in triplets.")

    node_to_id = {node: idx for idx, node in enumerate(sorted(nodes))}
    id_to_node = {idx: node for node, idx in node_to_id.items()}

    # store node to id mapping into a CSV file
    node_mapping_data = [{'node_name': node, 'node_id': idx} for node, idx in node_to_id.items()]
    node_mapping_df = pd.DataFrame(node_mapping_data)
    
    node_mapping_df.to_csv(node_mapping_path, index=False)
    print(f"Node to ID mapping saved with {len(node_mapping_df)} entries.")

    print()
    G = nx.DiGraph() if directed else nx.Graph()
    for _, row in triplet_df.iterrows():
        s = node_to_id[row['subject']]
        o = node_to_id[row['object']]
        G.add_node(s, name=row['subject'], type=row['subject_type'])
        G.add_node(o, name=row['object'], type=row['object_type'])
        G.add_edge(s, o, predicate=row['predicate'])
    
    return G, node_to_id, id_to_node