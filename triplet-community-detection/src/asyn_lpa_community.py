import networkx as nx
import pandas as pd
import os

def asyn_lpa_communities(G, output_dir="../data/asyn_lpa"):
    """
    Run asynchronous label propagation community detection using networkx.
    Returns:
        community_df: DataFrame with columns ['node_id', 'community_id', 'level']
        parent_df: empty DataFrame (for compatibility)
        levels: list with a single level [0]
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    communities = list(nx.algorithms.community.asyn_lpa_communities(G))
    community_data = []
    for comm_id, nodes in enumerate(communities):
        for node in nodes:
            community_data.append({'node_id': node, 'community_id': comm_id, 'level': 0})
    community_df = pd.DataFrame(community_data)
    community_csv_path = os.path.join(output_dir, "community_mapping.csv")
    community_df.to_csv(community_csv_path, index=False)
    print(f"Async LPA community mapping saved to {community_csv_path}")
    parent_df = pd.DataFrame(columns=['community_id', 'parent_community_id', 'level'])
    parent_csv_path = os.path.join(output_dir, "parent_mapping.csv")
    parent_df.to_csv(parent_csv_path, index=False)
    levels = [0]
    return community_df, parent_df, levels