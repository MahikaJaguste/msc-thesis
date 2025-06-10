from cdlib import algorithms
import pandas as pd
import os

def louvain_communities(G):
    """
    Run Louvain community detection using cdlib.
    Returns:
        community_df: DataFrame with columns ['node_id', 'community_id', 'level']
        parent_df: empty DataFrame (for compatibility)
        levels: list with a single level [0]
    """
    output_dir="../data/louvain"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result = algorithms.louvain(G)
    # result.communities is a list of lists of node ids
    community_data = []
    for comm_id, nodes in enumerate(result.communities):
        for node in nodes:
            community_data.append({'node_id': node, 'community_id': comm_id, 'level': 0})
    community_df = pd.DataFrame(community_data)
    community_csv_path = os.path.join(output_dir, "community_mapping.csv")
    community_df.to_csv(community_csv_path, index=False)
    print(f"Louvain community mapping saved to {community_csv_path}")
    parent_df = pd.DataFrame(columns=['community_id', 'parent_community_id', 'level'])
    parent_csv_path = os.path.join(output_dir, "parent_mapping.csv")
    parent_df.to_csv(parent_csv_path, index=False)
    levels = [0]
    return community_df, parent_df, levels