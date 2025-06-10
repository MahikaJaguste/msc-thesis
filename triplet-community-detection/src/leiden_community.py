from graspologic.partition import hierarchical_leiden
import pandas as pd
import os

def hierarchical_leiden_communities(G, max_cluster_size=100, random_state=None):
    """
    Run hierarchical Leiden community detection on a graph.
    Returns:
        - node_to_community: dict[level][node] = community_id
        - parent_mapping: dict[level][community_id] = parent_community_id
        - levels: sorted list of levels
    """
    partitions = hierarchical_leiden(G, max_cluster_size=max_cluster_size, random_seed=random_state)
    node_to_community = {}
    parent_mapping = {}
    for part in partitions:
        level = part.level
        if level not in node_to_community:
            node_to_community[level] = {}
            parent_mapping[level] = {}
        node_to_community[level][part.node] = part.cluster
        parent_mapping[level][part.cluster] = part.parent_cluster if part.parent_cluster is not None else -1
    levels = sorted(node_to_community.keys())

    # store node to community mapping into a CSV file, store all levels
    # also store parent mapping in a separate CSV file
    community_data = []
    parent_data = []
    for level in levels:
        for node, comm_id in node_to_community[level].items():
            community_data.append({'node_id': node, 'community_id': comm_id, 'level': level})
        for comm_id, parent_comm_id in parent_mapping[level].items():
            parent_data.append({'community_id': comm_id, 'parent_community_id': parent_comm_id, 'level': level})

    community_df = pd.DataFrame(community_data)
    community_csv_path = os.path.join("..", "data", "community_mapping.csv")
    community_df.to_csv(community_csv_path, index=False)
    print(f"Community mapping saved to {community_csv_path}")
    parent_df = pd.DataFrame(parent_data)
    parent_csv_path = os.path.join("..", "data", "parent_mapping.csv")
    parent_df.to_csv(parent_csv_path, index=False)
    print(f"Parent mapping saved to {parent_csv_path}")

    print(f"Hierarchical Leiden detected {len(levels)} levels of communities.")
    print(f"Total communities across all levels: {sum(len(set(comm.values())) for comm in node_to_community.values())} \n")

    return community_df, parent_df, levels
