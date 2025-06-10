import networkx as nx
import pandas as pd
import os
from graspologic.partition import hierarchical_leiden

def hybrid_lpa_leiden_communities(G, max_cluster_size=100, random_state=None, output_dir="../data/hybrid_lpa_leiden"):
    """
    Run LPA to get initial communities, then use as starting_communities for hierarchical_leiden.
    Returns:
        community_df: DataFrame with columns ['node_id', 'community_id', 'level']
        parent_df: DataFrame with columns ['community_id', 'parent_community_id', 'level']
        levels: list of levels
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Run LPA
    lpa_partition = list(nx.algorithms.community.asyn_lpa_communities(G))
    print(f"LPA found {len(lpa_partition)} initial communities.")

    for comm_id, nodes in enumerate(lpa_partition):
        print(f"Community {comm_id}: Nodes: {list(nodes)[:5]}... (total {len(nodes)} nodes)")

    # Step 2: Convert to dict[node_id] = community_id
    starting_communities = {}
    for comm_id, nodes in enumerate(lpa_partition):
        for node in nodes:
            starting_communities[node] = comm_id

    # Step 3: Run hierarchical_leiden with starting_communities
    partitions = hierarchical_leiden(
        G,
        max_cluster_size=max_cluster_size,
        random_seed=random_state,
        starting_communities=starting_communities
    )

    # Step 4: Format results as DataFrames
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

    community_data = []
    parent_data = []
    for level in levels:
        for node, comm_id in node_to_community[level].items():
            community_data.append({'node_id': node, 'community_id': comm_id, 'level': level})
        for comm_id, parent_comm_id in parent_mapping[level].items():
            parent_data.append({'community_id': comm_id, 'parent_community_id': parent_comm_id, 'level': level})

    community_df = pd.DataFrame(community_data)
    community_csv_path = os.path.join(output_dir, "community_mapping.csv")
    community_df.to_csv(community_csv_path, index=False)
    print(f"Hybrid LPA-Leiden community mapping saved to {community_csv_path}")

    parent_df = pd.DataFrame(parent_data)
    parent_csv_path = os.path.join(output_dir, "parent_mapping.csv")
    parent_df.to_csv(parent_csv_path, index=False)
    print(f"Hybrid LPA-Leiden parent mapping saved to {parent_csv_path}")

    print(f"Hybrid LPA-Leiden detected {len(levels)} levels of communities.")
    print(f"Total communities across all levels: {sum(len(set(comm.values())) for comm in node_to_community.values())} \n")

    return community_df, parent_df, levels