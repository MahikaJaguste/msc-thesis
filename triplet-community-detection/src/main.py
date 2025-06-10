import os

from io_utils import load_triplets_csv
from graph_utils import build_graph_from_triplets
from leiden_community import hierarchical_leiden_communities
from evaluation import leiden_modularity, print_community_stats

def main():

    # Load triplets
    csv_path = os.path.join("..", "data", "kg_triplets_for_neo4j.csv")
    triplet_df = load_triplets_csv(csv_path)
    print(f"Loaded {len(triplet_df)} triplets.")

    G, node_to_id, id_to_node = build_graph_from_triplets(triplet_df, directed=False)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Hierarchical Leiden community detection
    community_df, parent_df, levels = hierarchical_leiden_communities(G, max_cluster_size=100, random_state=42)
    for level in levels:
        print(f"Level {level}: {(community_df[community_df['level'] == level]['community_id']).nunique()} communities")
    print()

    # # Evaluation
    # mod = leiden_modularity(G, leiden_result)
    # print(f"Leiden modularity (finest level): {mod:.4f}")
    # print_community_stats(leiden_result, id_to_node)

if __name__ == "__main__":
    main()