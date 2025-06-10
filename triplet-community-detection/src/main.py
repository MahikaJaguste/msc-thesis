import os

from io_utils import load_triplets_csv
from graph_utils import build_graph_from_triplets
from leiden_community import leiden_communities
from leiden_hierarchical_community import hierarchical_leiden_communities
from louvain_community import louvain_communities
from asyn_lpa_community import asyn_lpa_communities
from hybrid_lpa_leiden_community import hybrid_lpa_leiden_communities
from evaluation import print_level_evaluations


def main():

    # Load triplets
    csv_path = os.path.join("..", "data", "kg_triplets_for_neo4j.csv")
    triplet_df = load_triplets_csv(csv_path)
    print(f"Loaded {len(triplet_df)} triplets.")

    G, node_to_id, id_to_node = build_graph_from_triplets(triplet_df, directed=False)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    mode = "leiden-hierarchical"
    # mode = "louvain"
    # mode = "leiden"
    # mode = "asyn-lpa"  
    # mode = "hybrid-lpa-leiden"

    if mode == "leiden-hierarchical":
        print("Running hierarchical Leiden community detection...")
        # Hierarchical Leiden community detection
        community_df, parent_df, levels = hierarchical_leiden_communities(G, max_cluster_size=10, random_state=42)
    elif mode == "louvain":
        print("Running Louvain community detection...")
        community_df, parent_df, levels = louvain_communities(G)
    elif mode == "leiden":
        print("Running Leiden community detection...")
        community_df, parent_df, levels = leiden_communities(G)
    elif mode == "asyn-lpa":
        print("Running Async Label Propagation community detection...")
        community_df, parent_df, levels = asyn_lpa_communities(G)
    elif mode == "hybrid-lpa-leiden":
        print("Running Hybrid LPA-Leiden community detection...")
        community_df, parent_df, levels = hybrid_lpa_leiden_communities(G, max_cluster_size=10, random_state=42)
        
    
    for level in levels:
        print(f"Level {level}: {(community_df[community_df['level'] == level]['community_id']).nunique()} communities")
    print()

    # Evaluation
    print_level_evaluations(G, community_df, levels)

if __name__ == "__main__":
    main()