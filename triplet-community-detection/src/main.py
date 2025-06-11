import os

from io_utils import load_triplets_csv
from graph_utils import build_graph_from_triplets
from leiden_community import leiden_communities
from leiden_hierarchical_community import hierarchical_leiden_communities
from louvain_community import louvain_communities
from asyn_lpa_community import asyn_lpa_communities
from hybrid_lpa_leiden_community import hybrid_lpa_leiden_communities
from community_assignment_utils import make_df_with_community
from evaluation import print_and_store_level_evaluations


def main():

    # Load triplets
    triplet_path = os.path.join("..", "data", "kg_triplets_for_neo4j.csv")
    triplet_df = load_triplets_csv(triplet_path)
    print(f"Loaded {len(triplet_df)} triplets.")

    triplet_key = os.path.basename(triplet_path).replace("kg_triplets_for_", "").replace(".csv", "")

    G, node_to_id, id_to_node = build_graph_from_triplets(triplet_df, triplet_key, directed=False)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # mode = "leiden-hierarchical"
    # mode = "louvain"
    # mode = "leiden"
    # mode = "asyn_lpa"  
    mode = "hybrid_lpa_leiden"

    if mode == "leiden-hierarchical":
        print("Running hierarchical Leiden community detection...")
        # Hierarchical Leiden community detection
        community_df, parent_df, levels = hierarchical_leiden_communities(G, triplet_key, max_cluster_size=10, random_state=42)
    elif mode == "louvain":
        print("Running Louvain community detection...")
        community_df, parent_df, levels = louvain_communities(G, triplet_key)
    elif mode == "leiden":
        print("Running Leiden community detection...")
        community_df, parent_df, levels = leiden_communities(G, triplet_key)
    elif mode == "asyn_lpa":
        print("Running Async Label Propagation community detection...")
        community_df, parent_df, levels = asyn_lpa_communities(G, triplet_key)
    elif mode == "hybrid_lpa_leiden":
        print("Running Hybrid LPA-Leiden community detection...")
        community_df, parent_df, levels = hybrid_lpa_leiden_communities(G, triplet_key, max_cluster_size=10, random_state=42)
        
    
    for level in levels:
        print(f"Level {level}: {(community_df[community_df['level'] == level]['community_id']).nunique()} communities")
    print()


    # After community detection
    clinical_path = "../../data/thesis/cll_broad_2022_clinical_data_thesis.csv"
    node_mapping_path = "../data/node_mapping_with_id.csv"
    df_with_community = make_df_with_community(mode, triplet_key, clinical_path, node_mapping_path)

    # Evaluation
    label_cols = ["CLL_EPITYPE", "TUMOR_MOLECULAR_SUBTYPE"]
    print_and_store_level_evaluations(
        G, community_df, levels, df_with_community, label_cols, mode, triplet_key, out_base=".."
    )

    
if __name__ == "__main__":
    main()