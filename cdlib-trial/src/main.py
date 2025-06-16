import os
import pandas as pd
from community_detection.io_utils import load_triplets, save_community_result, save_metrics, print_community_summary
from community_detection.graph_utils import build_graph_from_triplets, remap_graph_for_algorithm, remap_node_communities
from community_detection.detection import get_algorithms, run_grid_search
from community_detection.evaluation import evaluate_internal, evaluate_external

def main():
    triplet_file = "kg_triplets_for_neo4j.csv"  # Replace as needed
    triplet_key = triplet_file.split("_for_")[1].split(".")[0]  # e.g., 'pub'

    patient_profiles_file = "../../data/thesis/cll_broad_2022_clinical_data_thesis.csv"
    df = pd.read_csv(patient_profiles_file)

    triplet_path = f"../data/{triplet_key}/{triplet_file}"
    triplets = load_triplets(triplet_path)

    G = build_graph_from_triplets(triplets)
    G_int, mapping, rev_mapping = remap_graph_for_algorithm(G)

    algorithms_dict = get_algorithms()

    for name, (algo_fn, param_grid) in algorithms_dict.items():
        print(f"Running: {name}")
       
        community = run_grid_search(G_int, algo_fn, param_grid)
        comm_mapped = remap_node_communities(community, mapping)

        print_community_summary(comm_mapped)

        # Save communities
        comm_path = f"../data/{triplet_key}/{name}/communities.csv"
        # save_community_result(comm_mapped, comm_path)

        comm_path = f"../data/{triplet_key}/{name}/communities_int.csv"
        # save_community_result(community, comm_path)
       
        # Internal Evaluation
        internal_metrics = evaluate_internal(G_int, community)
        internal_path = f"../data/{triplet_key}/{name}/internal_metrics.csv"
        save_metrics(internal_metrics, internal_path)

        if community.overlap:
            print("Overlap detected, skipping external evaluation.")
            continue
        # External Evaluation
        external_metrics = evaluate_external(community, mapping, df, ['CLL_EPITYPE', 'TUMOR_MOLECULAR_SUBTYPE'])
        external_path = f"../data/{triplet_key}/{name}/external_metrics.csv"
        save_metrics(external_metrics, external_path)

if __name__ == "__main__":
    main()
