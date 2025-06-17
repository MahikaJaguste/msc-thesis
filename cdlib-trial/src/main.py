import pandas as pd
from community_detection.workflow_steps import (
    workflow_construct_graph,
    workflow_slpa,
    workflow_hierarchical_leiden,
    workflow_slpa_on_leiden_level
)

def main():
    # Example workflow: ["construct_graph", "slpa", "hierarchical_leiden", "slpa_on_leiden_level:0"]
    workflows = [
        "construct_graph", 
        "slpa", 
        "hierarchical_leiden", 
        "slpa_on_leiden_level:0"
    ]
    triplet_file = "kg_triplets_for_neo4j.csv"
    triplet_key = triplet_file.split("_for_")[1].split(".")[0]
    data_dir = "../data"
    triplet_path = f"{data_dir}/{triplet_key}/{triplet_file}"
    patient_profiles_file = "../../data/thesis/cll_broad_2022_clinical_data_thesis.csv"
    df = pd.read_csv(patient_profiles_file)
    label_columns = ['CLL_EPITYPE', 'TUMOR_MOLECULAR_SUBTYPE']

    G = G_int = mapping = None

    for step in workflows:
        if step == "construct_graph":
            G, G_int, mapping = workflow_construct_graph(triplet_path, triplet_key, data_dir)
        elif step == "slpa":
            if G_int is None:
                raise ValueError("G_int not constructed yet.")
            workflow_slpa(G_int, triplet_key, data_dir)
        elif step == "hierarchical_leiden":
            if G_int is None or mapping is None:
                raise ValueError("G_int or mapping not constructed yet.")
            workflow_hierarchical_leiden(G_int, triplet_key, data_dir, mapping, df, label_columns)
        elif step.startswith("slpa_on_leiden_level"):
            if G_int is None:
                raise ValueError("G_int not constructed yet.")
            level = int(step.split(":")[1])
            workflow_slpa_on_leiden_level(G_int, triplet_key, data_dir, level)
        else:
            print(f"Unknown workflow step: {step}")


if __name__ == "__main__":
    main()
