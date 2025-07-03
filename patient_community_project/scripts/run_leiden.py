import os
import pandas as pd
import networkx as nx
from graspologic.partition import hierarchical_leiden
from cdlib import evaluation
from cdlib import NodeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict

# Define paths
GRAPH_NAME = "full"
PATIENT_DF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/thesis/cll_broad_2022_clinical_data_thesis.csv")
)
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../data/{GRAPH_NAME}")
)
graph_path = os.path.join(DATA_PATH, "graph.gml")
mapping_path = os.path.join(DATA_PATH, "patient_id_mapping.csv")
patient_df_path = PATIENT_DF_PATH
output_dir = os.path.join(DATA_PATH, "leiden")

os.makedirs(output_dir, exist_ok=True)

# Load graph
G = nx.read_gml(graph_path, label='id')

# Load mapping
mapping_df = pd.read_csv(mapping_path)
mapping_df['nodeId'] = mapping_df['nodeId'].astype(int)
node_to_patient = dict(zip(mapping_df['nodeId'], mapping_df['patientId']))
patient_to_node = dict(zip(mapping_df['patientId'], mapping_df['nodeId']))

# Load patient_df
patient_df = pd.read_csv(patient_df_path)

# Run hierarchical Leiden
partitions = hierarchical_leiden(G, max_cluster_size=100, random_seed=42)

# Collect assignments for each level
level_assignments = defaultdict(list)
for part in partitions:
    level_assignments[part.level].append((part.node, part.cluster))

# Store community assignments at each level
summary = []
for level, assignments in level_assignments.items():
    comm_df = pd.DataFrame(assignments, columns=["nodeId", "communityId"])
    comm_df["patientId"] = comm_df["nodeId"].map(node_to_patient)
    comm_df.to_csv(os.path.join(output_dir, f"level_{level}_community_assignments.csv"), index=False)

    # Prepare clustering object for cdlib
    communities = {}
    for node, comm in assignments:
        communities.setdefault(comm, []).append(node)
    clustering = NodeClustering(communities=list(communities.values()), graph=G, method_name="leiden")

    # Compute internal metrics
    mod = evaluation.newman_girvan_modularity(G, clustering).score
    cond = evaluation.conductance(G, clustering).score

        # Only use nodes present in comm_df for this level
    comm_nodes = set(comm_df["nodeId"])
    graph_nodes = [n for n in G.nodes() if int(n) in comm_nodes]

    # Compute external metrics (only for disjoint)
    labels_true = patient_df.set_index("patientId").loc[
        [node_to_patient[int(n)] for n in graph_nodes]
    ][["CLL_EPITYPE", "TUMOR_MOLECULAR_SUBTYPE"]]

    labels_pred = [comm_df.set_index("nodeId").loc[int(n), "communityId"] for n in graph_nodes]
    labels_true["communityId"] = labels_pred

    # Drop rows with any NaN in true labels or communityId
    labels_true = labels_true.dropna(subset=["CLL_EPITYPE", "TUMOR_MOLECULAR_SUBTYPE", "communityId"])
    filtered_labels_pred = labels_true["communityId"].tolist()

    nmi_epitype = normalized_mutual_info_score(labels_true["CLL_EPITYPE"], filtered_labels_pred)
    ari_epitype = adjusted_rand_score(labels_true["CLL_EPITYPE"], filtered_labels_pred)
    nmi_subtype = normalized_mutual_info_score(labels_true["TUMOR_MOLECULAR_SUBTYPE"], filtered_labels_pred)
    ari_subtype = adjusted_rand_score(labels_true["TUMOR_MOLECULAR_SUBTYPE"], filtered_labels_pred)
    
    num_nodes_in_communities = len(set(node for comm in communities.values() for node in comm))
    # Save metrics
    metrics = {
        "level": level,
        "num_communities": len(communities),
        "num_nodes_in_communities": num_nodes_in_communities,
        "avg_community_size": sum(len(c) for c in communities.values()) / len(communities),
        "min_community_size": min(len(c) for c in communities.values()),
        "max_community_size": max(len(c) for c in communities.values()),
        "modularity": mod,
        "conductance": cond,
        "nmi_epitype": nmi_epitype,
        "ari_epitype": ari_epitype,
        "nmi_subtype": nmi_subtype,
        "ari_subtype": ari_subtype
    }
    summary.append(metrics)

    print(f"Level {level}: {len(communities)} communities")
    print(f"  Avg size: {metrics['avg_community_size']:.2f}, Min: {metrics['min_community_size']}, Max: {metrics['max_community_size']}")
    print(f"  Modularity: {mod:.4f}, Conductance: {cond:.4f}")
    print(f"  NMI (CLL_EPITYPE): {nmi_epitype:.4f}, ARI: {ari_epitype:.4f}")
    print(f"  NMI (TUMOR_MOLECULAR_SUBTYPE): {nmi_subtype:.4f}, ARI: {ari_subtype:.4f}")

# Save summary metrics
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)