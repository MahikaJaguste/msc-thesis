import os
import pandas as pd
import networkx as nx
import random
from collections import defaultdict
from cdlib import evaluation
from cdlib import NodeClustering
from utils.w_slpa import weighted_slpa
from utils.overlapping_summary import print_overlapping_node_summary

# Paths
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
output_dir = os.path.join(DATA_PATH, "w_slpa")

os.makedirs(output_dir, exist_ok=True)

# Load graph and mapping
G = nx.read_gml(graph_path, label='id')
mapping_df = pd.read_csv(mapping_path)
mapping_df['nodeId'] = mapping_df['nodeId'].astype(int)
nodeid_to_patientid = dict(zip(mapping_df["nodeId"], mapping_df["patientId"]))

if "community_assignments.csv" in os.listdir(output_dir):
    community_df = pd.read_csv(os.path.join(output_dir, "community_assignments.csv"))
    communities = []
    for community_id in community_df['communityId'].unique():
        members = community_df[community_df['communityId'] == community_id]['nodeId'].tolist()
        communities.append(members)
else:
    # Normalize edge weights to [0, 1]
    weights = [d['weight'] for u, v, d in G.edges(data=True)]
    min_w, max_w = min(weights), max(weights)
    for u, v, d in G.edges(data=True):
        d['weight'] = (d['weight'] - min_w) / (max_w - min_w + 1e-9)

    # Run weighted SLPA
    communities = weighted_slpa(G, freq_dist_path=os.path.join(output_dir, "community_size_distribution.png"), plot_freq_dist=True)

    # Save community assignments
    community_data = []
    for idx, community in enumerate(communities):
        for node in community:
            community_data.append({
                "nodeId": node,
                "patientId": nodeid_to_patientid[node],
                "communityId": idx
            })
    community_df = pd.DataFrame(community_data)
    community_df.to_csv(os.path.join(output_dir, "community_assignments.csv"), index=False)

clustering = NodeClustering(communities=communities, graph=G, method_name="weighted_slpa", overlap=True)

# Compute metrics
overlapping_mod = evaluation.modularity_overlap(G, clustering).score
conductance = evaluation.conductance(G, clustering).score
modularity = evaluation.newman_girvan_modularity(G, clustering).score

# Summary
num_communities = len(communities)
num_nodes = len(set([node for comm in communities for node in comm]))
avg_size = sum(len(comm) for comm in communities) / num_communities
min_size = min(len(comm) for comm in communities)
max_size = max(len(comm) for comm in communities)

metrics_df = pd.DataFrame([{
    "modularity": modularity,
    "conductance": conductance,
    'overlapping_modularity': overlapping_mod,
    'num_communities': num_communities,
    'num_nodes_in_communities': num_nodes,
    'avg_community_size': avg_size,
    'min_community_size': min_size,
    'max_community_size': max_size
}])
metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

print(f"Weighted SLPA Community Detection Summary:")
print(f"Number of communities: {num_communities}")
print(f"Number of nodes in communities: {num_nodes}")
print(f"Average community size: {avg_size:.2f}")
print(f"Min community size: {min_size}")
print(f"Max community size: {max_size}")
print(f"Modularity: {modularity:.4f}")
print(f"Overlapping Modularity: {overlapping_mod:.4f}")
print(f"Overlapping Conductance: {conductance:.4f}")

print_overlapping_node_summary(communities, G, nodeid_to_patientid)