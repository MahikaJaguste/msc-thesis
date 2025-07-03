import os
import pandas as pd
import networkx as nx
from cdlib import algorithms, evaluation
from build_graph import DATA_PATH, PATIENT_DF_PATH

# Define paths
graph_path = os.path.join(DATA_PATH, "graph.gml")
mapping_path = os.path.join(DATA_PATH, "patient_id_mapping.csv")
patient_df_path = PATIENT_DF_PATH
output_dir = os.path.join(DATA_PATH, "slpa")

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)

# Load graph
G = nx.read_gml(graph_path, label='id')

# Load mapping
mapping_df = pd.read_csv(mapping_path)
mapping_df['nodeId'] = mapping_df['nodeId'].astype(int)
nodeid_to_patientid = dict(zip(mapping_df["nodeId"], mapping_df["patientId"]))

# Run SLPA
slpa_communities = algorithms.slpa(G, r=0.01, t=100)

# Save community assignments
community_data = []
for idx, community in enumerate(slpa_communities.communities):
    for node in community:
        community_data.append({
            "nodeId": node,
            "patientId": nodeid_to_patientid[node],
            "communityId": idx
        })

community_df = pd.DataFrame(community_data)
community_df.to_csv(os.path.join(output_dir, "community_assignments.csv"), index=False)

# Compute metrics
modularity = evaluation.modularity_overlap(G, slpa_communities).score
conductance = evaluation.conductance(G, slpa_communities).score

# Print summary
num_communities = len(slpa_communities.communities)
num_nodes = len(set([node for comm in slpa_communities.communities for node in comm]))
avg_size = sum(len(comm) for comm in slpa_communities.communities) / num_communities
min_size = min(len(comm) for comm in slpa_communities.communities)
max_size = max(len(comm) for comm in slpa_communities.communities)

metrics_df = pd.DataFrame([{
    "modularity": modularity,
    "conductance": conductance,
    'num_communities': num_communities,
    'num_nodes_in_communities': num_nodes,
    'avg_community_size': avg_size,
    'min_community_size': min_size,
    'max_community_size': max_size
}])
metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)


print(f"SLPA Community Detection Summary:")
print(f"Number of communities: {num_communities}")
print(f"Number of nodes in communities: {num_nodes}")
print(f"Average community size: {avg_size:.2f}")
print(f"Min community size: {min_size}")
print(f"Max community size: {max_size}")
print(f"Overlapping Modularity: {modularity:.4f}")
print(f"Overlapping Conductance: {conductance:.4f}")

