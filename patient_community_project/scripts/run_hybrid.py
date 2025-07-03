import os
import pandas as pd
import networkx as nx
from cdlib import evaluation
from cdlib import NodeClustering
from utils.w_slpa import weighted_slpa
from utils.overlapping_summary import print_overlapping_node_summary

# Define paths
GRAPH_NAME = "full"
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../data/{GRAPH_NAME}")
)
GRAPH_PATH = os.path.join(DATA_PATH, "graph.gml")
MAPPING_PATH = os.path.join(DATA_PATH, "patient_id_mapping.csv")
LEIDEN_COMMUNITY_PATH = os.path.join(DATA_PATH, "leiden", "level_0_community_assignments.csv")
OUTPUT_DIR = os.path.join(DATA_PATH, "hybrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load graph
G = nx.read_gml(GRAPH_PATH, label='id')

# Load mapping
mapping_df = pd.read_csv(MAPPING_PATH)
mapping_df['nodeId'] = mapping_df['nodeId'].astype(int)
nodeid_to_patientid = dict(zip(mapping_df["nodeId"], mapping_df["patientId"]))

# Load Leiden level 0 community assignments
leiden_df = pd.read_csv(LEIDEN_COMMUNITY_PATH)
leiden_df['nodeId'] = leiden_df['nodeId'].astype(int)

# Run SLPA on each Leiden community subgraph
hybrid_communities = []
community_counter = 0

for community_id in leiden_df['communityId'].unique():
    sub_nodes = leiden_df[leiden_df['communityId'] == community_id]['nodeId'].tolist()
    subgraph = G.subgraph(sub_nodes).copy()
    # slpa_result = algorithms.slpa(subgraph, t=100, r=0.01).communities
    slpa_result = weighted_slpa(subgraph)

    for comm in slpa_result:
        hybrid_communities.append({
            "communityId": community_counter,
            "members": comm
        })
        community_counter += 1

# Save community assignments
community_data = []
for comm in hybrid_communities:
    for node in comm["members"]:
        community_data.append({
            "nodeId": node,
            "patientId": nodeid_to_patientid[node],
            "communityId": comm["communityId"]
        })

community_df = pd.DataFrame(community_data)
community_df.to_csv(os.path.join(OUTPUT_DIR, "community_assignments.csv"), index=False)

# Compute metrics
all_communities = [comm["members"] for comm in hybrid_communities]

clustering = NodeClustering(communities=all_communities, graph=G, method_name="hybrid_leiden_slpa", overlap=True)

modularity = evaluation.modularity_overlap(G, clustering).score
conductance = evaluation.conductance(G, clustering).score


# Print summary
num_communities = len(all_communities)
num_nodes = len(set([node for comm in all_communities for node in comm]))
avg_size = sum(len(comm) for comm in all_communities) / num_communities
min_size = min(len(comm) for comm in all_communities)
max_size = max(len(comm) for comm in all_communities)

metrics_df = pd.DataFrame([{
    "modularity": modularity,
    "conductance": conductance,
    'num_communities': num_communities,
    'num_nodes_in_communities': num_nodes,
    'avg_community_size': avg_size,
    'min_community_size': min_size,
    'max_community_size': max_size
}])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

print(f"Hybrid Community Detection Summary:")
print(f"Number of communities: {num_communities}")
print(f"Number of nodes in communities: {num_nodes}")
print(f"Average community size: {avg_size:.2f}")
print(f"Min community size: {min_size}")
print(f"Max community size: {max_size}")
print(f"Overlapping Modularity: {modularity:.4f}")
print(f"Overlapping Conductance: {conductance:.4f}")

print_overlapping_node_summary(all_communities, G, nodeid_to_patientid)