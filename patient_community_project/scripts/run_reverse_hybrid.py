import os
import pandas as pd
import networkx as nx
from graspologic.partition import leiden
from cdlib import NodeClustering, evaluation

# Define paths
GRAPH_NAME = "full"
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../data/{GRAPH_NAME}")
)
GRAPH_PATH = os.path.join(DATA_PATH, "graph.gml")
MAPPING_PATH = os.path.join(DATA_PATH, "patient_id_mapping.csv")
SLPA_COMMUNITY_PATH = os.path.join(DATA_PATH, "w_slpa", "community_assignments.csv")
OUTPUT_DIR = os.path.join(DATA_PATH, "reverse_hybrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load graph
G = nx.read_gml(GRAPH_PATH, label='id')

# Load mapping
mapping_df = pd.read_csv(MAPPING_PATH)
mapping_df['nodeId'] = mapping_df['nodeId'].astype(int)
nodeid_to_patientid = dict(zip(mapping_df["nodeId"], mapping_df["patientId"]))
patientid_to_nodeid = dict(zip(mapping_df["patientId"], mapping_df["nodeId"]))

# Load SLPA communities
slpa_df = pd.read_csv(SLPA_COMMUNITY_PATH)
# Group nodes by SLPA community
slpa_df['nodeId'] = slpa_df['nodeId'].astype(int)
slpa_communities = [group['nodeId'].tolist() for _, group in slpa_df.groupby('communityId')]

all_leiden_communities = []
community_counter = 0
community_data = []


# Predefined mapping from community sizes to community IDs
# size_to_commid = {
#     156: 0,
#     50: 1,
#     80: 2,
#     29: 3,
#     37: 4,
#     117: 5,
#     104: 6,
#     110: 7,
#     231: 8,
#     148: 9
# }


index = 0
for comm_nodes in slpa_communities:
    print(f"Processing SLPA community {index}/{len(slpa_communities)} with {len(comm_nodes)} nodes")
    subgraph = G.subgraph(comm_nodes).copy()
    if subgraph.number_of_nodes() == 0:
        continue
    # Run Leiden on the subgraph
    leiden_partition = leiden(subgraph, random_seed=42)
    # Collect communities
    comm_map = {}
    for node, comm in leiden_partition.items():
        comm_map.setdefault(comm, []).append(node)
    for comm_nodes_list in comm_map.values():
        all_leiden_communities.append(comm_nodes_list)
        
        # comm_size = len(comm_nodes_list)
        # if comm_size in size_to_commid:
        #             community_id = size_to_commid[comm_size]
        # else:
        #     community_id = f"unmapped_{comm_size}"

        for node in comm_nodes_list:
            community_data.append({
                "nodeId": node,
                "patientId": nodeid_to_patientid.get(node, node),
                # "communityId": community_id
                "communityId": community_counter
            })
        # print(community_id, ": ", len(comm_nodes_list))
        print(community_counter, ": ", len(comm_nodes_list))
        community_counter += 1
    index += 1
    print("Leiden communities found:", len(comm_map))

# Save merged community assignments
os.makedirs(OUTPUT_DIR, exist_ok=True)
community_df = pd.DataFrame(community_data)
community_df.to_csv(os.path.join(OUTPUT_DIR, "community_assignments.csv"), index=False)

# Compute and print metrics
clustering = NodeClustering(communities=all_leiden_communities, graph=G, method_name="reverse_hybrid_leiden")
mod = evaluation.newman_girvan_modularity(G, clustering).score
cond = evaluation.conductance(G, clustering).score
overlapping_mod = evaluation.modularity_overlap(G, clustering).score

num_communities = len(all_leiden_communities)
avg_size = sum(len(c) for c in all_leiden_communities) / num_communities
min_size = min(len(c) for c in all_leiden_communities)
max_size = max(len(c) for c in all_leiden_communities)
num_nodes_in_communities = len(set(node for comm in all_leiden_communities for node in comm))

print(f"Reverse Hybrid Leiden Community Detection Summary:")
print(f"Number of communities: {num_communities}")
print(f"Average community size: {avg_size:.2f}")
print(f"Min community size: {min_size}")
print(f"Max community size: {max_size}")
print(f"Number of nodes in communities: {num_nodes_in_communities}")
print(f"Modularity: {mod:.4f}")
print(f"Overlapping Modularity: {overlapping_mod:.4f}")
print(f"Conductance: {cond:.4f}")

# Save metrics
metrics_df = pd.DataFrame([{
    "modularity": mod,
    "overlapping_modularity": overlapping_mod,
    "conductance": cond,
    'num_communities': num_communities,
    'avg_community_size': avg_size,
    'min_community_size': min_size,
    'max_community_size': max_size,
    'num_nodes_in_communities': num_nodes_in_communities
}])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)