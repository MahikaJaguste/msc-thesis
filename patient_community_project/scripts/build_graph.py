import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import Binarizer
from sklearn.metrics import jaccard_score
import networkx as nx

THRESHOLD = 0.8  # similarity threshold to create edges
# Choose similarity function
SIMILARITY_TYPE = "cosine"
# SIMILARITY_TYPE = "euclidean"

# Configuration
GRAPH_NAME = "full"
PATIENT_DF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/thesis/cll_broad_2022_clinical_data_thesis.csv")
)
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../data/{GRAPH_NAME}")
)

os.makedirs(DATA_PATH, exist_ok=True)
patient_df = pd.read_csv(PATIENT_DF_PATH)

# Select treatment response features
features = [
    'FFS_MONTHS', 
    'FFS_STATUS', 
    'OS_MONTHS', 
    'OS_STATUS'
]
response_df = patient_df[['patientId'] + features].dropna()

# Convert categorical features to numerical
response_df['FFS_STATUS'] = response_df['FFS_STATUS'].map(lambda x: int(x[0]))
response_df['OS_STATUS'] = response_df['OS_STATUS'].map(lambda x: int(x[0]))


# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(response_df[features])

def compute_similarity(X, similarity_type):
    if similarity_type == "cosine":
        # Cosine similarity ranges from -1 (opposite) to 1 (identical).
        # To normalize to [0,1]: sim_norm = (sim + 1) / 2
        sim = cosine_similarity(X)
        sim_norm = (sim + 1) / 2
        return sim_norm
    elif similarity_type == "euclidean":
        # Euclidean distance: lower means more similar. We want higher = more similar.
        # Invert and normalize: sim = 1 - (dist / max_dist), so that 0 distance -> 1, max distance -> 0
        dist = euclidean_distances(X)
        max_dist = dist.max()
        sim = 1 - dist / max_dist if max_dist > 0 else 1 - dist
        return sim
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")

# Compute similarity matrix
similarity_matrix = compute_similarity(X, SIMILARITY_TYPE)

# Create graph
G = nx.Graph()
node_mapping = {}
for idx, patient_id in enumerate(response_df['patientId']):
    G.add_node(idx, patientId=patient_id)
    node_mapping[patient_id] = idx

# Add edges based on similarity threshold
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        sim = similarity_matrix[i, j]
        if sim >= THRESHOLD:
            G.add_edge(i, j, weight=sim)

# Save graph
nx.write_gml(G, os.path.join(DATA_PATH, "graph.gml"))

# Save mapping
mapping_df = pd.DataFrame(list(node_mapping.items()), columns=["patientId", "nodeId"])
mapping_df.to_csv(os.path.join(DATA_PATH, "patient_id_mapping.csv"), index=False)

# Print graph summary
print(f"Graph Summary:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
print(f"Density: {nx.density(G):.4f}")

