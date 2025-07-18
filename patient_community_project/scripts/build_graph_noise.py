import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Set seed for reproducibility
np.random.seed(42)

# === Paths ===


# Configuration
GRAPH_NAME = "full"
PATIENT_DF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/thesis/cll_broad_2022_clinical_data_thesis.csv")
)
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../data/{GRAPH_NAME}/noise")
)

graph_path = os.path.join(DATA_PATH, "graph.gml")
mapping_path = os.path.join(DATA_PATH, "patient_id_mapping.csv")
output_csv_path = os.path.join(DATA_PATH, "cll_broad_2022_clinical_data_thesis.csv")

os.makedirs(DATA_PATH, exist_ok=True)

# === Step 1: Load and Perturb Dataset ===
if 'cll_broad_2022_clinical_data_thesis.csv' not in os.listdir(DATA_PATH):
    df = pd.read_csv(PATIENT_DF_PATH)

    for col in ['FFS_MONTHS', 'OS_MONTHS']:
        if col in df.columns:
            mask = df[col].notnull()
            noise = np.random.normal(loc=0, scale=3, size=mask.sum())  # mean=0, std=3
            df.loc[mask, col] += noise

    df.to_csv(output_csv_path, index=False)
    print(f"✅ Perturbed dataset saved to: {output_csv_path}")
else:
    print(f"⚠️ Perturbed dataset already exists at: {output_csv_path}")
    df = pd.read_csv(output_csv_path)

# === Step 2: Build Graph ===
features = ['FFS_MONTHS', 'FFS_STATUS', 'OS_MONTHS', 'OS_STATUS']
response_df = df[['patientId'] + features].dropna()

response_df['FFS_STATUS'] = response_df['FFS_STATUS'].map(lambda x: int(x[0]))
response_df['OS_STATUS'] = response_df['OS_STATUS'].map(lambda x: int(x[0]))

scaler = StandardScaler()
X = scaler.fit_transform(response_df[features])

sim = cosine_similarity(X)
sim_norm = (sim + 1) / 2

G = nx.Graph()
node_mapping = {}
for idx, patient_id in enumerate(response_df['patientId']):
    G.add_node(idx, patientId=patient_id)
    node_mapping[patient_id] = idx

THRESHOLD = 0.8
for i in range(len(sim_norm)):
    for j in range(i + 1, len(sim_norm)):
        if sim_norm[i, j] >= THRESHOLD:
            G.add_edge(i, j, weight=sim_norm[i, j])

nx.write_gml(G, graph_path)
mapping_df = pd.DataFrame(list(node_mapping.items()), columns=["patientId", "nodeId"])
mapping_df.to_csv(mapping_path, index=False)

print(f"✅ Perturbed dataset saved to: {output_csv_path}")
print(f"✅ Graph saved to: {graph_path}")
print(f"✅ Mapping saved to: {mapping_path}")
print(f"📊 Graph Summary: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
