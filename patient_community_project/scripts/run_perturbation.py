import os
import random
import pandas as pd
import networkx as nx
from collections import defaultdict
from cdlib.evaluation import (
    normalized_mutual_information,
    adjusted_rand_index,
    overlapping_normalized_mutual_information_LFK,
    overlapping_normalized_mutual_information_MGH,
    omega
)
from cdlib import NodeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from itertools import combinations

GRAPH_NAME = "full"
# PERTURBATION_MODE = "normal"
# PERTURBATION_MODE = 'edge'
# PERTURBATION_MODE = "node"
PERTURBATION_MODE = "noise"
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../data/{GRAPH_NAME}")
)

graph_path = os.path.join(DATA_PATH, "graph.gml")
if PERTURBATION_MODE != "normal":
    graph_path = os.path.join(DATA_PATH, PERTURBATION_MODE, "graph.gml")

og_output_dir = os.path.join(DATA_PATH, "w_slpa")
results_path = os.path.join(DATA_PATH, PERTURBATION_MODE, "robustness_metrics.csv")

print("PERTURBATION_MODE:", PERTURBATION_MODE)

os.makedirs(os.path.dirname(results_path), exist_ok=True)

def weighted_slpa(graph, t=20, r=0.1, seed=None):
    if seed is not None:
        random.seed(seed)

    memory = {node: [node] for node in graph.nodes()}
    for _ in range(t):
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        for listener in nodes:
            label_counts = defaultdict(float)
            for neighbor in graph.neighbors(listener):
                speaker_memory = memory[neighbor]
                label = random.choice(speaker_memory)
                weight = graph[listener][neighbor].get('weight', 0.0)
                label_counts[label] += weight
            if label_counts:
                selected_label = max(label_counts.items(), key=lambda x: x[1])[0]
                memory[listener].append(selected_label)

    communities = defaultdict(list)
    for node, labels in memory.items():
        label_freq = defaultdict(int)
        for label in labels:
            label_freq[label] += 1
        total = len(labels)
        for label, count in label_freq.items():
            freq = count / total
            if freq >= r:
                communities[label].append(node)

    return list(communities.values())


def make_disjoint_clustering(membership_dict):
    disjoint_communities = defaultdict(list)
    assigned_nodes = set()
    for comm_id in sorted(set(c for comms in membership_dict.values() for c in comms)):
        for node, comms in membership_dict.items():
            if node not in assigned_nodes and comm_id in comms:
                disjoint_communities[comm_id].append(node)
                assigned_nodes.add(node)
    return list(disjoint_communities.values())


# Load baseline community assignments
baseline_df = pd.read_csv(os.path.join(og_output_dir, "community_assignments.csv"))
baseline_membership = defaultdict(set)
for _, row in baseline_df.iterrows():
    baseline_membership[row["nodeId"]].add(row["communityId"])

# Load graph
G = nx.read_gml(graph_path, label='id')
print(G)

# Normalize edge weights
weights = [d['weight'] for u, v, d in G.edges(data=True)]
min_w, max_w = min(weights), max(weights)
for u, v, d in G.edges(data=True):
    d['weight'] = (d['weight'] - min_w) / (max_w - min_w + 1e-9)

# Define seeds
seeds = [42, 101, 202, 303, 404]

# Prepare results
results = []

# Run SLPA multiple times and compute metrics
for seed in seeds:
    communities = weighted_slpa(G, seed=seed)
    clustering = NodeClustering(communities=communities, graph=G, method_name="weighted_slpa", overlap=True)

    # Create node to community mapping for comparison
    run_membership = defaultdict(set)
    for idx, comm in enumerate(communities):
        for node in comm:
            run_membership[node].add(idx)

    # Determine nodes to evaluate
    if PERTURBATION_MODE == "node":
        valid_nodes = set(G.nodes())
        filtered_baseline_membership = defaultdict(set)
        for node in valid_nodes:
            if node in baseline_membership:
                filtered_baseline_membership[node] = baseline_membership[node]
    else:
        filtered_baseline_membership = baseline_membership

    # Convert filtered baseline membership to NodeClustering
    baseline_communities = defaultdict(list)
    for node, comms in filtered_baseline_membership.items():
        for comm in comms:
            baseline_communities[comm].append(node)
    baseline_clustering = NodeClustering(
        communities=list(baseline_communities.values()),
        graph=G,
        method_name="baseline",
        overlap=True
    )


    # Disjoint baseline clustering
    disjoint_baseline_comms = make_disjoint_clustering(filtered_baseline_membership)
    disjoint_baseline_clustering = NodeClustering(
        communities=disjoint_baseline_comms,
        graph=G,
        method_name="baseline_disjoint",
        overlap=False
    )

    # Disjoint run clustering
    disjoint_run_comms = make_disjoint_clustering(run_membership)
    disjoint_run_clustering = NodeClustering(
        communities=disjoint_run_comms,
        graph=G,
        method_name="run_disjoint",
        overlap=False
    )


    # Compute CDlib metrics
    ari = adjusted_rand_index(disjoint_run_clustering, disjoint_baseline_clustering).score
    nmi = normalized_mutual_information(disjoint_run_clustering, disjoint_baseline_clustering).score
    onmi_lfk = overlapping_normalized_mutual_information_LFK(clustering, baseline_clustering).score
    onmi_mgh = overlapping_normalized_mutual_information_MGH(clustering, baseline_clustering).score
    omega_index = omega(clustering, baseline_clustering).score

    # # Percentage of nodes that changed communities
    # changed = sum(1 for n in all_nodes if baseline_membership[n] != run_membership[n])
    # pct_changed = changed / len(all_nodes) * 100

    results.append({
        "seed": seed,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "omega_index": omega_index,
        "onmi_lfk": onmi_lfk,
        "onmi": onmi_mgh,
        # "percent_nodes_changed": pct_changed,
        "omega_index": omega_index,
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(results_path, index=False)
print("Robustness metrics saved to 'robustness_metrics.csv'")


