import os
import random
import pandas as pd
import networkx as nx
from collections import defaultdict
from cdlib import evaluation
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
        all_nodes = sorted(set(baseline_membership.keys()) & set(run_membership.keys()) & set(G.nodes()))
    else:
        all_nodes = sorted(set(baseline_membership.keys()) & set(run_membership.keys()))

    baseline_labels = [sorted(baseline_membership[n])[0] if baseline_membership[n] else -1 for n in all_nodes]
    run_labels = [sorted(run_membership[n])[0] if run_membership[n] else -1 for n in all_nodes]

    ari = adjusted_rand_score(baseline_labels, run_labels)
    nmi = normalized_mutual_info_score(baseline_labels, run_labels)

    # Jaccard similarity for overlapping sets
    jaccard_scores = []
    for node in all_nodes:
        a = baseline_membership[node]
        b = run_membership[node]
        intersection = len(a & b)
        union = len(a | b)
        jaccard_scores.append(intersection / union if union > 0 else 0)
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)

    # Omega Index
    def omega_index(m1, m2):
        agree = 0
        total = 0
        for u, v in combinations(all_nodes, 2):
            same1 = len(m1[u] & m1[v]) > 0
            same2 = len(m2[u] & m2[v]) > 0
            if same1 == same2:
                agree += 1
            total += 1
        return agree / total if total > 0 else 0

    omega = omega_index(baseline_membership, run_membership)

    # Percentage of nodes that changed communities
    changed = sum(1 for n in all_nodes if baseline_membership[n] != run_membership[n])
    pct_changed = changed / len(all_nodes) * 100

    results.append({
        "seed": seed,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "avg_jaccard_similarity": avg_jaccard,
        "omega_index": omega,
        "percent_nodes_changed": pct_changed
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(results_path, index=False)
print("Robustness metrics saved to 'robustness_metrics.csv'")


