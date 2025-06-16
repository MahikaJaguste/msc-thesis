from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from cdlib.evaluation import newman_girvan_modularity, modularity_overlap, conductance

import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_internal(G, community):
    mod = newman_girvan_modularity(G, community).score
    print(f"Modularity: {mod:.4f}")

    if community.overlap:
        overlapping_mod = modularity_overlap(G, community).score
        print(f"Overlapping Modularity: {overlapping_mod:.4f}")
        shen_overlapping_mod = 0 # shen_overlapping_modularity(G, community.communities)

    cond = conductance(G, community).score
    print(f"Conductance: {cond:.4f}")

    num_communities = len(community.communities)
    sizes = [len(c) for c in community.communities]
    avg_size = sum(sizes) / num_communities if num_communities > 0 else 0
    max_size = max(sizes) if sizes else 0
    min_size = min(sizes) if sizes else 0
    nodes_set_in_communities = set(node for com in community.communities for node in com)
    num_nodes_in_communities = len(nodes_set_in_communities)

    return {
        'modularity': mod, 
        'conductance': cond, 
        'overlapping_modularity': overlapping_mod if community.overlap else None, 
        'shen_overlapping_modularity': shen_overlapping_mod if community.overlap else None,
        'num_communities': num_communities,
        'avg_size': avg_size,
        'max_size': max_size,
        'min_size': min_size,
        'num_nodes_in_communities': num_nodes_in_communities
        }

def evaluate_external(community, mapping, df, label_columns):
    results = {}

    df_patients = df.copy(deep=True)

    df_patients['community'] = -1
    df_patients['patientId'] = 'Patient_' + df_patients['patientId'].astype(str)
    for i, com in enumerate(community.communities):
        for node in com:
            if mapping[node].startswith('Patient_'):
                df_patients.loc[df_patients['patientId'] == mapping[node], 'community'] = i

    for label_col in label_columns:
        true_labels = df_patients[label_col].tolist()
        cluster_labels = df_patients['community'].tolist()
        try:
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            ari = adjusted_rand_score(true_labels, cluster_labels)
            results[f'{label_col}_NMI'] = nmi
            results[f'{label_col}_ARI'] = ari
        except:
            results[f'{label_col}_NMI'] = None
            results[f'{label_col}_ARI'] = None
    return results



def _community_modularity(args):
    idx, comm, G, node_alpha, m = args
    comm_set = set(comm)
    Q_c = 0.0
    for i in comm_set:
        for j in comm_set:
            A_ij = 1 if G.has_edge(i, j) else 0
            k_i = G.degree(i)
            k_j = G.degree(j)
            alpha_i = node_alpha.get((i, idx), 0)
            alpha_j = node_alpha.get((j, idx), 0)
            Q_c += (A_ij - (k_i * k_j) / (2 * m)) * alpha_i * alpha_j
    return Q_c

def shen_overlapping_modularity(G, communities, n_jobs=4):
    m = G.number_of_edges()
    if m == 0:
        print("Graph has no edges. Modularity is 0.")
        return 0.0

    print(f"Preparing node-community memberships for {len(communities)} communities...")
    node_to_comms = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_comms.setdefault(node, set()).add(idx)

    node_alpha = {}
    for node, comms in node_to_comms.items():
        for comm in comms:
            node_alpha[(node, comm)] = 1.0 / len(comms)

    args = [(idx, comm, G, node_alpha, m) for idx, comm in enumerate(communities)]

    Q = 0.0
    total = len(args)
    print(f"Starting parallel modularity computation for {total} communities...")
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_community_modularity, arg): i for i, arg in enumerate(args)}
        for count, future in enumerate(as_completed(futures), 1):
            idx, Q_c = future.result()
            Q += Q_c
            if count % 10 == 0 or count == total:
                print(f"Processed {count}/{total} communities...")

    Q = Q / (2 * m)
    print(f"Final overlapping modularity: {Q:.6f}")
    return Q