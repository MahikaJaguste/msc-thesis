from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from cdlib.evaluation import newman_girvan_modularity, modularity_overlap, conductance, erdos_renyi_modularity

import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_internal(G, community):
    newman_girvan_mod = None
    erdos_renyi_mod = None
    overlapping_mod = None

    if not community.overlap:
        newman_girvan_mod = newman_girvan_modularity(G, community).score
        print(f"Newman-Girvan Modularity: {newman_girvan_mod:.4f}")

        erdos_renyi_mod = erdos_renyi_modularity(G, community).score
        print(f"Erdos-Renyi Modularity: {erdos_renyi_mod:.4f}")
    else:
        overlapping_mod = modularity_overlap(G, community).score
        print(f"Overlapping Modularity: {overlapping_mod:.4f}")
        

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
        'newman_girvan_modularity': newman_girvan_mod,
        'erdos_renyi_modularity': erdos_renyi_mod,
        'overlapping_modularity': overlapping_mod,
        'conductance': cond, 
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


    """
    Compute overlapping modularity for a list of communities.
    Each node's contribution to a community is 1/(number of communities it belongs to).
    communities: list of lists of node ids (overlapping allowed)
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0

    # 1. Build node -> set of communities mapping
    node_to_comms = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_comms.setdefault(node, set()).add(idx)
    print("Node to communities mapping done.")

    # 2. Precompute alpha for each node in each community
    node_alpha = {}
    for node, comms in node_to_comms.items():
        for comm in comms:
            node_alpha[(node, comm)] = 1.0 / len(comms)
    print("Node alpha values computed.")

    Q = 0.0
    for idx, comm in enumerate(communities):
        comm_set = set(comm)
        for i in comm_set:
            for j in comm_set:
                A_ij = 1 if G.has_edge(i, j) else 0
                k_i = G.degree(i)
                k_j = G.degree(j)
                alpha_i = node_alpha.get((i, idx), 0)
                alpha_j = node_alpha.get((j, idx), 0)
                Q += (A_ij - (k_i * k_j) / (2 * m)) * alpha_i * alpha_j

    Q = Q / (2 * m)
    return Q