import os
import pandas as pd
import networkx as nx
from community_detection.io_utils import load_triplets, save_community_result, save_metrics, print_community_summary
from community_detection.graph_utils import build_graph_from_triplets, remap_graph_for_algorithm, remap_node_communities
from community_detection.evaluation import evaluate_internal, evaluate_external
from community_detection.io_utils import load_mapping, save_mapping, ensure_dir, load_community_result
from cdlib import algorithms, NodeClustering
from graspologic.partition import hierarchical_leiden

def workflow_construct_graph(triplet_path, triplet_key, data_dir):
    mapping_path = os.path.join(data_dir, triplet_key, "mapping.csv")
    G = build_graph_from_triplets(load_triplets(triplet_path))
    if os.path.exists(mapping_path):
        print("Loading mapping from file.")
        mapping = load_mapping(mapping_path)
        rev_mapping = {v: k for k, v in mapping.items()}
        G_int = nx.relabel_nodes(G, rev_mapping)
    else:
        print("Remapping graph to integer node ids.")
        G_int, mapping, rev_mapping = remap_graph_for_algorithm(G)
        ensure_dir(os.path.join(data_dir, triplet_key))
        save_mapping(mapping, mapping_path)
    return G, G_int, mapping

def workflow_slpa(G_int, triplet_key, data_dir):
    print("Running SLPA community detection...")
    algo = "slpa"
    out_dir = os.path.join(data_dir, triplet_key, algo)
    ensure_dir(out_dir)
    comm_path = os.path.join(out_dir, "communities.csv")
    metrics_path = os.path.join(out_dir, "metrics.csv")
    if os.path.exists(comm_path):
        print("SLPA communities already exist.")
        community = load_community_result(comm_path)
    else:
        community = algorithms.slpa(G_int)
        save_community_result(community, comm_path)
    print_community_summary(community)
    if os.path.exists(metrics_path):
        print("SLPA metrics already exist, skipping evaluation.")
        return
    community.overlap = True
    metrics = evaluate_internal(G_int, community)
    metrics["algo"] = algo
    save_metrics(metrics, metrics_path)

def workflow_hierarchical_leiden(G_int, triplet_key, data_dir, mapping, df, label_columns):
    print("Running Hierarchical Leiden community detection...")
    algo = "hierarchical_leiden"
    out_dir = os.path.join(data_dir, triplet_key, algo)
    ensure_dir(out_dir)
    metrics_path = os.path.join(out_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        print("Hierarchical Leiden metrics already exist, skipping.")
        return
    
    parents_path = os.path.join(out_dir, "parents.csv")
    parents_df = []
    partitions = hierarchical_leiden(G_int, max_cluster_size=100, random_seed=42)
    level_to_communities = {}
    for part in partitions:
        level = part.level
        comm_id = part.cluster
        node = part.node
        if level not in level_to_communities:
            level_to_communities[level] = {}
        if comm_id not in level_to_communities[level]:
            level_to_communities[level][comm_id] = []
        level_to_communities[level][comm_id].append(node)
        parent_id = part.parent_cluster if part.parent_cluster is not None else -1
        parents_df.append({
            "community_id": comm_id,
            "parent_community_id": parent_id,
            "level": level
        })
    pd.DataFrame(parents_df).to_csv(parents_path, index=False)
    metrics_rows = []
    for level in sorted(level_to_communities.keys()):
        comms = list(level_to_communities[level].values())
        comm_path = os.path.join(out_dir, f"communities_level{level}.csv")
        # Save communities for this level
        comm_obj = NodeClustering(comms, graph=G_int, method_name="Hierarchical Leiden", overlap=False)
        save_community_result(comm_obj, comm_path)
        print_community_summary(comm_obj)
        stats = {
            "algo": f"{algo}_level{level}",
        }
        # Internal metrics
        internal = evaluate_internal(G_int, comm_obj)
        stats.update(internal)
        # External metrics
        external = evaluate_external(comm_obj, mapping, df, label_columns)
        stats.update(external)
        metrics_rows.append(stats)
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

def workflow_slpa_on_leiden_level(G_int, triplet_key, data_dir, leiden_level):
    print(f"Running SLPA on Leiden level {leiden_level} communities...")
    leiden_dir = os.path.join(data_dir, triplet_key, "hierarchical_leiden")
    comm_path = os.path.join(leiden_dir, f"communities_level{leiden_level}.csv")
    out_dir = os.path.join(data_dir, triplet_key, f"slpa_on_leiden_level{leiden_level}")
    ensure_dir(out_dir)
    metrics_path = os.path.join(out_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        print(f"SLPA on Leiden level {leiden_level} metrics already exist, skipping.")
        return
    # Load communities for this level
  
    leiden_comm = load_community_result(comm_path)
    print(f"Loaded {len(leiden_comm.communities)} communities from Leiden level {leiden_level}.")
    all_slpa_communities = []
    for comm in leiden_comm.communities:
        subG = G_int.subgraph(comm).copy()
        slpa_result = algorithms.slpa(subG)
        all_slpa_communities.extend([list(c) for c in slpa_result.communities])
    comm_obj = NodeClustering(all_slpa_communities, graph=G_int, method_name=f"SLPA_on_Leiden_level{leiden_level}", overlap=True)
    save_community_result(comm_obj, os.path.join(out_dir, "communities.csv"))
    print_community_summary(comm_obj)
    metrics = evaluate_internal(G_int, comm_obj)
    metrics["algo"] = f"slpa_on_leiden_level{leiden_level}"
    save_metrics(metrics, metrics_path)

