from cdlib import evaluation
from cdlib.classes import NodeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import pandas as pd
import os

def compute_modularity(G, communities):
    """
    Compute modularity for a given partition (list of lists of node ids) using Newman-Girvan modularity.
    """
    # Wrap in NodeClustering for cdlib evaluation functions
    comm_obj = NodeClustering(communities, graph=G, method_name="custom")
    return evaluation.newman_girvan_modularity(G, comm_obj).score

def compute_conductance(G, communities):
    """
    Compute mean conductance for a given partition (list of lists of node ids).
    """
    comm_obj = NodeClustering(communities, graph=G, method_name="custom")
    return evaluation.conductance(G, comm_obj).score


def compute_nmi_ari(df_with_community, level, label_col):
    subset = df_with_community[(df_with_community['level'] == level) & (df_with_community[label_col].notnull())]
    if subset.empty:
        return None, None
    true_labels = subset[label_col]
    pred_labels = subset['community_id']
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return nmi, ari

def print_and_store_level_evaluations(
    G, community_df, levels, df_with_community, label_cols, mode, triplet_key, out_base=".."
):
    """
    For each level, compute and print modularity, conductance, NMI, ARI, and store in CSV.
    """
    results = []
    for level in levels:
        comms = []
        for comm_id in community_df[community_df['level'] == level]['community_id'].unique():
            nodes = community_df[(community_df['level'] == level) & (community_df['community_id'] == comm_id)]['node_id'].tolist()
            comms.append(nodes)
        mod = compute_modularity(G, comms)
        cond = compute_conductance(G, comms)
        row = {
            "algoname_level": f"{mode}_level{level}",
            "modularity": mod,
            "conductance": cond
        }
        print(f"Level {level}: {len(comms)} communities | Modularity: {mod:.4f} | Mean Conductance: {cond:.4f}")
        # Compute NMI/ARI for each label
        for label in label_cols:
            nmi, ari = compute_nmi_ari(df_with_community, level, label)
            row[f"{label}_nmi"] = nmi
            row[f"{label}_ari"] = ari
            print(f"  {label}: NMI={nmi if nmi is not None else 'NA'} | ARI={ari if ari is not None else 'NA'}")
        results.append(row)
    print()
    # Save results to CSV
    out_dir = os.path.join(out_base, "data", mode, triplet_key)
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, "results.csv")
    pd.DataFrame(results).to_csv(results_csv, index=False)
    print(f"Saved metrics to {results_csv}")
