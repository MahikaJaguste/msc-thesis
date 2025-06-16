from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from cdlib.evaluation import newman_girvan_modularity, modularity_overlap, conductance
import numpy as np

def evaluate_internal(G, community):
    mod = newman_girvan_modularity(G, community).score

    if community.overlap:
        overlapping_mod = modularity_overlap(G, community).score

    cond = conductance(G, community).score
    return {'modularity': mod, 'conductance': cond, 'overlapping_modularity': overlapping_mod if community.overlap else None}

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
