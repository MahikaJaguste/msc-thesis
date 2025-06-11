import pandas as pd
import os

def make_df_with_community(mode, triplet_key, clinical_path, node_mapping_path):
    """
    Returns df_with_community for the given mode and triplet_key.
    """
    # Load clinical and node mapping
    df = pd.read_csv(clinical_path)
    node_id_df = pd.read_csv(node_mapping_path)
    # Compose community mapping path
    community_path = os.path.join("..", "data", mode, triplet_key, "community_mapping.csv")
    community_df = pd.read_csv(community_path)
    community_df['node_id'] = community_df['node_id'].astype(int)
    # Add node_id to df, using patientId in df and "Patient_{patientId}" in node_to_id
    df_with_id = df.copy(deep=True)
    df_with_id['Patient'] = df_with_id['patientId'].apply(lambda x: f"Patient_{x}")
    df_with_id = df_with_id.merge(node_id_df, left_on='Patient', right_on='node_name', how='left')
    df_with_id = df_with_id.rename(columns={'node_id': 'patientNodeId'})
    df_with_id = df_with_id.drop(columns=['node_name', 'Patient'])
    # Merge with community assignment
    df_with_community = df_with_id.merge(community_df, left_on='patientNodeId', right_on='node_id', how='left')
    # drop rows where community_id is NaN
    df_with_community = df_with_community.dropna(subset=['community_id'])
    return df_with_community