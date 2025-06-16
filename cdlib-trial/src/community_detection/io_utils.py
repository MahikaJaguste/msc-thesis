import os
import pandas as pd
from cdlib.readwrite import write_community_csv, read_community_csv

def load_triplets(path):
    df = pd.read_csv(path)
    return df

def save_community_result(community, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    write_community_csv(community, save_path)

def load_community_result(path):
    return read_community_csv(path, nodetype=int)

def save_metrics(metrics_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame([metrics_dict]).to_csv(save_path, index=False)

def print_community_summary(comm_obj):
    print(f"→ Detected {len(comm_obj.communities)} communities.")
    sizes = [len(c) for c in comm_obj.communities]
    if sizes:
        print(f"→ Avg community size: {sum(sizes)/len(sizes):.2f}")
        print(f"→ Min size: {min(sizes)}, Max size: {max(sizes)}")
    else:
        print("→ No communities detected.")
    

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_mapping(mapping, path):
    mapping_df = pd.DataFrame(list(mapping.items()), columns=['node', 'original_label'])
    mapping_df.set_index('node', inplace=True)
    mapping_df.to_csv(path)

def load_mapping(path):
    if not os.path.exists(path):
        return None
    mapping_df = pd.read_csv(path, index_col='node')
    return mapping_df['original_label'].to_dict()
