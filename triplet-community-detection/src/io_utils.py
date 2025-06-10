import pandas as pd

def load_triplets_csv(csv_path):
    """
    Load triplet CSV into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    print(df.head())
    return df