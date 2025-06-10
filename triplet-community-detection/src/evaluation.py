from cdlib import evaluation
from cdlib.classes import NodeClustering

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

def print_level_evaluations(G, community_df, levels):
    """
    For each level, compute and print modularity and conductance.
    """
    for level in levels:
        # Get partition as list of lists of node ids
        comms = []
        for comm_id in community_df[community_df['level'] == level]['community_id'].unique():
            nodes = community_df[(community_df['level'] == level) & (community_df['community_id'] == comm_id)]['node_id'].tolist()
            comms.append(nodes)
        mod = compute_modularity(G, comms)
        cond = compute_conductance(G, comms)
        print(f"Level {level}: {len(comms)} communities | Modularity: {mod:.4f} | Mean Conductance: {cond:.4f}")
    print()