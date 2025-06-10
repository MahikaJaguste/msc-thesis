from cdlib import evaluation

def leiden_modularity(G, communities):
    """
    Compute modularity for Leiden communities.
    """
    return evaluation.modularity(G, communities).score

def print_community_stats(communities, id_to_node):
    """
    Print number of communities and their sizes.
    """
    print(f"Number of communities: {len(communities.communities)}")
    for i, comm in enumerate(communities.communities):
        print(f"Community {i}: size={len(comm)} | nodes={[id_to_node[n] for n in comm[:5]]} ...")