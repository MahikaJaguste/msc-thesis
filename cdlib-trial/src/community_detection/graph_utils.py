import networkx as nx
from cdlib.utils import nx_node_integer_mapping
from cdlib import NodeClustering

def build_graph_from_triplets(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        s, p, o = row['subject'], row['predicate'], row['object']
        G.add_edge(s, o, label=p)
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def remap_graph_for_algorithm(G):
    G_int, mapping = nx_node_integer_mapping(G)
    if mapping is None:
        return G_int, mapping, None
    rev_mapping = {v: k for k, v in mapping.items()}
    return G_int, mapping, rev_mapping


def remap_node_communities(community, mapping):
    """
    Remaps a NodeClustering object from integer node IDs to original labels.

    Args:
        community (NodeClustering): CDlib community object with integer node IDs.
        mapping (dict): Mapping from int -> original node labels.

    Returns:
        NodeClustering: New community object with string node labels.
    """
    new_communities = []
    for com in community.communities:
        new_com = [mapping[n] for n in com]
        new_communities.append(new_com)

    return NodeClustering(
        communities=new_communities,
        graph=None,  # not needed for evaluation
        method_name=community.method_name,
        overlap=community.overlap
    )