def print_overlapping_node_summary(communities, G, nodeid_to_patientid=None):
    """
    Prints the percentage of overlapping nodes, lists all overlapping node ids (and patient ids if mapping provided),
    and their respective multiple community memberships.
    communities: list of lists of node ids (overlapping allowed)
    G: networkx graph (for total node count)
    nodeid_to_patientid: dict mapping node id to patient id (optional)
    """
    from collections import defaultdict
    node_to_comms = defaultdict(list)
    for comm_id, comm in enumerate(communities):
        for node in comm:
            node_to_comms[node].append(comm_id)
    overlapping_nodes = {node: comms for node, comms in node_to_comms.items() if len(comms) > 1}
    percent = 100 * len(overlapping_nodes) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    print(f"Percentage of overlapping nodes: {percent:.2f}% ({len(overlapping_nodes)}/{G.number_of_nodes()})")
    if overlapping_nodes:
        if len(overlapping_nodes) < 10:
            print("Overlapping nodes and their community memberships:")
            for node, comms in overlapping_nodes.items():
                if nodeid_to_patientid:
                    pid = nodeid_to_patientid.get(node, node)
                    print(f"  Node {node} (PatientId: {pid}): Communities {comms}")
                else:
                    print(f"  Node {node}: Communities {comms}")
    else:
        print("No overlapping nodes found.")
