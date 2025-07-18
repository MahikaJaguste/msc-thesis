
import networkx as nx
import random
import os

# Load original graph
GRAPH_NAME = "full"
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../data/{GRAPH_NAME}")
)
graph_path = os.path.join(DATA_PATH, "graph.gml")
G = nx.read_gml(graph_path, label='id')

# Set seed for reproducibility
random.seed(42)


# Remove 10% of nodes
nodes = list(G.nodes())
num_nodes_to_remove = int(0.1 * len(nodes))
nodes_to_remove = random.sample(nodes, num_nodes_to_remove)
G.remove_nodes_from(nodes_to_remove)

# Save perturbed graph
output_path = os.path.join(DATA_PATH, 'node', "graph.gml")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
nx.write_gml(G, output_path)
print(f"Saved: {output_path}")
