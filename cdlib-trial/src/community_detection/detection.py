from cdlib import algorithms, evaluation
# from cdlib.model import NodeClustering
# from cdlib import benchmark
# from cdlib.evaluation import newman_girvan_modularity, modularity_overlap, conductance
from cdlib.ensemble import grid_search, Parameter, BoolParameter
def get_algorithms():
    return {
        # 'label_propagation': (algorithms.label_propagation, {}),
        # 'leiden': (algorithms.leiden, {}),
        # 'louvain': (algorithms.louvain, {}),
        'slpa': (algorithms.slpa, {}),
        # 'lpanni': (algorithms.lpanni, {}),
        # 'kclique': (algorithms.kclique, [
        #     Parameter(name="k", start=3, end=4, step=1)
        # ]), # 0 communities found
        # 'congo': (algorithms.congo, [
        #     Parameter(name="number_communities", start=3, end=4, step=1),
        # ]), # did not work
        # 'aslpaw': (algorithms.aslpaw, []), # did not work
        # 'coach': (algorithms.coach, []), # 0 communities found
        # 'dcs': (algorithms.dcs, []), # did not work
        # 'dpclus': (algorithms.dpclus, []), # did not work
        # 'graph_entropy': (algorithms.graph_entropy, []),
    }

def run_grid_search(G, method, param_grid):
    best_community = grid_search(G, method, param_grid, quality_score=evaluation.newman_girvan_modularity, aggregate=max)[0]
    return best_community
