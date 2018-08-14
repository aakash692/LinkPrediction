from ged4py.algorithm.abstract_graph_edit_dist import AbstractGraphEditDistance
from ged4py.algorithm.edge_edit_dist import EdgeEditDistance
from ged4py.graph.edge_graph import EdgeGraph
import sys


def compare(g1, g2, print_details=False, normalized=True):
    ged = GraphEditDistance(g1, g2)

    if print_details:
        ged.print_matrix()

    if normalized:
        return ged.normalized_distance()
    else:
        return ged.distance()


class GraphEditDistance(AbstractGraphEditDistance):

    def __init__(self, g1, g2):
        AbstractGraphEditDistance.__init__(self, g1, g2)

    def substitute_cost(self, node1, node2):
        return self.relabel_cost(node1, node2) + self.edge_diff(node1, node2)

    def relabel_cost(self, node1, node2):
        if node1 == node2:
            return 0.
        else:
            return 1.

    def delete_cost(self, i, j, nodes1):
        if i == j:
            return 1
        return sys.maxsize

    def insert_cost(self, i, j, nodes2):
        if i == j:
            return 1
        return sys.maxsize

    def pos_insdel_weight(self, node):
        return 1

    def edge_diff(self, node1, node2):
        edges1 = [edge[1] for edge in self.g1.edges(node1)]
        edges2 = [edge[1] for edge in self.g2.edges(node2)]
        if len(edges1) == 0 or len(edges2) == 0:
            return max(len(edges1), len(edges2))

        edit_edit_dist = EdgeEditDistance(EdgeGraph(node1,edges1), EdgeGraph(node2,edges2))
        return edit_edit_dist.normalized_distance()
