from scipy.optimize import linear_sum_assignment
import sys
import numpy as np


class AbstractGraphEditDistance(object):
    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2

    def normalized_distance(self):
        """
        Returns the graph edit distance between graph g1 & g2
        The distance is normalized on the size of the two graphs.
        This is done to avoid favorisation towards smaller graphs
        """
        avg_graphlen = (len(self.g1) + len(self.g2)) / 2
        return self.distance() / avg_graphlen

    def distance(self):
        return sum(self.edit_costs())

    def edit_costs(self):
        cost_matrix = self.create_cost_matrix()
        row_ind,col_ind = linear_sum_assignment(cost_matrix)
        return [cost_matrix[row_ind[i]][col_ind[i]] for i in range(len(row_ind))]

    def create_cost_matrix(self):
        """
        Creates a |N+M| X |N+M| cost matrix between all nodes in
        graphs g1 and g2
        Each cost represents the cost of substituting,
        deleting or inserting a node
        The cost matrix consists of four regions:

        substitute 	| insert costs
        -------------------------------
        delete 		| delete -> delete

        The delete -> delete region is filled with zeros
        """
        n = len(self.g1)
        m = len(self.g2)
        cost_matrix = np.zeros((n+m, n+m))
        nodes1 = self.g1.nodes()
        nodes2 = self.g2.nodes()

        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                cost_matrix[i,j] = self.substitute_cost(node1, node2)

        for i in range(m):
            for j in range(m):
                cost_matrix[i+n,j] = self.insert_cost(i, j, nodes2)

        for i in range(n):
            for j in range(n):
                cost_matrix[j,i+m] = self.delete_cost(i, j, nodes1)

        self.cost_matrix = cost_matrix
        return cost_matrix

    def insert_cost(self, i, j, nodes1):
        raise NotImplementedError

    def delete_cost(self, i, j, nodes2):
        raise NotImplementedError

    def substitute_cost(self, nodes1, nodes2):
        raise NotImplementedError

    def print_matrix(self):
        print("cost matrix:")
        cost_mat = self.create_cost_matrix()
        np.place(cost_mat, cost_mat==sys.maxsize, np.inf)
        print(cost_mat)
