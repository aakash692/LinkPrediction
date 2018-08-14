disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from time import time

import sys
sys.path.append('./')
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from subprocess import call

from .static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz


class verse(StaticGraphEmbedding):

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the verse class

        Args:
            d: dimension of the embedding
            alpha: PPR
            threads:
            nsamples:
        '''
        hyper_params = {
            'method_name': 'verse_rw'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):
        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if edge_f:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
        graph_util.write_edgelist(graph, 'tempGraph_verse.graph')
        try:
            os.system("python gem/verse-master/python/convert.py tempGraph_verse.graph outgraph_verse.bcsr")
        except Exception as e:
            print (str(e))
        args = "gem/verse-master/src/verse -input outgraph_verse.bcsr -output tempGraph_verse.emb -dim " + str(self._d) + " -alpha " + str(self._alpha) + " -threads " + str(self._threads) + " -nsamples " + str(self._nsamples)
        t1 = time()
        try:
            os.system(args)
        except Exception as e:
            print(str(e))
            raise Exception('./verse not found. Please compile, place verse in the path and grant executable permission')
        self._X = np.fromfile('tempGraph_verse.emb', np.float32).reshape(graph.number_of_nodes(), self._d)
        t2 = time()
        return self._X, (t2 - t1)

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :], self._X[j, :])/((np.linalg.norm(self._X[i, :])*np.linalg.norm(self._X[j, :])))
    
    # def get_edge_weight(self, i, j):
    #     return np.dot(self._X[i, :], self._X[j, :])

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        # adj_mtx_r1=np.matmul(X,np.transpose(X))
        # for it in xrange(node_num):
        #     adj_mtx_r1[it][it]=0.0
        # print (np.nonzero(abs(adj_mtx_r-adj_mtx_r1)>1e-6))
        return adj_mtx_r


if __name__ == '__main__' and __package__ is None:
    # load Zachary's Karate graph
    edge_f = 'data/karate.edgelist'
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
    G = G.to_directed()
    res_pre = 'results/testKarate'
    graph_util.print_graph_stats(G)
    t1 = time()
    embedding = verse(2, 1, 80, 10, 10, 1, 1)
    embedding.learn_embedding(graph=G, edge_f=None,
                              is_weighted=True, no_python=True)
    print('verse:\n\tTraining time: %f' % (time() - t1))

    viz.plot_embedding2D(embedding.get_embedding(),
                         di_graph=G, node_colors=None)
    plt.show()
