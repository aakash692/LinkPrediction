import numpy as np
import random
import networkx as nx
from gem.utils import evaluation_util
from networkx.algorithms import tree

def splitDiGraphToTrainTest1(di_graph, train_ratio = 0.5, is_undirected=True, file_name = None):
    
    train_digraph = di_graph.copy()
    test_digraph = di_graph.copy()
    node_num = di_graph.number_of_nodes()
    count = 0
    num_edges = di_graph.number_of_edges() * (1-train_ratio)
    edges = []
    
    for (st, ed, w) in di_graph.edges(data='weight', default=1):
        edges.append((st,ed,w))
    random.shuffle(edges)
    co = 0

    G_temp = train_digraph.to_undirected()
    con_comp = nx.number_connected_components(G_temp)

    mst = nx.algorithms.tree.mst.minimum_spanning_tree(G_temp, algorithm='kruskal')
    # print (mst.number_of_edges())
    # print (nx.number_connected_components(G_temp),nx.number_connected_components(mst))

    for (st, ed, w) in edges:
        # if(co%1000==0):
            # print (co, count)
        co+=1
        if(count>num_edges):
            break
        if(is_undirected and st>=ed):
            continue
        G_temp.remove_edge(st, ed)
        if nx.number_connected_components(G_temp)>con_comp:
            G_temp.add_edge(st,ed)
        else:
            train_digraph.remove_edge(st, ed)
            if(is_undirected):
                train_digraph.remove_edge(ed, st)
            count+=2
            
    
    for (st, ed, w) in train_digraph.edges(data='weight', default=1):
        test_digraph.remove_edge(st,ed)


    if (file_name):
        with open(file_name+"_train.pkl", 'w') as f:
            for (st, ed, w) in train_graph.edges(data='weight', default=1):
                f.write('%d %d %f\n' % (st, ed))

        with open(file_name+"_test", 'w') as f:
            for (st, ed, w) in test_graph.edges(data='weight', default=1):
                f.write('%d %d %f\n' % (st, ed))

    return (train_digraph, test_digraph)

def splitDiGraphToTrainTest2(di_graph, train_ratio = 0.5, is_undirected=True, file_name = None):
    
    train_digraph, test_digraph = evaluation_util.splitDiGraphToTrainTest(
        di_graph,
        train_ratio=train_ratio,
        is_undirected=is_undirected
    )
    if not nx.is_connected(train_digraph.to_undirected()):
        train_digraph = max(
            nx.weakly_connected_component_subgraphs(train_digraph),
            key=len
        )
        tdl_nodes = train_digraph.nodes()
        nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
        train_digraph=nx.relabel_nodes(train_digraph, nodeListMap, copy=True)
        test_digraph = test_digraph.subgraph(tdl_nodes)
        test_digraph=nx.relabel_nodes(test_digraph, nodeListMap, copy=True)

    if (file_name):
        with open(file_name+"_train", 'w') as f:
            for (st, ed, w) in train_graph.edges(data='weight', default=1):
                f.write('%d %d %f\n' % (st, ed))

        with open(file_name+"_test", 'w') as f:
            for (st, ed, w) in test_graph.edges(data='weight', default=1):
                f.write('%d %d %f\n' % (st, ed))

    return (train_digraph, test_digraph)

