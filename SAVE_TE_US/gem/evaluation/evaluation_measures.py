import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from gem.utils import train_test_split, graph_util, evaluation_util
from gem.evaluation import metrics as scores
import pickle

# heuristic socring functions 

def cn(G, i, j):
    return len(sorted(nx.common_neighbors(G, i, j)))

def jc(G, i, j):
    return sorted(nx.jaccard_coefficient(G, [(i, j)]))[0][2]

def pa(G, i, j):
    return sorted(nx.preferential_attachment(G, [(i, j)]))[0][2]

def aa(G, i, j):
    return sorted(nx.adamic_adar_index(G, [(i, j)]))[0][2]

def rci(G, i, j):
    return sorted(nx.resource_allocation_index(G, [(i, j)]))[0][2]

def nd(G, i):
    return G.degree(i)

def allh(G, i, j):
    feat = np.zeros(7)
    feat[0] = cn(G, i ,j)
    feat[1] = jc(G, i ,j)
    feat[2] = aa(G, i ,j)
    feat[3] = pa(G, i ,j)
    feat[4] = rci(G, i ,j)
    feat[5] = nd(G, i)
    feat[6] = nd(G, j)
    return feat

# fuctions for combining vector of nodes to produce vector for edge

def concat(vi, vj):
    return np.asarray(vi.tolist()+vj.tolist())

def dotp(vi, vj):
    prod=np.dot(vi, vj)/(np.linalg.norm(vi)*np.linalg.norm(vj))
    prod1=np.zeros(2)
    prod1[0]=prod
    return prod1

def average(vi, vj):
    return (vi+vj)/2.0

def hadamard1(vi, vj):
    prod=np.multiply(vi, vj)
    return prod

def hadamard2(vi, vj):
    prod=np.multiply(vi, vj)/(np.linalg.norm(vi)*np.linalg.norm(vj))
    return prod

def hadamard3(vi, vj):
    di=len(vi)//2
    prod=np.multiply(vi[:di], vj[di:])
    return prod    

def wL1(vi, vj):
    return abs(vi-vj)

def wL2(vi, vj):
    return np.square(vi-vj)

# create dataset for training (equal positive and negatice samples)

def create_edge_dataset(train_digraph, test_digraph, is_undirected=True):

    node_num = train_digraph.number_of_nodes()
    p_samples=[]
    n_samples=[]

    for (i,j) in test_digraph.edges():
        if(is_undirected and i>=j):
            continue
        p_samples.append((i,j))

    random.shuffle(p_samples)
    co = len(p_samples)

    s1= np.random.choice(node_num*node_num, 4*len(p_samples))

    for e in s1:
        if(co==0):
            break
        i = e%node_num
        j = int(e/node_num)
        if(is_undirected and i>=j):
            continue
        if(train_digraph.has_edge(i,j)):
            continue
        if(test_digraph.has_edge(i,j)):
            continue
        n_samples.append((i,j))
        co=co-1

    random.shuffle(n_samples)
    return p_samples, n_samples

def create_vector_dataset(p_edge, n_edge, eval_method, X):

    p_samples= []
    n_samples= []

    for i,j in p_edge:
        p_samples.append(eval_method(X[i],X[j]))
    for i,j in n_edge:
        n_samples.append(eval_method(X[i],X[j]))

    train_data = np.asarray(p_samples+n_samples)
    train_label1 = np.full(len(p_samples),1)
    train_label0 = np.full(len(n_samples),0)
    train_label = np.concatenate((train_label1,train_label0))
    
    return train_data, train_label

def create_score_dataset(p_edge, n_edge, eval_method, train_digraph):

    p_samples= []
    n_samples= []
    G=train_digraph.to_undirected()
    for i,j in p_edge:
        p_samples.append(eval_method(G,i,j))
    for i,j in n_edge:
        n_samples.append(eval_method(G,i,j))

    train_data = np.asarray(p_samples+n_samples)
    train_label1 = np.full(len(p_samples),1)
    train_label0 = np.full(len(n_samples),0)
    train_label = np.concatenate((train_label1,train_label0))
    
    return train_data, train_label

def normalise_train_test(train_data, test_data):

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data-mean)/std
    test_data = (test_data-mean)/std
    
    return train_data, test_data

def train_classifier(train_data, train_label):

    logistic = LogisticRegression()
    logistic.fit(train_data,train_label)
    return logistic

def visualise_data(positive_data, negative_data):

    # pca = TSNE(n_components = 2)
    # plt_data1 = pca.fit_transform(np.concatenate(positive_data,negative_data))
    plt_data1=np.concatenate((positive_data,negative_data))
    # print ("PCA complete")
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plt_data1[:len(positive_data),0], plt_data1[:len(positive_data),0], 'ro', alpha=0.1)
    plt.subplot(212)
    plt.plot(plt_data1[len(positive_data):,0], plt_data1[len(positive_data):,0], 'bx', alpha=0.1)
    plt.show()

def sample_edge_new(train_digraph, test_digraph, ratio, num_edges=500000, is_undirected=True):

    node_num = train_digraph.number_of_nodes()
    total_edges = node_num*node_num - train_digraph.number_of_edges()
    positive_edges = ratio * num_edges
    if(ratio==-1):
        positive_edges = 10 + (train_digraph.number_of_edges() * num_edges)/(node_num * node_num)
    positive_edges = int(positive_edges)
    co = num_edges - positive_edges
    s1= np.random.choice(node_num*node_num, 4*num_edges)
    p_samples = []
    n_samples = []

    for (i,j) in test_digraph.edges():
        if(is_undirected and i>=j):
            continue
        p_samples.append((i,j))
    random.shuffle(p_samples)

    if(len(p_samples)<positive_edges):
        positive_edges=len(p_samples)
        co=positive_edges

    for e in s1:
        if(co==0):
            break
        i = e%node_num
        j = int(e/node_num)
        if(is_undirected and i>=j):
            continue
        if(train_digraph.has_edge(i,j)):
            continue
        if(test_digraph.has_edge(i,j)):
            continue
        n_samples.append((i,j))
        co=co-1

    random.shuffle(n_samples)
    print (positive_edges, len(n_samples))
    return p_samples[:positive_edges]+n_samples

def getscore1(train_digraph, node_l, eval_method):

    G = train_digraph.to_undirected()
    node_num = len(node_l)
    estimated_adj = np.zeros((node_num,node_num))
    for i in xrange(node_num):
        for j in xrange(node_num):
            if(i==j):
                continue
            estimated_adj[i][j] = eval_method(G, node_l[i], node_l[j])
    return estimated_adj

def getscore2(train_digraph, node_l, clasifier, eval_method, X, mean, std):
    
    node_num = len(node_l)
    estimated_adj = np.zeros((node_num,node_num))
    for i in xrange(node_num):
        for j in xrange(node_num):
            if(i==j):
                continue
            estimated_adj[i][j] = clasifier.decision_function([(eval_method(X[i],X[j])-mean)/std])[0]

    return estimated_adj

def getscore3(train_digraph, sample_edges, eval_method):

    G = train_digraph.to_undirected()
    score_list = []
    for (st,ed) in sample_edges:
        score_list.append((st,ed,eval_method(G,st,ed)))

    return score_list

def getscore4(train_digraph, graph_embedding, sample_edges):

    score_list = []
    for (st,ed) in sample_edges:
        score_list.append((st,ed,graph_embedding.get_edge_weight(st,ed)))

    return score_list

def getscore5(train_digraph, sample_edges, clasifier, eval_method, X, mean, std):
    score_list = []
    for (st,ed) in sample_edges:
        score_list.append((st,ed,clasifier.decision_function([(eval_method(X[st],X[ed])-mean)/std])[0]))

    return score_list

def getscore6(train_digraph, node_l, clasifier, eval_method, mean, std):
    
    node_num = len(node_l)
    estimated_adj = np.zeros((node_num,node_num))
    G = train_digraph.to_undirected()
    for i in xrange(node_num):
        for j in xrange(node_num):
            if(i==j):
                continue
            estimated_adj[i][j] = clasifier.decision_function([(eval_method(G, node_l[i], node_l[j])-mean)/std])[0]

    return estimated_adj

def getscore7(train_digraph, sample_edges, clasifier, eval_method, mean, std):

    score_list = []
    G = train_digraph.to_undirected()
    for (st,ed) in sample_edges:
        score_list.append((st,ed,clasifier.decision_function([(eval_method(G,st,ed)-mean)/std])[0]))

    return score_list

def check_samples(train_digraph, test_digraph, is_undirected=True):

    for (st,ed) in train_digraph.edges():
        if(test_digraph.has_edge(st,ed)):
            test_digraph.remove_edge(st,ed)

    print (train_digraph.number_of_nodes(), train_digraph.number_of_edges(), test_digraph.number_of_edges())
    sample_edges = sample_edge_new(train_digraph,test_digraph)
    test_digraph1, node_l = graph_util.sample_graph(test_digraph, 1024)

    G = train_digraph.to_undirected()
    mydata1=np.zeros(11)
    mydata2=np.zeros(11)
    
    for (st1,ed1) in test_digraph1.edges():
        st=node_l[st1]
        ed=node_l[ed1]

        if(is_undirected and st>=ed):
            continue
        a = nx.shortest_path_length(G,source=st,target=ed)
        if a<10:
            if(test_digraph.has_edge(st,ed)):
                mydata1[a]=mydata1[a]+1
            else:
                mydata2[a]=mydata2[a]+1
        else:
            if(test_digraph.has_edge(st,ed)):
                mydata1[10]=mydata1[10]+1
            else:
                mydata2[10]=mydata2[10]+1
        
    print (mydata1,mydata2)

    mydata1=np.zeros(11)
    mydata2=np.zeros(11)
    G = train_digraph.to_undirected()
    co=0
    for (st,ed) in sample_edges:
        a = nx.shortest_path_length(G,source=st,target=ed)
        if a<10:
            if(test_digraph.has_edge(st,ed)):
                mydata1[a]=mydata1[a]+1
            else:
                mydata2[a]=mydata2[a]+1
        else:
            if(test_digraph.has_edge(st,ed)):
                mydata1[10]=mydata1[10]+1
            else:
                mydata2[10]=mydata2[10]+1
        
    print (mydata1,mydata2)

    mydata1=np.zeros(11)
    mydata2=np.zeros(11)

    for (st,ed) in test_digraph.edges():
        if(is_undirected and st>=ed):
            continue
        a = nx.shortest_path_length(G,source=st,target=ed)
        if a<10:
            if(test_digraph.has_edge(st,ed)):
                mydata1[a]=mydata1[a]+1
            else:
                mydata2[a]=mydata2[a]+1
        else:
            if(test_digraph.has_edge(st,ed)):
                mydata1[10]=mydata1[10]+1
            else:
                mydata2[10]=mydata2[10]+1
        
    print (mydata1,mydata2)
    return mydata1,mydata2


def evaluate_unsupervised_all(di_graph, is_undirected=True):

    train_digraph, test_digraph = train_test_split.splitDiGraphToTrainTest2(di_graph, train_ratio = 0.8, is_undirected=True)
    sample_edges = sample_edge_new(train_digraph,test_digraph)
    test_digraph1, node_l = graph_util.sample_graph(test_digraph, 1024)
    AP=[];ROC=[];MAP=[]
    heurestics = [cn,jc,pa,aa]

    for x in heurestics:

        estimated_adj = getscore1(train_digraph, node_l, x)
        predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
        filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
        MAP1 = scores.computeMAP(filtered_edge_list, test_digraph1)
        MAP.append(MAP1)
        
        filtered_edge_list = getscore3(train_digraph, sample_edges, x)
        AP1, ROC1 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
        AP.append(AP1);ROC.append(ROC1)

        print (AP1,ROC1,MAP1)

    return AP, ROC, MAP

def evaluate_supervised(di_graph, graph_embedding, is_undirected=True):

    train_digraph, test_digraph = train_test_split.splitDiGraphToTrainTest2(di_graph, train_ratio = 0.6, is_undirected=True)
    train_digraph1, test_digraph = evaluation_util.splitDiGraphToTrainTest(
        test_digraph,
        train_ratio=0.5,
        is_undirected=is_undirected
    )
    
    X, _ = graph_embedding.learn_embedding(graph=train_digraph, no_python=False)

    trp, trn = create_edge_dataset(train_digraph, train_digraph1)
    trd, trl = create_vector_dataset(trp, trn, hadamard2, X)
    mean=np.mean(trd,axis=0)
    std=np.std(trd,axis=0)
    trd=(trd-mean)/std

    clasifier = train_classifier(trd, trl)

    for (st,ed) in train_digraph1.edges():
        train_digraph.add_edge(st,ed)

    sample_edges = sample_edge_new(train_digraph,test_digraph,0.5)
    
    X, _ = graph_embedding.learn_embedding(graph=train_digraph, no_python=False)
    
    filtered_edge_list = getscore5(train_digraph, sample_edges, clasifier, hadamard2, X, mean, std)    
    AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    
    test_digraph, node_l = graph_util.sample_graph(test_digraph, 1024)
    X = X[node_l]
    estimated_adj = getscore2(train_digraph, node_l, clasifier, hadamard2, X, mean, std)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    MAP = scores.computeMAP(filtered_edge_list, test_digraph)

    print (MAP)
    
    return AP, ROC, MAP

def evaluate_unsupervised_embedding(di_graph, graph_embedding, is_undirected=True):

    train_digraph, test_digraph = train_test_split.splitDiGraphToTrainTest2(di_graph, train_ratio = 0.8, is_undirected=True)    
    
    X, _ = graph_embedding.learn_embedding(graph=train_digraph, no_python=False)
    
    sample_edges = sample_edge_new(train_digraph,test_digraph,0.5)
    filtered_edge_list = getscore4(train_digraph, graph_embedding, sample_edges)
    AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)

    test_digraph1, node_l = graph_util.sample_graph(test_digraph, 1024)
    X = X[node_l]
    estimated_adj = graph_embedding.get_reconstructed_adj(X, node_l)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
    filtered_edge_list = [e for e in predicted_edge_list if not (train_digraph.has_edge(node_l[e[0]], node_l[e[1]]))] 
    MAP = scores.computeMAP(filtered_edge_list, test_digraph1)

    print (AP,ROC,MAP)

    return AP, ROC, MAP

def calc_map_us(embedding, X, node_l, train_digraph, test_digraph1):

    estimated_adj = embedding.get_reconstructed_adj(X, node_l)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
    filtered_edge_list = [e for e in predicted_edge_list if not (train_digraph.has_edge(node_l[e[0]], node_l[e[1]]))] 
    MAP = scores.computeMAP(filtered_edge_list, test_digraph1)

    print (MAP)
    return MAP

def calc_map_heu(node_l, train_digraph, test_digraph1):

    MAP = []
    heurestics = [cn,jc,pa,aa]
    for x in heurestics:
        estimated_adj = getscore1(train_digraph, node_l, x)
        predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
        filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
        MAP1 = scores.computeMAP(filtered_edge_list, test_digraph1)
        MAP.append(MAP1)

    print (MAP)
    return MAP
    
def calc_aproc_us(embedding, X, train_digraph, test_digraph, sample_edges):

    filtered_edge_list = getscore4(train_digraph, embedding, sample_edges)
    AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)
 
    print (AP,ROC)
    return AP,ROC

def calc_aproc_heu(train_digraph, test_digraph, sample_edges):

    AP=[];ROC=[]
    heurestics = [cn,jc,pa,aa]

    for x in heurestics:

        filtered_edge_list = getscore3(train_digraph, sample_edges, x)
        AP1, ROC1 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
        AP.append(AP1); ROC.append(ROC1)

    print (AP,ROC)
    return AP,ROC

def calc_map_heu_s(node_l, train_digraph, train_digraph1, test_digraph1, trp, trn):

    trd, trl = create_score_dataset(trp, trn, allh, train_digraph)
    mean=np.mean(trd,axis=0)
    std=np.std(trd,axis=0)
    trd=(trd-mean)/std    
    clasifier = train_classifier(trd, trl)
    
    estimated_adj = getscore6(train_digraph, node_l, clasifier, allh, mean, std)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)

    for (st,ed) in train_digraph1.edges():
        train_digraph.add_edge(st,ed)
    
    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    MAP1 = scores.computeMAP(filtered_edge_list, test_digraph1)

    estimated_adj = getscore6(train_digraph, node_l, clasifier, allh, mean, std)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    MAP2 = scores.computeMAP(filtered_edge_list, test_digraph1)    

    print (MAP1,MAP2)
    return MAP1,MAP2

def calc_map_s(embedding, X1, X2, train_digraph, train_digraph1, node_l, test_digraph1, trp, trn, had):

    if(had==1):
        func=hadamard1
    elif(had==0):
        func=hadamard2
    elif(had==-1):
        func=hadamard3

    trd, trl = create_vector_dataset(trp, trn, func, X1)
    mean=np.mean(trd,axis=0)
    std=np.std(trd,axis=0)
    trd=(trd-mean)/std
    clasifier = train_classifier(trd, trl)
    
    X1=X1[node_l]
    estimated_adj = getscore2(train_digraph, node_l, clasifier, func, X1, mean, std)

    for (st,ed) in train_digraph1.edges():
        train_digraph.add_edge(st,ed)
    
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    MAP1 = scores.computeMAP(filtered_edge_list, test_digraph1)
    
    X2=X2[node_l]
    estimated_adj = getscore2(train_digraph, node_l, clasifier, func, X2, mean, std)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    MAP2 = scores.computeMAP(filtered_edge_list, test_digraph1)

    print (MAP1, MAP2)
    return MAP1, MAP2

def calc_aproc_heu_s(train_digraph, train_digraph1, test_digraph, trp, trn, sample_edges):

    trd, trl = create_score_dataset(trp, trn, allh, train_digraph)
    mean=np.mean(trd,axis=0)
    std=np.std(trd,axis=0)
    trd=(trd-mean)/std
    clasifier = train_classifier(trd, trl)

    filtered_edge_list = getscore7(train_digraph, sample_edges, clasifier, allh, mean, std)
    AP1, ROC1 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    print (AP1,ROC1)

    for (st,ed) in train_digraph1.edges():
        train_digraph.add_edge(st,ed)
    
    filtered_edge_list = getscore7(train_digraph, sample_edges, clasifier, allh, mean, std)
    AP2, ROC2 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    print (AP2,ROC2)

    return AP1,AP2,ROC1,ROC2

def calc_aproc_s(embedding, X1, X2, train_digraph, train_digraph1, test_digraph, sample_edges, trp, trn, had):

    if(had==1):
        func=hadamard1
    elif(had==0):
        func=hadamard2
    elif(had==-1):
        func=hadamard3
    trd, trl = create_vector_dataset(trp, trn, func, X1)
    mean=np.mean(trd,axis=0)
    std=np.std(trd,axis=0)
    trd=(trd-mean)/std
    clasifier = train_classifier(trd, trl)

    filtered_edge_list = getscore5(train_digraph, sample_edges, clasifier, func, X1, mean, std)    
    AP1, ROC1 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    print (AP1,ROC1)
    for (st,ed) in train_digraph1.edges():
        train_digraph.add_edge(st,ed)
    
    filtered_edge_list = getscore5(train_digraph, sample_edges, clasifier, func, X2, mean, std)    
    AP2, ROC2 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    
    print (AP2,ROC2)
    return AP1,AP2,ROC1,ROC2