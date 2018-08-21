import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn import tree
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
from sklearn import tree
import graphviz 

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

def concat(vi, vj):
    return np.asarray(vi.tolist()+vj.tolist())

def dotp1(vi, vj):
    prod=np.dot(vi, vj)
    prod1=np.zeros(1)
    prod1[0]=prod
    return prod1

def dotp2(vi, vj):
    prod=np.dot(vi, vj)/(np.linalg.norm(vi)*np.linalg.norm(vj))
    prod1=np.zeros(1)
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

def wL1(vi, vj):
    return abs(vi-vj)

def wL2(vi, vj):
    return np.square(vi-vj)

def create_edge_dataset(train_digraph, test_digraph, ty, is_undirected=True):

    node_num = train_digraph.number_of_nodes()
    p_samples=[]
    n_samples=[]
    G=train_digraph.to_undirected()
    for (i,j) in test_digraph.edges():
        if (ty==1):
            if(nx.shortest_path_length(G,source=i,target=j)==2):
                continue
        elif (ty==0):
            if(not(nx.shortest_path_length(G,source=i,target=j)==2)):
                continue
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
        if (ty==1):
            if(nx.shortest_path_length(G,source=i,target=j)==2):
                continue
        elif (ty==0):
            if(not(nx.shortest_path_length(G,source=i,target=j)==2)):
                continue
        if(is_undirected and i>=j):
            continue
        if(train_digraph.has_edge(i,j)):
            continue
        if(test_digraph.has_edge(i,j)):
            continue
        n_samples.append((i,j))
        co=co-1

    if(co!=0):
        p_samples=p_samples[:len(n_samples)]

    random.shuffle(n_samples)
    # p_samples=p_samples[:len(p_samples)//10]
    print (len(p_samples),len(n_samples))
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

def visualise_data(positive_data, negative_data):

    plt_data1=np.concatenate((positive_data,negative_data))
    # mean=np.mean(plt_data1,axis=0)
    # std=np.std(plt_data1,axis=0)
    # plt_data1=(plt_data1-mean)/std
    # pca = TSNE(n_components = 2)
    # plt_data1 = pca.fit_transform(plt_data1)
    # print ("PCA complete")
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plt_data1[:len(positive_data),0], plt_data1[:len(positive_data),1], 'ro', alpha=0.1)
    plt.subplot(212)
    plt.plot(plt_data1[len(positive_data):,0], plt_data1[len(positive_data):,1], 'bx', alpha=0.1)
    plt.show()

def func_new(train_digraph,G,i,j):

    feat = np.zeros(7)
    feat[0] = aa(G, i ,j)
    feat[1] = pa(G, i ,j)
    feat[2] = cn(G, i ,j)
    feat[3] = jc(G, i ,j)
    feat[4] = rci(G, i ,j)
    feat[5] = nd(G, i)
    feat[6] = nd(G, j)
    return feat

def func_new2(train_digraph,G,i,j):

    feat = np.zeros(3)
    feat[0] = pa(G, i ,j)
    feat[1] = nd(G, i)
    feat[2] = nd(G, j)
    return feat

def func_new1(train_digraph,G,i,j):

    feat = np.zeros(1)
    # feat[0] = aa(G, i ,j)
    # feat[1] = pa(G, i ,j)
    # feat[2] = cn(G, i ,j)
    # feat[3] = jc(G, i ,j)
    # feat[4] = rci(G, i ,j)
    # feat[5] = nd(G, i)
    # feat[6] = nd(G, j)
    return feat

def create_mix_dataset(p_edge, n_edge, train_digraph, embeddings, combine, meth):

    p_samples= []
    n_samples= []
    G=train_digraph.to_undirected()

    for i,j in p_edge:
        feat = meth(train_digraph,G,i,j)
        for it in xrange(len(embeddings)):
            emb=embeddings[it]
            func=combine[it]
            feat=np.concatenate((feat,func(emb[i],emb[j])))
            
        p_samples.append(feat)
    
    for i,j in n_edge:
        feat = meth(train_digraph,G,i,j)
        for it in xrange(len(embeddings)):
            emb=embeddings[it]
            func=combine[it]
            feat=np.concatenate((feat,func(emb[i],emb[j])))
            
        n_samples.append(feat)

    # visualise_data(np.asarray(p_samples[:5000]),np.asarray(n_samples[:5000]))

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
    # logistic = tree.DecisionTreeClassifier(max_leaf_nodes=5)
    logistic.fit(train_data,train_label)
    pred = logistic.decision_function(train_data)
    print (average_precision_score(train_label,pred))
    return logistic

def sample_edge(train_digraph, num_edges=100000, is_undirected=True):

    node_num = train_digraph.number_of_nodes()
    s1 = np.arange(node_num*node_num)
    np.random.shuffle(s1)
    samples = []
    co = num_edges
    for e in s1:
        if(co==0):
            break
        i = e%node_num
        j = int(e/node_num)
        if(is_undirected and i>=j):
            continue
        if(train_digraph.has_edge(i,j)):
            continue
        samples.append((i,j))
        co=co-1
    return samples

def sample_edge_new(train_digraph, test_digraph, ratio, num_edges=500000, is_undirected=True):

    node_num = train_digraph.number_of_nodes()
    total_edges = node_num*node_num - train_digraph.number_of_edges()
    positive_edges = ratio * num_edges
    if(ratio==-1):
        positive_edges = 10 + (train_digraph.number_of_edges() * num_edges)/(node_num * node_num)
    positive_edges = int(positive_edges)
    co = num_edges - positive_edges
    s1= np.random.choice(node_num*node_num, 5*num_edges)
    p_samples = []
    n_samples = []
    G=train_digraph.to_undirected()
    for (i,j) in test_digraph.edges():
        # if not(nx.shortest_path_length(G,source=i,target=j)==2):
        #     continue
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
        # if not(nx.shortest_path_length(G,source=i,target=j)==2):
        #     continue
        if(is_undirected and i>=j):
            continue
        if(train_digraph.has_edge(i,j)):
            continue
        if(test_digraph.has_edge(i,j)):
            continue
        n_samples.append((i,j))
        co=co-1

    if(co!=0):
        positive_edges=int(len(n_samples)*(ratio/(1-ratio)))

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

def getscore2(train_digraph, node_l, clasifier, eval_method, X):
    
    node_num = len(node_l)
    estimated_adj = np.zeros((node_num,node_num))
    for i in xrange(node_num):
        for j in xrange(node_num):
            if(i==j):
                continue
            estimated_adj[i][j] = clasifier.decision_function([eval_method(X[i],X[j])])[0]

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

def getscore5(train_digraph, sample_edges, clasifier, eval_method, X):
    score_list = []
    for (st,ed) in sample_edges:
        score_list.append((st,ed,clasifier.decision_function([eval_method(X[st],X[ed])])[0]))

    return score_list

def getscore6(train_digraph, node_l, clasifier, eval_method):
    
    node_num = len(node_l)
    estimated_adj = np.zeros((node_num,node_num))
    G = train_digraph.to_undirected()
    for i in xrange(node_num):
        for j in xrange(node_num):
            if(i==j):
                continue
            estimated_adj[i][j] = clasifier.decision_function([eval_method(G, node_l[i], node_l[j])])[0]

    return estimated_adj

def getscore7(train_digraph, sample_edges, clasifier, eval_method, mean, std):

    score_list = []
    G = train_digraph.to_undirected()
    for (st,ed) in sample_edges:
        score_list.append((st,ed,clasifier.decision_function([(eval_method(G,st,ed)-mean)/std])[0]))

    return score_list

def getscore8(train_digraph, node_l, clasifier, embeddings, combine, meth):
    
    node_num = len(node_l)
    estimated_adj = np.zeros((node_num,node_num))
    G=train_digraph.to_undirected()

    for i in xrange(node_num):
        for j in xrange(node_num):
            if(i==j):
                continue
            feat = meth(train_digraph,G,node_l[i],node_l[j])
            for it in xrange(len(embeddings)):
                emb=embeddings[it]
                func=combine[it]
                feat=np.concatenate((feat,func(emb[i],emb[j])))
            estimated_adj[i][j] = clasifier.decision_function([feat])[0]

    return estimated_adj

def getscore9(train_digraph, sample_edges, clasifier, embeddings, combine, meth, mean, std):

    score_list = []
    G=train_digraph.to_undirected()
    for (i,j) in sample_edges:
        feat = meth(train_digraph,G,i,j)
        for it in xrange(len(embeddings)):
            emb=embeddings[it]
            func=combine[it]
            # if(feat[1]==0):
                # feat=np.concatenate((feat,np.full((1),0.000001)))
            feat=np.concatenate((feat,func(emb[i],emb[j])))
            # else:
                # feat=np.concatenate((feat,func(emb[i],emb[j])))
                # feat=np.concatenate((feat,np.full((1),0.000001)))
        score_list.append((i,j,clasifier.decision_function([(feat-mean)/std])[0]))

    return score_list

# def evaluate_unsupervised(di_graph, eval_method, is_undirected=True):

#     train_digraph, test_digraph = train_test_split.splitDiGraphToTrainTest2(di_graph, train_ratio = 0.8, is_undirected=True)
#     sample_edges = sample_edge(train_digraph)

#     estimated_adj = getscore1(train_digraph, node_l, eval_method)
#     predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
#     filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
#     MAP = scores.computeMAP(filtered_edge_list, test_digraph)
#     AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    
#     return AP, ROC, MAP

def check_samples(train_digraph, test_digraph, is_undirected=True):

    for (st,ed) in train_digraph.edges():
        if(test_digraph.has_edge(st,ed)):
            test_digraph.remove_edge(st,ed)

    print (train_digraph.number_of_nodes(), train_digraph.number_of_edges(), test_digraph.number_of_edges())
    sample_edges = sample_edge_new(train_digraph,test_digraph,-1)
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
    # train_digraph, test_digraph = train_test_split.splitDiGraphToTrainTest2(di_graph, train_ratio = 0.75, is_undirected=True)
    # train_digraph1, test_digraph = evaluation_util.splitDiGraphToTrainTest(
    #     test_digraph,
    #     train_ratio=0.2,
    #     is_undirected=is_undirected
    # )

    # train_digraph_temp=train_digraph.copy()
    # for (st,ed) in train_digraph1.edges():
    #     train_digraph_temp.add_edge(st,ed)

    # sample_edges = sample_edge_new(train_digraph_temp,test_digraph, -1)
    
    sample_edges = sample_edge_new(train_digraph,test_digraph, -1)

    filtered_edge_list = getscore3(train_digraph, sample_edges, aa)
    AP1, ROC1 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    print (AP1,ROC1)
    return AP1,ROC1

    # test_digraph1, node_l = graph_util.sample_graph(test_digraph, 1024)
    # AP=[];ROC=[];MAP=[]
    # # heurestics = [cn,jc,pa,aa]
    # heurestics = [aa]

    # for x in heurestics:

    #     estimated_adj = getscore1(train_digraph, node_l, x)
    #     predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
    #     filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    #     MAP1 = scores.computeMAP(filtered_edge_list, test_digraph1)
    #     MAP.append(MAP1)
        
    #     filtered_edge_list = getscore3(train_digraph, sample_edges, x)
    #     AP1, ROC1 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    #     AP.append(AP1);ROC.append(ROC1)

    #     print (AP1,ROC1,MAP1)

    # return AP, ROC, MAP

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
    clasifier = train_classifier(trd, trl)

    for (st,ed) in train_digraph1.edges():
        train_digraph.add_edge(st,ed)

    sample_edges = sample_edge_new(train_digraph,test_digraph,0.5)
    
    X, _ = graph_embedding.learn_embedding(graph=train_digraph, no_python=False)
    
    filtered_edge_list = getscore5(train_digraph, sample_edges, clasifier, hadamard2, X)    
    AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    
    test_digraph, node_l = graph_util.sample_graph(test_digraph, 1024)
    X = X[node_l]
    estimated_adj = getscore2(train_digraph, node_l, clasifier, hadamard2, X)
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

def evaluate_te(train_digraph, test_digraph, graph_embedding, verbose=False, is_undirected=True):

    # for (st,ed) in train_digraph.edges():
    #     if(test_digraph.has_edge(st,ed)):
    #         test_digraph.remove_edge(st,ed)
    
    # print (test_digraph.number_of_edges())

    # test_digraph1, test_digraph = evaluation_util.splitDiGraphToTrainTest(
    #     test_digraph,
    #     train_ratio=0.5,
    #     is_undirected=is_undirected
    # )

    # X, _ = graph_embedding.learn_embedding(graph=train_digraph, no_python=False)

    # if(verbose):
    #     parameter_file=open('graph_and_samplesn2v.p',"wb")
    #     pickle.dump(train_digraph, parameter_file)
    #     pickle.dump(test_digraph1, parameter_file)
    #     pickle.dump(test_digraph, parameter_file)
    #     pickle.dump(X,parameter_file)

    parameter_file=open('graph_and_samples.p',"rb")
    train_digraph = pickle.load(parameter_file)
    test_digraph1 = pickle.load(parameter_file)
    test_digraph = pickle.load(parameter_file)
    X = pickle.load(parameter_file)

    print ("embedding done")
        
    for (st,ed) in test_digraph1.edges():
        test_digraph.add_edge(st,ed)

    trp, trn = create_edge_dataset(train_digraph, test_digraph)
    print (len(trp), len(trn))


    trd1, trl1 = create_vector_dataset(trp[:len(trp)/2], trn[:len(trp)/2], hadamard1, X)
    trd2, trl2 = create_vector_dataset(trp[len(trp)/2:], trn[len(trp)/2:], hadamard1, X)
    clasifier = train_classifier(trd1, trl1)
    print ("trained")
    lab = clasifier.predict(trd2)
    print (metrics.accuracy_score(trl2,lab))


    trd1, trl1 = create_score_dataset(trp[:len(trp)/2], trn[:len(trp)/2], allh, train_digraph)
    trd2, trl2 = create_score_dataset(trp[len(trp)/2:], trn[len(trp)/2:], allh, train_digraph)
    clasifier = train_classifier(trd1, trl1)
    print ("trained")
    lab = clasifier.predict(trd2)
    print (metrics.accuracy_score(trl2,lab))

    # co=0
    # G=train_digraph.to_undirected()
    # for (st,ed) in trp[len(trp):]:  
    #     if(is_undirected and st>=ed):
    #         continue
    #     # if(clasifier.predict([average(X[st],X[ed])])[0]==1):
    #     if(clasifier.predict([allh(G,st,ed)])[0]==1):
    #         co=co+1

    # print (float(co)/test_digraph.to_undirected().number_of_edges())


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
    clasifier = train_classifier(trd, trl)
    
    estimated_adj = getscore6(train_digraph, node_l, clasifier, allh)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)

    for (st,ed) in train_digraph1.edges():
        train_digraph.add_edge(st,ed)
    
    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    MAP1 = scores.computeMAP(filtered_edge_list, test_digraph1)

    estimated_adj = getscore6(train_digraph, node_l, clasifier, allh)
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
    trd, trl = create_vector_dataset(trp, trn, func, X1)
    clasifier = train_classifier(trd, trl)

    X1=X1[node_l]
    estimated_adj = getscore2(train_digraph, node_l, clasifier, func, X1)

    for (st,ed) in train_digraph1.edges():
        train_digraph.add_edge(st,ed)
    
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    MAP1 = scores.computeMAP(filtered_edge_list, test_digraph1)
    
    X2=X2[node_l]
    estimated_adj = getscore2(train_digraph, node_l, clasifier, func, X2)
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
    trd, trl = create_vector_dataset(trp, trn, func, X1)
    clasifier = train_classifier(trd, trl)

    filtered_edge_list = getscore5(train_digraph, sample_edges, clasifier, func, X1)    
    AP1, ROC1 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    print (AP1,ROC1)
    for (st,ed) in train_digraph1.edges():
        train_digraph.add_edge(st,ed)
    
    filtered_edge_list = getscore5(train_digraph, sample_edges, clasifier, func, X2)    
    AP2, ROC2 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    
    print (AP2,ROC2)
    return AP1,AP2,ROC1,ROC2
    
# def evaluate_supervised_new(digraph, embeddings, hads, is_undirected=True):
def evaluate_supervised_new(train_digraph, embeddings, hads, is_undirected=True):

    train_digraph, test_digraph = train_test_split.splitDiGraphToTrainTest2(train_digraph, train_ratio = 0.6, is_undirected=True)
    for (st,ed) in train_digraph.edges():
        if(test_digraph.has_edge(st,ed)):
            test_digraph.remove_edge(st,ed)
        
    train_digraph1, test_digraph = evaluation_util.splitDiGraphToTrainTest(
        test_digraph,
        train_ratio=0.5,
        is_undirected=is_undirected
    )

    l_emb = []
    combine = []
    for emb in embeddings:
        X, _ = emb.learn_embedding(graph=train_digraph, no_python=False)
        l_emb.append(X)

    for had in hads:
        if(had==1):
            combine.append(hadamard1)
        elif(had==0):
            combine.append(hadamard2)

    # combine.append(dotp1)
    
    print ("embeddings learned")

    trp, trn = create_edge_dataset(train_digraph, train_digraph1)
    trd, trl = create_mix_dataset(trp, trn, train_digraph, l_emb, combine)
    mean=np.mean(trd,axis=0)
    std=np.std(trd,axis=0)
    trd=(trd-mean)/std
    clasifier = train_classifier(trd, trl)
    # print (clasifier.coef_)
    # print (clasifier.intercept_)
    
    train_digraph_temp=train_digraph.copy()
    for (st,ed) in train_digraph1.edges():
        train_digraph_temp.add_edge(st,ed)

    sample_edges = sample_edge_new(train_digraph_temp,test_digraph,-1,num_edges=500000)

    # co=0
    # for (st,ed) in sample_edges:
    #     for (st1,ed1) in trn:
    #         if(st==st1 and ed==ed1):
    #             if(test_digraph.has_edge(st,ed)):
    #                 print ("1")

    #     for (st1,ed1) in trp:
    #         if(st==st1 and ed==ed1):
    #             if(test_digraph.has_edge(st,ed)):
    #                 print ("2")
    #             else:
    #                 print ("3")
        
    # l_emb1 = []
    # for emb in embeddings:
    #     X, _ = emb.learn_embedding(graph=train_digraph_temp, no_python=False)
    #     l_emb1.append(X)
    #     break
    # l_emb1.append(l_emb[1])

    print ("embeddings learned")

    # filtered_edge_list = getscore9(train_digraph, sample_edges, clasifier, l_emb1, combine, mean, std)
    # AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    # print (AP,ROC)

    filtered_edge_list = getscore9(train_digraph_temp, sample_edges, clasifier, l_emb, combine, mean, std)
    AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    print (AP,ROC)

    trd, trl = create_score_dataset(trp, trn, allh, train_digraph)
    mean=np.mean(trd,axis=0)
    std=np.std(trd,axis=0)
    trd=(trd-mean)/std
    clasifier = train_classifier(trd, trl)
    filtered_edge_list = getscore7(train_digraph_temp, sample_edges, clasifier, allh, mean, std)
    AP2, ROC2 = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    print (AP2,ROC2)

    # G11 = train_digraph.to_undirected()
    # f1=[]
    # f2=[]
    # for (st,ed,w) in filtered_edge_list:
    #     f1.append(w)
    #     f2.append(cn(G11,st,ed))

    # f1=np.array(f1)
    # f2=np.array(f2)
    # ind1 = np.argsort(-1*f1)
    # ind2 = np.argsort(-1*f2)
    # print (ind1[:1000])
    # print (ind2[:1000])
    # print (f1[ind1[:1000]])
    # print (f2[ind1[:1000]])



    # filtered_edge_list = getscore3(train_digraph_temp, sample_edges, aa)
    # AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    # print (AP,ROC)

    return AP,ROC

    # labels=[]
    # score=[]
    # dist=[]
    # G=train_digraph.to_undirected()
    # print (len(filtered_edge_list))
    # for (st,ed,w) in filtered_edge_list:
    #     # if not(nx.shortest_path_length(G,source=st,target=ed)==2):
    #     #     continue
    #     if(test_digraph.has_edge(st,ed)):
    #         labels.append(1)
    #     else:
    #         labels.append(0)
    #     score.append(w)
    # ap = average_precision_score(labels, score)
    # print (ap)

    # ind = np.argsort(-1*np.asarray(score))
    # labels = np.array(labels)
    # print (labels[ind[:1000]])
    
    # labels=[]
    # score=[]
    # dist=[]
    # G=train_digraph.to_undirected()
    # for (st,ed,w) in filtered_edge_list:
        # if (nx.shortest_path_length(G,source=st,target=ed)==2):
        #     continue
    #     if(test_digraph.has_edge(st,ed)):
    #         labels.append(1)
    #     else:
    #         labels.append(0)
    #     score.append(w)
    # ap = average_precision_score(labels, score)
    # print (ap)

    # ind = np.argsort(-1*np.asarray(score))
    # labels = np.array(labels)
    # print (labels[ind[:1000]])
        
    # test_digraph, node_l = graph_util.sample_graph(test_digraph, 1024)
    # estimated_adj = getscore8(train_digraph, node_l, clasifier, l_emb1, combine)
    # predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj,is_undirected=True)
    # filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    # MAP = scores.computeMAP(filtered_edge_list, test_digraph)

    # print (MAP)
    MAP=0
    
    return AP, ROC, MAP


def evaluate_supervised_new1(train_digraph, train_digraph1, test_digraph, trp1, trn1, trp2, trn2, trp3, trn3, sample_edges, l_emb, hads, is_undirected=True):

    combine=[]
    for had in hads:
        if(had==1):
            combine.append(hadamard1)
        elif(had==0):
            combine.append(hadamard2)

    # create dataset for training the classifier with appropriate combination of embeddings and heuristics.

    # only heuristics
    trd1, trl1 = create_mix_dataset(trp1, trn1, train_digraph, [], combine, func_new)
    # combination of embeddings
    trd2, trl2 = create_mix_dataset(trp2, trn2, train_digraph, l_emb, combine, func_new1)
    # combination of embeddings and heuristics
    trd3, trl3 = create_mix_dataset(trp3, trn3, train_digraph, l_emb, combine, func_new)

    # learn classifiers for the three types of featre combination.

    mean1=np.mean(trd1,axis=0)
    std1=np.std(trd1,axis=0)
    for x in xrange(len(std1)):
        std1[x]=1;
    trd1=(trd1-mean1)/std1
    clasifier1 = train_classifier(trd1, trl1)
    
    mean2=np.mean(trd2,axis=0)
    std2=np.std(trd2,axis=0)
    for x in xrange(len(std2)):
        std2[x]=1;
    trd2=(trd2-mean2)/std2
    clasifier2 = train_classifier(trd2, trl2)
    
    mean3=np.mean(trd3,axis=0)
    std3=np.std(trd3,axis=0)
    for x in xrange(len(std3)):
        std3[x]=1;
    trd3=(trd3-mean3)/std3
    clasifier3 = train_classifier(trd3, trl3)
    
    train_digraph_temp=train_digraph.copy()
    for (st,ed) in train_digraph1.edges():
        train_digraph_temp.add_edge(st,ed)

    f2=[]
    f3=[]
    G=train_digraph_temp.to_undirected()
    for (st,ed) in sample_edges:
        dis=nx.shortest_path_length(G,source=st,target=ed)
        if (dis==2):
            f2.append((st,ed))
        else:
            f3.append((st,ed))
        
    # use train_graph_temp=train_graph + train_graph1 for calculating heuristic scores. 


    # use this section for evaluating on distance 2 edges

    # filtered_edge_list1 = getscore9(train_digraph_temp, f2, clasifier1, [], combine, func_new, mean1, std1)
    # filtered_edge_list2 = getscore9(train_digraph_temp, f2, clasifier2, l_emb, combine, func_new1, mean2, std2)
    # filtered_edge_list3 = getscore9(train_digraph_temp, f2, clasifier3, l_emb, combine, func_new, mean3, std3)


    # use this section for evaluating on distance 2+ edges
    
    # filtered_edge_list1 = getscore9(train_digraph_temp, f3, clasifier1, [], combine, func_new, mean1, std1)
    # filtered_edge_list2 = getscore9(train_digraph_temp, f3, clasifier2, l_emb, combine, func_new1, mean2, std2)
    # filtered_edge_list3 = getscore9(train_digraph_temp, f3, clasifier3, l_emb, combine, func_new, mean3, std3)


    # use this section for evaluating on all edges
    
    filtered_edge_list1 = getscore9(train_digraph_temp, sample_edges, clasifier1, [], combine, func_new, mean1, std1)
    filtered_edge_list2 = getscore9(train_digraph_temp, sample_edges, clasifier2, l_emb, combine, func_new1, mean2, std2)
    filtered_edge_list3 = getscore9(train_digraph_temp, sample_edges, clasifier3, l_emb, combine, func_new, mean3, std3)


    AP11, ROC11 = scores.computeAP_ROC(filtered_edge_list1, test_digraph)
    print (AP11,ROC11)
    AP12, ROC12 = scores.computeAP_ROC(filtered_edge_list2, test_digraph)
    print (AP12,ROC12)
    AP13, ROC13 = scores.computeAP_ROC(filtered_edge_list3, test_digraph)
    print (AP13,ROC13)

    return AP11,ROC11,AP12,ROC12,AP13,ROC13
    
    # prec_curve11,_ = scores.computePrecisionCurve(filtered_edge_list1, test_digraph)
    # prec_curve12,_ = scores.computePrecisionCurve(filtered_edge_list2, test_digraph)
    # prec_curve13,_ = scores.computePrecisionCurve(filtered_edge_list3, test_digraph)

    
    # plt.plot(np.arange(min(100,len(prec_curve11))),prec_curve11[:100],'b',label='embeddings')
    # plt.plot(np.arange(min(100,len(prec_curve21))),prec_curve12[:100],'r',label='heurestics')
    # plt.plot(np.arange(min(100,len(prec_curve21))),prec_curve123[:100],'r',label='heurestics')
    # plt.legend()
    # plt.show()

    
    # print (AP21,ROC21)
    # AP22, ROC22 = scores.computeAP_ROC(f3, test_digraph)
    # print (AP22,ROC22)
    # AP23, ROC23 = scores.computeAP_ROC(f4, test_digraph)
    # print (AP23,ROC23)
    # AP24, ROC24 = scores.computeAP_ROC(f5, test_digraph)
    # print (AP24,ROC24)

    # f2=[]
    # f3=[]
    # f4=[]
    # f5=[]
    # G=train_digraph.to_undirected()
    # for (st,ed,w) in filtered_edge_list:
    #     # f2.append((st,ed,w))
    #     dis=nx.shortest_path_length(G,source=st,target=ed)
    #     if (dis==2):
    #         f2.append((st,ed,w))
    #     else:
    #         f3.append((st,ed,w))
    #     # elif (dis==3):
    #     #     f3.append((st,ed,w))
    #     # elif (dis==4):
    #     #     f4.append((st,ed,w))
    #     # else:
    #         # f5.append((st,ed,w))

    # # prec_curve11,_ = scores.computePrecisionCurve(f2, test_digraph)
    # # prec_curve12,_ = scores.computePrecisionCurve(f3, test_digraph)
    # # prec_curve13,_ = scores.computePrecisionCurve(f4, test_digraph)
    # # prec_curve14,_ = scores.computePrecisionCurve(f5, test_digraph)
    
    # AP11, ROC11 = scores.computeAP_ROC(f2, test_digraph)
    # print (AP11,ROC11)
    # AP12, ROC12 = scores.computeAP_ROC(f3, test_digraph)
    # print (AP12,ROC12)
    # # AP13, ROC13 = scores.computeAP_ROC(f4, test_digraph)
    # # print (AP13,ROC13)
    # # AP14, ROC14 = scores.computeAP_ROC(f5, test_digraph)
    # # print (AP14,ROC14)

    # # plt.show()
    # # AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    # # print (AP,ROC)

    # trd, trl = create_mix_dataset(trp, trn, train_digraph, [], combine, func_new)
    # mean=np.mean(trd,axis=0)
    # std=np.std(trd,axis=0)
    # trd=(trd-mean)/std
    # clasifier = train_classifier(trd, trl)
    
    # train_digraph_temp=train_digraph.copy()
    # for (st,ed) in train_digraph1.edges():
    #     train_digraph_temp.add_edge(st,ed)

    # filtered_edge_list = getscore9(train_digraph_temp, sample_edges, clasifier, [], combine, func_new, mean, std)
    
    # f2=[]
    # f3=[]
    # f4=[]
    # f5=[]
    # G=train_digraph.to_undirected()
    # for (st,ed,w) in filtered_edge_list:
    #     # f2.append((st,ed,w))
    #     dis=nx.shortest_path_length(G,source=st,target=ed)
    #     if (dis==2):
    #         f2.append((st,ed,w))
    #     else:
    #         f3.append((st,ed,w))
    #     # elif (dis==3):
    #     #     f3.append((st,ed,w))
    #     # elif (dis==4):
    #     #     f4.append((st,ed,w))
    #     # else:
    #     #     f5.append((st,ed,w))

    # # prec_curve21,_ = scores.computePrecisionCurve(f2, test_digraph)
    # # prec_curve22,_ = scores.computePrecisionCurve(f3, test_digraph)
    # # prec_curve23,_ = scores.computePrecisionCurve(f4, test_digraph)
    # # prec_curve24,_ = scores.computePrecisionCurve(f5, test_digraph)

    # AP21, ROC21 = scores.computeAP_ROC(f2, test_digraph)
    # # AP, ROC = scores.computeAP_ROC(filtered_edge_list, test_digraph)
    # # print (AP,ROC)


    # # plt.plot(np.arange(1000),prec_curve11[:1000],'b')
    # # plt.plot(np.arange(1000),prec_curve12[:1000],'r')
    # # plt.plot(np.arange(1000),prec_curve21[:1000],'g')
    # # plt.plot(np.arange(1000),prec_curve22[:1000],'y')
    # # plt.show()

    # # fig = plt.figure()
    # # ax = plt.subplot(211)
    # # ax.set_title('d2', fontsize=10)
    # # box = ax.get_position()
    # # ax.set_position([box.x0-0.05, box.y0 - box.height * 0.05,box.width*0.9, box.height])
    # # plt.plot(np.arange(min(100,len(prec_curve11))),prec_curve11[:100],'b',label='proposed')
    # # plt.plot(np.arange(min(100,len(prec_curve21))),prec_curve21[:100],'r',label='heurestics')
    
    # # ax = plt.subplot(212)
    # # ax.set_title('d3', fontsize=10)
    # # box = ax.get_position()
    # # ax.set_position([box.x0-0.05, box.y0 - box.height * 0.05,box.width*0.9, box.height])
    # # plt.plot(np.arange(min(100,len(prec_curve12))),prec_curve12[:100],'b',label='proposed')
    # # plt.plot(np.arange(min(100,len(prec_curve22))),prec_curve22[:100],'r',label='heurestics')
    
    # # ax = plt.subplot(223)
    # # ax.set_title('d4', fontsize=10)
    # # box = ax.get_position()
    # # ax.set_position([box.x0-0.05, box.y0 - box.height * 0.05,box.width*0.9, box.height])
    # # plt.plot(np.arange(min(100,len(prec_curve13))),prec_curve13[:100],'b',label='proposed')
    # # plt.plot(np.arange(min(100,len(prec_curve23))),prec_curve23[:100],'r',label='heurestics')
    
    # # ax = plt.subplot(224)
    # # ax.set_title('d4+', fontsize=10)
    # # box = ax.get_position()
    # # ax.set_position([box.x0-0.05, box.y0 - box.height * 0.05,box.width*0.9, box.height])
    # # plt.plot(np.arange(min(100,len(prec_curve14))),prec_curve14[:100],'b',label='proposed')
    # # plt.plot(np.arange(min(100,len(prec_curve24))),prec_curve24[:100],'r',label='heuristics')

    # # plt.legend(loc=(1.05,0))
    # # plt.show()


    # # print (prec_curve1)
    # # print (prec_curve2)
    # # print (prec_curve3)
    # # print (prec_curve4)

    # return AP12,ROC12,AP22,ROC22