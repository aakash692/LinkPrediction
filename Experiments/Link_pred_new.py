import networkx.generators.random_graphs as rangr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from gem.evaluation import evaluation_measures
from gem.utils import train_test_split, graph_util
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.hope     import HOPE
from gem.embedding.node2vec import node2vec
from gem.embedding.verse 	import verse
import sys
import pickle

# provide the data file name in list_graphs and specify if it is directed or not in lis_directed
# specify the dimension of the embedding in dimensions array

# edge_f = 'gem/data/karate.edgelist'
# edge_f = 'gem/data/CA-AstroPh.txt'
# edge_f = 'gem/data/facebook_combined.txt'
# list_graphs = ['gem/data/karate.edgelist','gem/data/facebook_combined.txt','gem/data/CA-AstroPh.txt']
list_graphs = ['gem/data/CA-AstroPh.txt']
# list_graphs = ['gem/data/facebook_combined.txt']
# list_directed = [False,False,True]
list_directed = [True]
# list_directed = [False]
# fig_name = ['CoCit']
# fig_name = ['ASTROPH']
fig_name = ['VK']
# fig_name = ['FACEBOOK']
# dimensions=[2,4,8]
# dimensions=[2,4,8,16,32,64,128,256]
dimensions=[128]
num_samples=5
ratios=[-1]

# for ASTROPH graph

# for rat in ratios:

# 	print (rat)

# 	AP1=np.zeros(num_samples)
# 	ROC1=np.zeros(num_samples)
# 	AP2=np.zeros(num_samples)
# 	ROC2=np.zeros(num_samples)
# 	AP3=np.zeros(num_samples)
# 	ROC3=np.zeros(num_samples)

# 	for it1 in xrange(num_samples):

# 		emb=[]
# 		hads=[]
		
# 		# graph_and_samples.p stores the train graph(60%),train_graph1(20%). Both of them are used to train the classsifier.
# 		# it also has the test graph  

# 		file_name='../../FULL/SAVE_S/Evaluations/SAVER_SUP/'+fig_name[0]+str(it1+1)+'/graph_and_samples.p'
# 		parameter_file=open(file_name, 'rb')
# 		train_digraph = pickle.load(parameter_file)
# 		train_digraph1 = pickle.load(parameter_file)
# 		test_digraph =  pickle.load(parameter_file)
# 		parameter_file.close()

# 		# create train dataset using train graph and train graph1 with equal number of positive and negative examples.
# 		# can give different values of third parameter to this function, 0-> distance 2 edges, 1-> distance 2+ edges, anything else->take all edges irrespective of distance.

# 		trp1, trn1 = evaluation_measures.create_edge_dataset(train_digraph, train_digraph1, 0)
# 		# trp1, trn1 = evaluation_measures.create_edge_dataset(train_digraph, train_digraph1, 1)
# 		# trp1, trn1 = evaluation_measures.create_edge_dataset(train_digraph, train_digraph1, 2)

# 		train_digraph_temp = train_digraph.copy()
# 		for (st,ed) in train_digraph1.edges():
# 			train_digraph_temp.add_edge(st,ed)

# 		# sample edges not present during training with desired ratio to create the test set.
# 		sample_edges = evaluation_measures.sample_edge_new(train_digraph_temp,test_digraph,rat)
		
# 		# emb->specify which all embeddings to use in the combibation, for example VERSE1_128 represents embeddings with verse algorithm, 1 represents without relearning and 128 represents the dimension.
# 		# hads->specify how to combine the two node vectors, 0 represents normalised hadamard, 1 represents un-normalised hadamard.

# 		file_name='../../FULL/SAVE_S/Evaluations/SAVER_SUP/'+fig_name[0]+str(it1+1)+'/n2vA1_'+str(128)
# 		parameter_file=open(file_name, 'rb')
# 		X1 = pickle.load(parameter_file)
# 		emb.append(X1)
# 		hads.append(0)
# 		parameter_file.close()

# 		file_name='../../FULL/SAVE_S/Evaluations/SAVER_SUP/'+fig_name[0]+str(it1+1)+'/VERSE1_'+str(128)
# 		parameter_file=open(file_name, 'rb')
# 		X1 = pickle.load(parameter_file)
# 		emb.append(X1)
# 		hads.append(1)
# 		parameter_file.close()

# 		ap1,roc1,ap2,roc2,ap3,roc3 = evaluation_measures.evaluate_supervised_new1(train_digraph,train_digraph1,test_digraph,trp1,trn1,trp1,trn1,trp1,trn1,sample_edges,emb,hads)
# 		AP1[it1]=ap1
# 		ROC1[it1]=roc1
# 		AP2[it1]=ap2
# 		ROC2[it1]=roc2
# 		AP3[it1]=ap3
# 		ROC3[it1]=roc3

# 	mean_ap=np.mean(AP1)
# 	std_ap=np.std(AP1)
# 	mean_roc=np.mean(ROC1)
# 	std_roc=np.std(ROC1)

# 	print (mean_ap)
# 	print (std_ap)
# 	print (mean_roc)
# 	print (std_roc)

# 	mean_ap=np.mean(AP2)
# 	std_ap=np.std(AP2)
# 	mean_roc=np.mean(ROC2)
# 	std_roc=np.std(ROC2)

# 	print (mean_ap)
# 	print (std_ap)
# 	print (mean_roc)
# 	print (std_roc)

# 	mean_ap=np.mean(AP3)
# 	std_ap=np.std(AP3)
# 	mean_roc=np.mean(ROC3)
# 	std_roc=np.std(ROC3)

# 	print (mean_ap)
# 	print (std_ap)
# 	print (mean_roc)
# 	print (std_roc)


# _-----------------------------------------------------------------_
# for VK graph

for rat in ratios:

	print (rat)

	AP1=np.zeros(num_samples)
	ROC1=np.zeros(num_samples)
	AP2=np.zeros(num_samples)
	ROC2=np.zeros(num_samples)
	AP3=np.zeros(num_samples)
	ROC3=np.zeros(num_samples)

	for it1 in xrange(num_samples):

		emb=[]
		hads=[]

 		# graph_and_samples.p stores the train graph(60%),train_graph1(20%). Both of them are used to train the classsifier.
 		# it also has the test graph  

		file_name='../../FULL/SAVE_TE_S/Evaluations/SAVER_TE_SUP/'+fig_name[0]+'/graph_and_samples.p'
		parameter_file=open(file_name, 'rb')
		train_digraph = pickle.load(parameter_file)
		train_digraph1 = pickle.load(parameter_file)
		test_digraph =  pickle.load(parameter_file)
		parameter_file.close()

		# create train dataset using train graph and train graph1 with equal number of positive and negative examples.
		# can give different values of third parameter to this function, 0-> distance 2 edges, 1-> distance 2+ edges, anything else->take all edges irrespective of distance.

		# trp1, trn1 = evaluation_measures.create_edge_dataset(train_digraph, train_digraph1, 0)
		# trp1, trn1 = evaluation_measures.create_edge_dataset(train_digraph, train_digraph1, 1)
		trp1, trn1 = evaluation_measures.create_edge_dataset(train_digraph, train_digraph1, 2)
		train_digraph_temp = train_digraph.copy()
		for (st,ed) in train_digraph1.edges():
			train_digraph_temp.add_edge(st,ed)

		# sample edges not present during training with desired ratio to create the test set.
		sample_edges = evaluation_measures.sample_edge_new(train_digraph_temp,test_digraph,rat)

		# emb->specify which all embeddings to use in the combibation, for example VERSE1_128 represents embeddings with verse algorithm, 1 represents without relearning and 128 represents the dimension.
		# hads->specify how to combine the two node vectors, 0 represents normalised hadamard, 1 represents un-normalised hadamard.
		
		file_name='../../FULL/SAVE_TE_S/Evaluations/SAVER_TE_SUP/'+fig_name[0]+'/n2vA1_'+str(128)
		parameter_file=open(file_name, 'rb')
		X1 = pickle.load(parameter_file)
		emb.append(X1)
		hads.append(0)
		parameter_file.close()

		file_name='../../FULL/SAVE_TE_S/Evaluations/SAVER_TE_SUP/'+fig_name[0]+'/VERSE1_'+str(128)
		parameter_file=open(file_name, 'rb')
		X1 = pickle.load(parameter_file)
		emb.append(X1)
		hads.append(1)
		parameter_file.close()

		ap1,roc1,ap2,roc2,ap3,roc3 = evaluation_measures.evaluate_supervised_new1(train_digraph,train_digraph1,test_digraph,trp1,trn1,trp1,trn1,trp1,trn1,sample_edges,emb,hads)
		AP1[it1]=ap1
		ROC1[it1]=roc1
		AP2[it1]=ap2
		ROC2[it1]=roc2
		AP3[it1]=ap3
		ROC3[it1]=roc3

	mean_ap=np.mean(AP1)
	std_ap=np.std(AP1)
	mean_roc=np.mean(ROC1)
	std_roc=np.std(ROC1)

	print (mean_ap)
	print (std_ap)
	print (mean_roc)
	print (std_roc)

	mean_ap=np.mean(AP2)
	std_ap=np.std(AP2)
	mean_roc=np.mean(ROC2)
	std_roc=np.std(ROC2)

	print (mean_ap)
	print (std_ap)
	print (mean_roc)
	print (std_roc)
	
	mean_ap=np.mean(AP3)
	std_ap=np.std(AP3)
	mean_roc=np.mean(ROC3)
	std_roc=np.std(ROC3)

	print (mean_ap)
	print (std_ap)
	print (mean_roc)
	print (std_roc)

