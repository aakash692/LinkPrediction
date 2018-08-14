# for non temporal graphs 
# save embeddings for supervised prediction

import networkx.generators.random_graphs as rangr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from gem.evaluation 		import evaluation_measures
from gem.utils 				import train_test_split, graph_util, evaluation_util
from gem.embedding.lap	 	import LaplacianEigenmaps
from gem.embedding.hope		import HOPE
from gem.embedding.node2vec import node2vec
from gem.embedding.verse 	import verse
import sys
import os
import pickle

# list_graphs stores the list of edge list file for which embeddings need to be saved

# list_graphs = ['gem/data/facebook_combined.txt','gem/data/CA-AstroPh.txt']
# list_graphs = ['gem/data/CoCit.txt']
# list_directed = [True]
# fig_name = ['FACEBOOK','ASTROPH']
# fig_name = ['CoCit']
# dimensions=[2,4,8,16,32,64,128,256]

list_graphs = ['gem/data/karate.edgelist']
list_directed = [False]
fig_name = ['Karate']
dimensions=[2,4,8]
num_samples=1

if not os.path.exists('SAVER_SUP'):
	os.makedirs('SAVER_SUP')
print ("saving for supervised")

for grp in xrange(len(list_graphs)):
	for x in xrange(num_samples):

		# load the graph as a networkx graph

		G = graph_util.loadGraphFromEdgeListTxt(list_graphs[grp], directed=list_directed[grp])
		G = G.to_directed()
		
		if not os.path.exists('SAVER_SUP/'+fig_name[grp]+str(x+1)):
			os.makedirs('SAVER_SUP/'+fig_name[grp]+str(x+1))
		
		# split the graph into 60-20-20 ratio, 60% for calculating the edge features, 20% for training the classifier, 20% for evaluating the model.

		train_digraph, test_digraph = train_test_split.splitDiGraphToTrainTest2(G, train_ratio = 0.6, is_undirected=True)
		train_digraph1, test_digraph = evaluation_util.splitDiGraphToTrainTest(test_digraph, train_ratio=0.5, is_undirected=True)

		# embeddings without relearning

		print ("saving for LE")
		for dim in dimensions:
			embedding=LaplacianEigenmaps(d=dim)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/LE1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for DEEPWALK")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/DEEPWALK1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
		
		print ("saving for n2vA")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=4, inout_p=0.5)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/n2vA1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for n2vB")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=4)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/n2vB1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
		
		print ("saving for verse")
		for dim in dimensions:
			embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/VERSE1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for hope")
		for dim in dimensions:
			embedding=HOPE(d=dim, beta=0.01)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/HOPE1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/graph_and_samples.p'
		parameter_file=open(file_name, 'wb')
		pickle.dump(train_digraph,parameter_file)
		pickle.dump(train_digraph1,parameter_file)
		pickle.dump(test_digraph,parameter_file)
		
		# embeddings with relearning

		for (st,ed) in train_digraph1.edges():
			train_digraph.add_edge(st,ed)
		
		print ("saving for LE")
		for dim in dimensions:
			embedding=LaplacianEigenmaps(d=dim)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/LE2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for DEEPWALK")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/DEEPWALK2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
		
		print ("saving for n2vA")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=4, inout_p=0.5)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/n2vA2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for n2vB")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=4)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/n2vB2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
	
		print ("saving for verse")
		for dim in dimensions:
			embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/VERSE2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for hope")
		for dim in dimensions:
			embedding=HOPE(d=dim, beta=0.01)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/HOPE2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		# for adding new embeddings

		# file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/graph_and_samples.p'
		# parameter_file=open(file_name, 'rb')
		# train_digraph=pickle.load(parameter_file)
		# train_digraph1=pickle.load(parameter_file)
		# test_digraph=pickle.load(parameter_file)
		# parameter_file.close()

		# for dim in dimensions:
		# 	embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
		# 	X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
		# 	file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/VERSE2_'+str(dim)
		# 	parameter_file=open(file_name, 'wb')
		# 	pickle.dump(X,parameter_file)
		# 	parameter_file.close()

		# for (st,ed) in train_digraph1.edges():
		# 	train_digraph.add_edge(st,ed)
	
		# for dim in dimensions:
		# 	embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
		# 	X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
		# 	file_name='SAVER_SUP/'+fig_name[grp]+str(x+1)+'/VERSE2_'+str(dim)
		# 	parameter_file=open(file_name, 'wb')
		# 	pickle.dump(X,parameter_file)
		# 	parameter_file.close()		
		
