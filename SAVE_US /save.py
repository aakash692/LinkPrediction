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

# list_graphs = ['gem/data/facebook_combined.txt','gem/data/CA-AstroPh.txt']
# list_graphs = ['gem/data/CoCit.txt']
# list_directed = [False,True]
# list_directed = [True]
# fig_name = ['FACEBOOK','ASTROPH']
# fig_name = ['CoCit']

# dimensions=[2,4,8,16,32,64,128,256]

list_graphs = ['gem/data/karate.edgelist']
list_directed = [False]
fig_name = ['Karate']
dimensions=[2,4,8]

num_samples=1

if not os.path.exists('SAVER'):
	os.makedirs('SAVER')
print ("saving for unsupervised")

for grp in xrange(len(list_graphs)):
	for x in xrange(num_samples):

		# for storing embeddings

		G = graph_util.loadGraphFromEdgeListTxt(list_graphs[grp], directed=list_directed[grp])
		G = G.to_directed()
		if not os.path.exists('SAVER/'+fig_name[grp]+str(x+1)):
			os.makedirs('SAVER/'+fig_name[grp]+str(x+1))
		train_digraph, test_digraph = train_test_split.splitDiGraphToTrainTest2(G, train_ratio = 0.8, is_undirected=True)

		file_name='SAVER/'+fig_name[grp]+str(x+1)+'/graph_and_samples.p'
		parameter_file=open(file_name, 'wb')
		pickle.dump(train_digraph,parameter_file)
		pickle.dump(test_digraph,parameter_file)
		parameter_file.close()
		
		print ("saving for LE")
		for dim in dimensions:
			embedding=LaplacianEigenmaps(d=dim)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER/'+fig_name[grp]+str(x+1)+'/LE_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for hope")
		for dim in dimensions:
			embedding=HOPE(d=dim, beta=0.01)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER/'+fig_name[grp]+str(x+1)+'/HOPE_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()	

		print ("saving for DEEPWALK")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER/'+fig_name[grp]+str(x+1)+'/DEEPWALK_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
		
		print ("saving for n2vA")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=4, inout_p=0.5)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER/'+fig_name[grp]+str(x+1)+'/n2vA_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for n2vB")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=4)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER/'+fig_name[grp]+str(x+1)+'/n2vB_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
		
		print ("saving for verse")
		for dim in dimensions:
			embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER/'+fig_name[grp]+str(x+1)+'/VERSE_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()	

		# for adding new embeddings

		# file_name='SAVER/'+fig_name[grp]+str(x+1)+'/graph_and_samples.p'
		# parameter_file=open(file_name, 'rb')
		# train_digraph=pickle.load(parameter_file)
		# parameter_file.close()

		# for dim in dimensions:
		# 	embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
		# 	X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
		# 	file_name='SAVER/'+fig_name[grp]+str(x+1)+'/VERSE_'+str(dim)
		# 	parameter_file=open(file_name, 'wb')
		# 	pickle.dump(X,parameter_file)
		# 	parameter_file.close()		
