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

list_graphs = ['gem/data/vk_2016']
list_graphs1 = ['gem/data/vk_2017']
list_directed = [True]
fig_name = ['VK']

dimensions=[2,4,8,16,32,64,128]
num_samples=1

if not os.path.exists('SAVER_TE_SUP'):
	os.makedirs('SAVER_TE_SUP')
print ("saving for supervised")

for grp in xrange(len(list_graphs)):
	for x in xrange(num_samples):

		# for storing embeddings

		train_digraph = graph_util.loadGraphFromEdgeListTxt(list_graphs[grp], directed=list_directed[grp])
		test_digraph = graph_util.loadGraphFromEdgeListTxt(list_graphs1[grp], directed=list_directed[grp])
		train_digraph = train_digraph.to_directed()
		test_digraph = test_digraph.to_directed()
		
		for (st,ed) in train_digraph.edges():
			if(test_digraph.has_edge(st,ed)):
				test_digraph.remove_edge(st,ed)
		
		if not os.path.exists('SAVER_TE_SUP/'+fig_name[grp]):
			os.makedirs('SAVER_TE_SUP/'+fig_name[grp])
		
		train_digraph1, test_digraph = evaluation_util.splitDiGraphToTrainTest(
		test_digraph,
		train_ratio=0.5,
		is_undirected=True
		)

		print ("saving for DEEPWALK")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_TE_SUP/'+fig_name[grp]+'/DEEPWALK1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
		
		print ("saving for n2vA")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=4, inout_p=0.5)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_TE_SUP/'+fig_name[grp]+'/n2vA1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for n2vB")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=4)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_TE_SUP/'+fig_name[grp]+'/n2vB1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
		
		print ("saving for verse")
		for dim in dimensions:
			embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_TE_SUP/'+fig_name[grp]+'/VERSE1_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		file_name='SAVER_TE_SUP/'+fig_name[grp]+'/graph_and_samples.p'
		parameter_file=open(file_name, 'wb')
		pickle.dump(train_digraph,parameter_file)
		pickle.dump(train_digraph1,parameter_file)
		pickle.dump(test_digraph,parameter_file)
	
		for (st,ed) in train_digraph1.edges():
			train_digraph.add_edge(st,ed)

		print ("saving for DEEPWALK")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_TE_SUP/'+fig_name[grp]+'/DEEPWALK2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
		
		print ("saving for n2vA")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=4, inout_p=0.5)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_TE_SUP/'+fig_name[grp]+'/n2vA2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		print ("saving for n2vB")
		for dim in dimensions:
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=4)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_TE_SUP/'+fig_name[grp]+'/n2vB2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()
	
		print ("saving for verse")
		for dim in dimensions:
			embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
			X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
			file_name='SAVER_TE_SUP/'+fig_name[grp]+'/VERSE2_'+str(dim)
			parameter_file=open(file_name, 'wb')
			pickle.dump(X,parameter_file)
			parameter_file.close()

		# for adding new embeddings

		# file_name='SAVER_TE_SUP/'+fig_name[grp]+'/graph_and_samples.p'
		# parameter_file=open(file_name, 'rb')
		# train_digraph=pickle.load(parameter_file)
		# train_digraph1=pickle.load(parameter_file)
		# test_digraph=pickle.load(parameter_file)
		# parameter_file.close()
		
		# for dim in dimensions:
		# 	embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
		# 	X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
		# 	file_name='SAVER_TE_SUP/'+fig_name[grp]+'/VERSE1_'+str(dim)
		# 	parameter_file=open(file_name, 'wb')
		# 	pickle.dump(X,parameter_file)
		# 	parameter_file.close()

		# for (st,ed) in train_digraph1.edges():
		# 	train_digraph.add_edge(st,ed)
		
		# for dim in dimensions:
		# 	embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
		# 	X, _ = embedding.learn_embedding(graph=train_digraph, no_python=False)
		# 	file_name='SAVER_TE_SUP/'+fig_name[grp]+'/VERSE2_'+str(dim)
		# 	parameter_file=open(file_name, 'wb')
		# 	pickle.dump(X,parameter_file)
		# 	parameter_file.close()
		
