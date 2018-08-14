import networkx.generators.random_graphs as rangr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from gem.evaluation 		import evaluation_measures
from gem.utils				import train_test_split, graph_util, evaluation_util
from gem.embedding.lap	 	import LaplacianEigenmaps
from gem.embedding.hope     import HOPE
from gem.embedding.node2vec import node2vec
from gem.embedding.verse 	import verse
import sys
import pickle
from gem.evaluation import metrics as scores
import csv

list_graphs = ['gem/data/vk_2016']
list_graphs1 = ['gem/data/vk_2017']
list_directed = [True]
fig_name = ['VK']

dimensions=[2,4,8,16,32,64,128]
num_samples=5


# MAP for unsupervised

# for fig in xrange(len(fig_name)):

# 	MAP_DEEPWALK=np.zeros((len(dimensions),num_samples))
# 	MAP_n2vA=np.zeros((len(dimensions),num_samples))
# 	MAP_n2vB=np.zeros((len(dimensions),num_samples))
# 	MAP_VERSE=np.zeros((len(dimensions),num_samples))
# 	MAP_us=np.zeros((4,num_samples))

# 	for it1 in xrange(num_samples):

# 		file_name='SAVER_TE/'+fig_name[fig]+'/graph_and_samples.p'
# 		parameter_file=open(file_name, 'rb')
# 		train_digraph = pickle.load(parameter_file)
# 		test_digraph =  pickle.load(parameter_file)
# 		parameter_file.close()

# 		test_digraph1, node_l = graph_util.sample_graph(test_digraph, 1024)

# 		print ("evaluating for heuristic")		
# 		ma = evaluation_measures.calc_map_heu(node_l, train_digraph, test_digraph1)
# 		for it2 in xrange(4):
# 			print (it1,it2)
# 			MAP_us[it2][it1] = ma[it2]

# 		print ("evaluating for verse")
# 		for it2 in xrange(len(dimensions)):
# 			print (it1,it2)
# 			dim=dimensions[it2]		
# 			file_name='SAVER_TE/'+fig_name[fig]+'/VERSE_'+str(dim)
# 			parameter_file=open(file_name, 'rb')
# 			X = pickle.load(parameter_file)
# 			parameter_file.close()
# 			X = X[node_l]
# 			embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
# 			MAP_VERSE[it2][it1] = evaluation_measures.calc_map_us(embedding, X, node_l, train_digraph, test_digraph1)

# 		print ("evaluating for DEEPWALK")
# 		for it2 in xrange(len(dimensions)):
# 			print (it1,it2)
# 			dim=dimensions[it2]		
# 			file_name='SAVER_TE/'+fig_name[fig]+'/DEEPWALK_'+str(dim)
# 			parameter_file=open(file_name, 'rb')
# 			X = pickle.load(parameter_file)
# 			parameter_file.close()
# 			X = X[node_l]
# 			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
# 			MAP_DEEPWALK[it2][it1] = evaluation_measures.calc_map_us(embedding, X, node_l, train_digraph, test_digraph1)

# 		print ("evaluating for n2vA")
# 		for it2 in xrange(len(dimensions)):
# 			print (it1,it2)
# 			dim=dimensions[it2]		
# 			file_name='SAVER_TE/'+fig_name[fig]+'/n2vA_'+str(dim)
# 			parameter_file=open(file_name, 'rb')
# 			X = pickle.load(parameter_file)
# 			parameter_file.close()
# 			X = X[node_l]
# 			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=4, inout_p=0.5)
# 			MAP_n2vA[it2][it1] = evaluation_measures.calc_map_us(embedding, X, node_l, train_digraph, test_digraph1)

# 		print ("evaluating for n2vB")
# 		for it2 in xrange(len(dimensions)):
# 			print (it1,it2)
# 			dim=dimensions[it2]		
# 			file_name='SAVER_TE/'+fig_name[fig]+'/n2vB_'+str(dim)
# 			parameter_file=open(file_name, 'rb')
# 			X = pickle.load(parameter_file)
# 			parameter_file.close()
# 			X = X[node_l]
# 			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=4)
# 			MAP_n2vB[it2][it1] = evaluation_measures.calc_map_us(embedding, X, node_l, train_digraph, test_digraph1)

# 	mean_DEEPWALK = np.mean(MAP_DEEPWALK,axis=1)
# 	std_DEEPWALK = np.std(MAP_DEEPWALK,axis=1)
# 	mean_n2vA = np.mean(MAP_n2vA,axis=1)
# 	std_n2vA = np.std(MAP_n2vA,axis=1)
# 	mean_n2vB = np.mean(MAP_n2vB,axis=1)
# 	std_n2vB = np.std(MAP_n2vB,axis=1)
# 	mean_VERSE = np.mean(MAP_VERSE,axis=1)
# 	std_VERSE = np.std(MAP_VERSE,axis=1)
# 	mean_us = np.mean(MAP_us,axis=1)
# 	std_us = np.std(MAP_us,axis=1)

# 	with open('MAP_US.csv','a') as outfile:
# 		writer = csv.writer(outfile, delimiter=',', lineterminator='\n')
# 		writer.writerow([fig_name[fig]]);
# 		writer.writerow(['DEEPWALK']);writer.writerow(mean_DEEPWALK);writer.writerow(std_DEEPWALK)
# 		writer.writerow(['n2vA']);writer.writerow(mean_n2vA);writer.writerow(std_n2vA)
# 		writer.writerow(['n2vB']);writer.writerow(mean_n2vB);writer.writerow(std_n2vB)
# 		writer.writerow(['VERSE']);writer.writerow(mean_VERSE);writer.writerow(std_VERSE)
# 		writer.writerow(['heuristic']);writer.writerow(mean_us);writer.writerow(std_us)
	
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# AP/ROC for unsupervised

ratios = [-1,0.5]

for rat in xrange(len(ratios)):

	for fig in xrange(len(fig_name)):

		AP_DEEPWALK=np.zeros((len(dimensions),num_samples))
		AP_n2vA=np.zeros((len(dimensions),num_samples))
		AP_n2vB=np.zeros((len(dimensions),num_samples))
		AP_VERSE=np.zeros((len(dimensions),num_samples))
		AP_us=np.zeros((4,num_samples))

		ROC_DEEPWALK=np.zeros((len(dimensions),num_samples))
		ROC_n2vA=np.zeros((len(dimensions),num_samples))
		ROC_n2vB=np.zeros((len(dimensions),num_samples))
		ROC_VERSE=np.zeros((len(dimensions),num_samples))
		ROC_us=np.zeros((4,num_samples))

		for it1 in xrange(num_samples):

			file_name='SAVER_TE/'+fig_name[fig]+'/graph_and_samples.p'
			parameter_file=open(file_name, 'rb')
			train_digraph = pickle.load(parameter_file)
			test_digraph =  pickle.load(parameter_file)
			parameter_file.close()

			sample_edges = evaluation_measures.sample_edge_new(train_digraph,test_digraph,ratios[rat])
			
			print ("evaluating for heuristic")
			AP, ROC = evaluation_measures.calc_aproc_heu(train_digraph, test_digraph, sample_edges)
			for it2 in xrange(4):
				print (it1,it2)
				AP_us[it2][it1] = AP[it2]
				ROC_us[it2][it1] = ROC[it2]

			print ("evaluating for DEEPWALK")
			for it2 in xrange(len(dimensions)):
				print (it1,it2)
				dim=dimensions[it2]		
				file_name='SAVER_TE/'+fig_name[fig]+'/DEEPWALK_'+str(dim)
				parameter_file=open(file_name, 'rb')
				X = pickle.load(parameter_file)
				parameter_file.close()
				embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
				embedding._X = X
				AP, ROC = evaluation_measures.calc_aproc_us(embedding, X, train_digraph, test_digraph, sample_edges)
				AP_DEEPWALK[it2][it1] = AP
				ROC_DEEPWALK[it2][it1] = ROC

			print ("evaluating for n2vA")
			for it2 in xrange(len(dimensions)):
				print (it1,it2)
				dim=dimensions[it2]		
				file_name='SAVER_TE/'+fig_name[fig]+'/n2vA_'+str(dim)
				parameter_file=open(file_name, 'rb')
				X = pickle.load(parameter_file)
				parameter_file.close()
				embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=4, inout_p=0.5)
				embedding._X = X
				AP, ROC = evaluation_measures.calc_aproc_us(embedding, X, train_digraph, test_digraph, sample_edges)
				AP_n2vA[it2][it1] = AP
				ROC_n2vA[it2][it1] = ROC

			print ("evaluating for n2vB")
			for it2 in xrange(len(dimensions)):
				print (it1,it2)
				dim=dimensions[it2]		
				file_name='SAVER_TE/'+fig_name[fig]+'/n2vB_'+str(dim)
				parameter_file=open(file_name, 'rb')
				X = pickle.load(parameter_file)
				parameter_file.close()
				embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=4)
				embedding._X = X
				AP, ROC = evaluation_measures.calc_aproc_us(embedding, X, train_digraph, test_digraph, sample_edges)
				AP_n2vB[it2][it1] = AP
				ROC_n2vB[it2][it1] = ROC

			print ("evaluating for VERSE")
			for it2 in xrange(len(dimensions)):
				print (it1,it2)
				dim=dimensions[it2]		
				file_name='SAVER_TE/'+fig_name[fig]+'/VERSE_'+str(dim)
				parameter_file=open(file_name, 'rb')
				X = pickle.load(parameter_file)
				parameter_file.close()
				embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
				embedding._X = X
				AP, ROC = evaluation_measures.calc_aproc_us(embedding, X, train_digraph, test_digraph, sample_edges)
				AP_VERSE[it2][it1] = AP
				ROC_VERSE[it2][it1] = ROC

		mean_DEEPWALK = np.mean(AP_DEEPWALK,axis=1)
		std_DEEPWALK = np.std(AP_DEEPWALK,axis=1)
		mean_n2vA = np.mean(AP_n2vA,axis=1)
		std_n2vA = np.std(AP_n2vA,axis=1)
		mean_n2vB = np.mean(AP_n2vB,axis=1)
		std_n2vB = np.std(AP_n2vB,axis=1)
		mean_VERSE = np.mean(AP_VERSE,axis=1)
		std_VERSE = np.std(AP_VERSE,axis=1)
		mean_us = np.mean(AP_us,axis=1)
		std_us = np.std(AP_us,axis=1)

		with open('AP_US.csv','a') as outfile:
			writer = csv.writer(outfile, delimiter=',', lineterminator='\n')
			writer.writerow([fig_name[fig]]);
			writer.writerow([ratios[rat]]);
			writer.writerow(['AP']);
			writer.writerow(['DEEPWALK']);writer.writerow(mean_DEEPWALK);writer.writerow(std_DEEPWALK)
			writer.writerow(['n2vA']);writer.writerow(mean_n2vA);writer.writerow(std_n2vA)
			writer.writerow(['n2vB']);writer.writerow(mean_n2vB);writer.writerow(std_n2vB)
			writer.writerow(['VERSE']);writer.writerow(mean_VERSE);writer.writerow(std_VERSE)
			writer.writerow(['heuristic']);writer.writerow(mean_us);writer.writerow(std_us)

		mean_DEEPWALK = np.mean(ROC_DEEPWALK,axis=1)
		std_DEEPWALK = np.std(ROC_DEEPWALK,axis=1)
		mean_n2vA = np.mean(ROC_n2vA,axis=1)
		std_n2vA = np.std(ROC_n2vA,axis=1)
		mean_n2vB = np.mean(ROC_n2vB,axis=1)
		std_n2vB = np.std(ROC_n2vB,axis=1)
		mean_VERSE = np.mean(ROC_VERSE,axis=1)
		std_VERSE = np.std(ROC_VERSE,axis=1)
		mean_us = np.mean(ROC_us,axis=1)
		std_us = np.std(ROC_us,axis=1)

		with open('ROC_US.csv','a') as outfile:
			writer = csv.writer(outfile, delimiter=',', lineterminator='\n')
			writer.writerow([fig_name[fig]]);
			writer.writerow([ratios[rat]]);
			writer.writerow(['ROC']);
			writer.writerow(['DEEPWALK']);writer.writerow(mean_DEEPWALK);writer.writerow(std_DEEPWALK)
			writer.writerow(['n2vA']);writer.writerow(mean_n2vA);writer.writerow(std_n2vA)
			writer.writerow(['n2vB']);writer.writerow(mean_n2vB);writer.writerow(std_n2vB)
			writer.writerow(['VERSE']);writer.writerow(mean_VERSE);writer.writerow(std_VERSE)
			writer.writerow(['heuristic']);writer.writerow(mean_us);writer.writerow(std_us)

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

