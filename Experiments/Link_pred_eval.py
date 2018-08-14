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

# list_graphs = ['gem/data/CA-AstroPh.txt']
# list_graphs = ['gem/data/facebook_combined.txt']
# list_directed = [False,False,True]
# list_directed = [False]
# fig_name = ['ASTROPH']
# fig_name = ['FACEBOOK']
# dimensions=[2,4,8,16,32,64,128,256]

list_graphs = ['gem/data/karate.edgelist']
list_directed = [False]
fig_name = ['karate']
dimensions=[8]
num_samples=5
ratios=[-1]

for num_graph in xrange(len(list_graphs)):
	
	isDirected = list_directed[num_graph]
	edge_f = list_graphs[num_graph]
	G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
	G = G.to_directed()

	# edge_f1 = 'gem/data/vk_2016'
	# edge_f2 = 'gem/data/vk_2017'
	# isDirected = True
	# fig_name = ['VK']
	# G1 = graph_util.loadGraphFromEdgeListTxt(edge_f1, directed=isDirected)
	# G2 = graph_util.loadGraphFromEdgeListTxt(edge_f2, directed=isDirected)
	# G1 = G1.to_directed()
	# G2 = G2.to_directed()

	print("unsupervised")
	AP_us=np.zeros((4,num_samples))
	ROC_us=np.zeros((4,num_samples))
	MAP_us=np.zeros((4,num_samples))

	for it1 in xrange(num_samples):
		ap,roc,ma = evaluation_measures.evaluate_unsupervised_all(G, is_undirected=True)
		for it2 in xrange(4):
			print (it1,it2)
			AP_us[it2][it1] = ap[it2]
			ROC_us[it2][it1] = roc[it2]
			MAP_us[it2][it1] = ma[it2]

	mean_us1 = np.mean(AP_us,axis=1)
	mean_us2 = np.mean(ROC_us,axis=1)
	mean_us3 = np.mean(MAP_us,axis=1)
	std_us1 = np.std(AP_us,axis=1)
	std_us2 = np.std(ROC_us,axis=1)
	std_us3 = np.std(MAP_us,axis=1)

	print ("US")
	print ("AP")
	print (mean_us1);print (std_us1)
	print ("ROC")
	print (mean_us2);print (std_us2)
	print ("MAP")
	print (mean_us3);print (std_us3)

	print("VERSE")
	AP_VERSE=np.zeros((len(dimensions),num_samples))
	ROC_VERSE=np.zeros((len(dimensions),num_samples))
	MAP_VERSE=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=verse(d=dim, alpha=0.85, threads=3, nsamples=3)
			ap,roc,ma = evaluation_measures.evaluate_supervised(G, embedding, is_undirected=True)
			AP_VERSE[it1][it2]=ap
			ROC_VERSE[it1][it2]=roc
			MAP_VERSE[it1][it2] = ma


	mean_VERSE1 = np.mean(AP_VERSE,axis=1)
	mean_VERSE2 = np.mean(ROC_VERSE,axis=1)
	mean_VERSE3 = np.mean(MAP_VERSE,axis=1)
	std_VERSE1 = np.std(AP_VERSE,axis=1)
	std_VERSE2 = np.std(ROC_VERSE,axis=1)
	std_VERSE3 = np.std(MAP_VERSE,axis=1)

	print ("AP")
	print (mean_VERSE1);print (std_VERSE1)
	print ("ROC")
	print (mean_VERSE2);print (std_VERSE2)
	print ("MAP")
	print (mean_VERSE3);print (std_VERSE3)

	print("LE")
	AP_LE=np.zeros((len(dimensions),num_samples))
	ROC_LE=np.zeros((len(dimensions),num_samples))
	MAP_LE=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=LaplacianEigenmaps(d=dim)
			ap,roc,ma = evaluation_measures.evaluate_supervised(G, embedding, is_undirected=True)
			AP_LE[it1][it2]=ap
			ROC_LE[it1][it2]=roc
			MAP_LE[it1][it2] = ma

	mean_LE1 = np.mean(AP_LE,axis=1)
	mean_LE2 = np.mean(ROC_LE,axis=1)
	mean_LE3 = np.mean(MAP_LE,axis=1)
	std_LE1 = np.std(AP_LE,axis=1)
	std_LE2 = np.std(ROC_LE,axis=1)
	std_LE3 = np.std(MAP_LE,axis=1)
	
	print ("AP")
	print (mean_LE1);print (std_LE1)
	print ("ROC")
	print (mean_LE2);print (std_LE2)
	print ("MAP")
	print (mean_LE3);print (std_LE3)


	print("LE_US")
	AP_LE_US=np.zeros((len(dimensions),num_samples))
	ROC_LE_US=np.zeros((len(dimensions),num_samples))
	MAP_LE_US=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=LaplacianEigenmaps(d=dim)
			ap,roc,ma = evaluation_measures.evaluate_unsupervised_embedding(G, embedding, is_undirected=True)
			AP_LE_US[it1][it2]=ap
			ROC_LE_US[it1][it2]=roc
			MAP_LE_US[it1][it2] = ma

	mean_LE_US1 = np.mean(AP_LE_US,axis=1)
	mean_LE_US2 = np.mean(ROC_LE_US,axis=1)
	mean_LE_US3 = np.mean(MAP_LE_US,axis=1)
	std_LE_US1 = np.std(AP_LE_US,axis=1)
	std_LE_US2 = np.std(ROC_LE_US,axis=1)
	std_LE_US3 = np.std(MAP_LE_US,axis=1)

	print ("AP")
	print (mean_LE_US1);print (std_LE_US1)
	print ("ROC")
	print (mean_LE_US2);print (std_LE_US2)
	print ("MAP")
	print (mean_LE_US3);print (std_LE_US3)

	print("DEEP_WALK")
	AP_deepwalk=np.zeros((len(dimensions),num_samples))
	ROC_deepwalk=np.zeros((len(dimensions),num_samples))
	MAP_deepwalk=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
			ap,roc,ma = evaluation_measures.evaluate_supervised(G, embedding, is_undirected=True)
			AP_deepwalk[it1][it2]=ap
			ROC_deepwalk[it1][it2]=roc
			MAP_deepwalk[it1][it2] = ma

	mean_deepwalk1 = np.mean(AP_deepwalk,axis=1)
	mean_deepwalk2 = np.mean(ROC_deepwalk,axis=1)
	mean_deepwalk3 = np.mean(MAP_deepwalk,axis=1)
	std_deepwalk1 = np.std(AP_deepwalk,axis=1)
	std_deepwalk2 = np.std(ROC_deepwalk,axis=1)
	std_deepwalk3 = np.std(MAP_deepwalk,axis=1)

	print ("AP")
	print (mean_deepwalk1);print (std_deepwalk1)
	print ("ROC")
	print (mean_deepwalk2);print (std_deepwalk2)
	print ("MAP")
	print (mean_deepwalk3);print (std_deepwalk3)

	print("DEEP_WALK_US")
	AP_deepwalk_US=np.zeros((len(dimensions),num_samples))
	ROC_deepwalk_US=np.zeros((len(dimensions),num_samples))
	MAP_deepwalk_US=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
			ap,roc,ma = evaluation_measures.evaluate_unsupervised_embedding(G, embedding, is_undirected=True)
			AP_deepwalk_US[it1][it2]=ap
			ROC_deepwalk_US[it1][it2]=roc
			MAP_deepwalk_US[it1][it2] = ma

	mean_deepwalk_US1 = np.mean(AP_deepwalk_US,axis=1)
	mean_deepwalk_US2 = np.mean(ROC_deepwalk_US,axis=1)
	mean_deepwalk_US3 = np.mean(MAP_deepwalk_US,axis=1)
	std_deepwalk_US1 = np.std(AP_deepwalk_US,axis=1)
	std_deepwalk_US2 = np.std(ROC_deepwalk_US,axis=1)
	std_deepwalk_US3 = np.std(MAP_deepwalk_US,axis=1)

	print ("AP")
	print (mean_deepwalk_US1);print (std_deepwalk_US1)
	print ("ROC")
	print (mean_deepwalk_US2);print (std_deepwalk_US2)
	print ("MAP")
	print (mean_deepwalk_US3);print (std_deepwalk_US3)

	print("n2vA")
	AP_n2vA=np.zeros((len(dimensions),num_samples))
	ROC_n2vA=np.zeros((len(dimensions),num_samples))
	MAP_n2vA=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=4, inout_p=0.5)
			ap,roc,ma = evaluation_measures.evaluate_supervised(G, embedding, is_undirected=True)
			AP_n2vA[it1][it2]=ap
			ROC_n2vA[it1][it2]=roc
			MAP_n2vA[it1][it2] = ma

	mean_n2vA1 = np.mean(AP_n2vA,axis=1)
	mean_n2vA2 = np.mean(ROC_n2vA,axis=1)
	mean_n2vA3 = np.mean(MAP_n2vA,axis=1)
	std_n2vA1 = np.std(AP_n2vA,axis=1)
	std_n2vA2 = np.std(ROC_n2vA,axis=1)
	std_n2vA3 = np.std(MAP_n2vA,axis=1)

	print ("AP")
	print (mean_n2vA1);print (std_n2vA1)
	print ("ROC")
	print (mean_n2vA2);print (std_n2vA2)
	print ("MAP")
	print (mean_n2vA3);print (std_n2vA3)

	print("n2vA_US")
	AP_n2vA_US=np.zeros((len(dimensions),num_samples))
	ROC_n2vA_US=np.zeros((len(dimensions),num_samples))
	MAP_n2vA_US=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=4, inout_p=0.5)
			ap,roc,ma = evaluation_measures.evaluate_unsupervised_embedding(G, embedding, is_undirected=True)
			AP_n2vA_US[it1][it2]=ap
			ROC_n2vA_US[it1][it2]=roc
			MAP_n2vA_US[it1][it2] = ma

	mean_n2vA_US1 = np.mean(AP_n2vA_US,axis=1)
	mean_n2vA_US2 = np.mean(ROC_n2vA_US,axis=1)
	mean_n2vA_US3 = np.mean(MAP_n2vA_US,axis=1)
	std_n2vA_US1 = np.std(AP_n2vA_US,axis=1)
	std_n2vA_US2 = np.std(ROC_n2vA_US,axis=1)
	std_n2vA_US3 = np.std(MAP_n2vA_US,axis=1)

	print ("AP")
	print (mean_n2vA_US1);print (std_n2vA_US1)
	print ("ROC")
	print (mean_n2vA_US2);print (std_n2vA_US2)
	print ("MAP")
	print (mean_n2vA_US3);print (std_n2vA_US3)

	
	print("n2vB")
	AP_n2vB=np.zeros((len(dimensions),num_samples))
	ROC_n2vB=np.zeros((len(dimensions),num_samples))
	MAP_n2vB=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			# embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=4)
			ap,roc,ma = evaluation_measures.evaluate_supervised(G, embedding, is_undirected=True)
			AP_n2vB[it1][it2]=ap
			ROC_n2vB[it1][it2]=roc
			MAP_n2vB[it1][it2] = ma

	mean_n2vB1 = np.mean(AP_n2vB,axis=1)
	mean_n2vB2 = np.mean(ROC_n2vB,axis=1)
	mean_n2vB3 = np.mean(MAP_n2vB,axis=1)
	std_n2vB1 = np.std(AP_n2vB,axis=1)
	std_n2vB2 = np.std(ROC_n2vB,axis=1)
	std_n2vB3 = np.std(MAP_n2vB,axis=1)

	print ("AP")
	print (mean_n2vB1);print (std_n2vB1)
	print ("ROC")
	print (mean_n2vB2);print (std_n2vB2)
	print ("MAP")
	print (mean_n2vB3);print (std_n2vB3)

	print("n2vB_US")
	AP_n2vB_US=np.zeros((len(dimensions),num_samples))
	ROC_n2vB_US=np.zeros((len(dimensions),num_samples))
	MAP_n2vB_US=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=4)
			ap,roc,ma = evaluation_measures.evaluate_unsupervised_embedding(G, embedding, is_undirected=True)
			AP_n2vB_US[it1][it2]=ap
			ROC_n2vB_US[it1][it2]=roc
			MAP_n2vB_US[it1][it2] = ma

	mean_n2vB_US1 = np.mean(AP_n2vB_US,axis=1)
	mean_n2vB_US2 = np.mean(ROC_n2vB_US,axis=1)
	mean_n2vB_US3 = np.mean(MAP_n2vB_US,axis=1)
	std_n2vB_US1 = np.std(AP_n2vB_US,axis=1)
	std_n2vB_US2 = np.std(ROC_n2vB_US,axis=1)
	std_n2vB_US3 = np.std(MAP_n2vB_US,axis=1)

	print ("AP")
	print (mean_n2vB_US1);print (std_n2vB_US1)
	print ("ROC")
	print (mean_n2vB_US2);print (std_n2vB_US2)
	print ("MAP")
	print (mean_n2vB_US3);print (std_n2vB_US3)


	print("HOPE")
	AP_HOPE=np.zeros((len(dimensions),num_samples))
	ROC_HOPE=np.zeros((len(dimensions),num_samples))
	MAP_HOPE=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=HOPE(d=dim, beta=0.01)
			ap,roc,ma = evaluation_measures.evaluate_supervised(G, embedding, is_undirected=True)
			AP_HOPE[it1][it2]=ap
			ROC_HOPE[it1][it2]=roc
			MAP_HOPE[it1][it2] = ma

	mean_HOPE1 = np.mean(AP_HOPE,axis=1)
	mean_HOPE2 = np.mean(ROC_HOPE,axis=1)
	mean_HOPE3 = np.mean(MAP_HOPE,axis=1)
	std_HOPE1 = np.std(AP_HOPE,axis=1)
	std_HOPE2 = np.std(ROC_HOPE,axis=1)
	std_HOPE3 = np.std(MAP_HOPE,axis=1)

	print ("AP")
	print (mean_HOPE1);print (std_HOPE1)
	print ("ROC")
	print (mean_HOPE2);print (std_HOPE2)
	print ("MAP")
	print (mean_HOPE3);print (std_HOPE3)

	print("HOPE_US")
	AP_HOPE_US=np.zeros((len(dimensions),num_samples))
	ROC_HOPE_US=np.zeros((len(dimensions),num_samples))
	MAP_HOPE_US=np.zeros((len(dimensions),num_samples))
	for it1 in range(len(dimensions)):
		dim=dimensions[it1]
		for it2 in xrange(num_samples):
			print (it1,it2)
			embedding=HOPE(d=dim, beta=0.01)
			ap,roc,ma = evaluation_measures.evaluate_unsupervised_embedding(G, embedding, is_undirected=True)
			AP_HOPE_US[it1][it2]=ap
			ROC_HOPE_US[it1][it2]=roc
			MAP_HOPE_US[it1][it2] = ma

	mean_HOPE_US1 = np.mean(AP_HOPE_US,axis=1)
	mean_HOPE_US2 = np.mean(ROC_HOPE_US,axis=1)
	mean_HOPE_US3 = np.mean(MAP_HOPE_US,axis=1)
	std_HOPE_US1 = np.std(AP_HOPE_US,axis=1)
	std_HOPE_US2 = np.std(ROC_HOPE_US,axis=1)
	std_HOPE_US3 = np.std(MAP_HOPE_US,axis=1)

	print ("AP")
	print (mean_HOPE_US1);print (std_HOPE_US1)
	print ("ROC")
	print (mean_HOPE_US2);print (std_HOPE_US2)
	print ("MAP")
	print (mean_HOPE_US3);print (std_HOPE_US3)

	sys.stdout = open("results.txt", "w")
	print (edge_f)
	print ("US")
	print ("AP");print (mean_us1);print (std_us1)
	print ("ROC");print (mean_us2);print (std_us2)
	print ("MAP");print (mean_us3);print (std_us3)
	print ("LE")
	print ("AP");print (mean_LE1);print (std_LE1)
	print ("ROC");print (mean_LE2);print (std_LE2)
	print ("MAP");print (mean_LE3);print (std_LE3)
	print ("LE_US")
	print ("AP");print (mean_LE_US1);print (std_LE_US1)
	print ("ROC");print (mean_LE_US2);print (std_LE_US2)
	print ("MAP");print (mean_LE_US3);print (std_LE_US3)
	print ("HOPE")
	print ("AP");print (mean_HOPE1);print (std_HOPE1)
	print ("ROC");print (mean_HOPE2);print (std_HOPE2)
	print ("MAP");print (mean_HOPE3);print (std_HOPE3)
	print ("HOPE_US")
	print ("AP");print (mean_HOPE_US1);print (std_HOPE_US1)
	print ("ROC");print (mean_HOPE_US2);print (std_HOPE_US2)
	print ("MAP");print (mean_HOPE_US3);print (std_HOPE_US3)
	print ("deepwalk")
	print ("AP");print (mean_deepwalk1);print (std_deepwalk1)
	print ("ROC");print (mean_deepwalk2);print (std_deepwalk2)
	print ("MAP");print (mean_deepwalk3);print (std_deepwalk3)
	print ("deepwalk_US")
	print ("AP");print (mean_deepwalk_US1);print (std_deepwalk_US1)
	print ("ROC");print (mean_deepwalk_US2);print (std_deepwalk_US2)
	print ("MAP");print (mean_deepwalk_US3);print (std_deepwalk_US3)
	print ("n2vA")
	print ("AP");print (mean_n2vA1);print (std_n2vA1)
	print ("ROC");print (mean_n2vA2);print (std_n2vA2)
	print ("MAP");print (mean_n2vA3);print (std_n2vA3)
	print ("n2vA_US")
	print ("AP");print (mean_n2vA_US1);print (std_n2vA_US1)
	print ("ROC");print (mean_n2vA_US2);print (std_n2vA_US2)
	print ("MAP");print (mean_n2vA_US3);print (std_n2vA_US3)
	print ("n2vB")
	print ("AP");print (mean_n2vB1);print (std_n2vB1)
	print ("ROC");print (mean_n2vB2);print (std_n2vB2)
	print ("MAP");print (mean_n2vB3);print (std_n2vB3)
	print ("n2vB_US")
	print ("AP");print (mean_n2vB_US1);print (std_n2vB_US1)
	print ("ROC");print (mean_n2vB_US2);print (std_n2vB_US2)
	print ("MAP");print (mean_n2vB_US3);print (std_n2vB_US3)	
	
	# n = len(dimensions)
	# plt_arr = np.arange(n)
	# plt.figure(10)
	# plt.errorbar(plt_arr, np.asarray([mean_us1[0]]*n), np.asarray([std_us1[0]]*n), marker='o', label="CN")
	# plt.errorbar(plt_arr, np.asarray([mean_us1[1]]*n), np.asarray([std_us1[1]]*n), marker='o', label="JC")
	# plt.errorbar(plt_arr, np.asarray([mean_us1[2]]*n), np.asarray([std_us1[2]]*n), marker='o', label="PA")
	# plt.errorbar(plt_arr, np.asarray([mean_us1[3]]*n), np.asarray([std_us1[3]]*n), marker='o', label="AA")
	# plt.errorbar(plt_arr, np.asarray([mean_us1[4]]*n), np.asarray([std_us1[4]]*n), marker='o', label="RANDOM")
	# plt.errorbar(plt_arr, mean_LE1, std_LE1, marker='^', label="LE")
	# plt.errorbar(plt_arr, mean_HOPE1, std_HOPE1, marker='x', label="HOPE")
	# plt.errorbar(plt_arr, mean_n2v1, std_n2v1, marker='v', label="n2v")
	# plt.errorbar(plt_arr, mean_LE_US1, std_LE_US1, marker='^', label="LE_US")
	# plt.errorbar(plt_arr, mean_HOPE_US1, std_HOPE_US1, marker='x', label="HOPE_US")
	# plt.errorbar(plt_arr, mean_n2v_US1, std_n2v_US1, marker='v', label="n2v_US")
	# plt.xticks(plt_arr, dimensions)
	# plt.legend(loc=(1.1,0))
	# # plt.show()
	# plt.savefig("PLOTS/"+"AP_"+fig_name[num_graph],bbox_inches="tight")

	# plt.figure()
	# plt.errorbar(plt_arr, np.asarray([mean_us2[0]]*n), np.asarray([std_us2[0]]*n), marker='o', label="CN")
	# plt.errorbar(plt_arr, np.asarray([mean_us2[1]]*n), np.asarray([std_us2[1]]*n), marker='o', label="JC")
	# plt.errorbar(plt_arr, np.asarray([mean_us2[2]]*n), np.asarray([std_us2[2]]*n), marker='o', label="PA")
	# plt.errorbar(plt_arr, np.asarray([mean_us2[3]]*n), np.asarray([std_us2[3]]*n), marker='o', label="AA")
	# plt.errorbar(plt_arr, np.asarray([mean_us1[4]]*n), np.asarray([std_us1[4]]*n), marker='o', label="RANDOM")
	# plt.errorbar(plt_arr, mean_LE2, std_LE2, marker='^', label="LE")
	# plt.errorbar(plt_arr, mean_HOPE2, std_HOPE2, marker='x', label="HOPE")
	# plt.errorbar(plt_arr, mean_n2v2, std_n2v2, marker='v', label="n2v")
	# plt.errorbar(plt_arr, mean_LE_US2, std_LE_US2, marker='^', label="LE_US")
	# plt.errorbar(plt_arr, mean_HOPE_US2, std_HOPE_US2, marker='x', label="HOPE_US")
	# plt.errorbar(plt_arr, mean_n2v_US2, std_n2v_US2, marker='v', label="n2v_US")
	# plt.xticks(plt_arr, dimensions)
	# plt.legend(loc=(1.1,0))
	# # plt.show()
	# plt.savefig("PLOTS/"+"ROC_"+fig_name[num_graph],bbox_inches="tight")
	
	# plt.figure()
	# plt.errorbar(plt_arr, np.asarray([mean_us3[0]]*n), np.asarray([std_us3[0]]*n), marker='o', label="CN")
	# plt.errorbar(plt_arr, np.asarray([mean_us3[1]]*n), np.asarray([std_us3[1]]*n), marker='o', label="JC")
	# plt.errorbar(plt_arr, np.asarray([mean_us3[2]]*n), np.asarray([std_us3[2]]*n), marker='o', label="PA")
	# plt.errorbar(plt_arr, np.asarray([mean_us3[3]]*n), np.asarray([std_us3[3]]*n), marker='o', label="AA")
	# plt.errorbar(plt_arr, np.asarray([mean_us1[4]]*n), np.asarray([std_us1[4]]*n), marker='o', label="RANDOM")
	# plt.errorbar(plt_arr, mean_LE3, std_LE3, marker='^', label="LE")
	# plt.errorbar(plt_arr, mean_HOPE3, std_HOPE3, marker='x', label="HOPE")
	# plt.errorbar(plt_arr, mean_n2v3, std_n2v3, marker='v', label="n2v")
	# plt.errorbar(plt_arr, mean_LE_US3, std_LE_US3, marker='^', label="LE_US")
	# plt.errorbar(plt_arr, mean_HOPE_US3, std_HOPE_US3, marker='x', label="HOPE_US")
	# plt.errorbar(plt_arr, mean_n2v_US3, std_n2v_US3, marker='v', label="n2v_US")
	# plt.xticks(plt_arr, dimensions)
	# plt.legend(loc=(1.1,0))
	# # plt.show()
	# plt.savefig("PLOTS/"+"MAP_"+fig_name[num_graph],bbox_inches="tight")