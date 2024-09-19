"""
# word_to_ignore
ignore = ["would", using, new, Be, sometimes. It, if]

---------------------------------------------------------------------------------------
# To-do:
# Break it up by runs. (Still do not understand how to. :( )
Notes from Jane:
	1. "Delivering/Delivered to <hutch id>"
	2. Times between 6-9 am and 6-9 pm. Days can be narrowed down by run schedule
	3. mJ as unit, pulse intensity(?)

"""



# from numba import jit, cuda
# import os
# print (os.getcwd())
# exit()
import requests
import time
import datetime
import numpy as np
import pandas as pd
import json
import copy
import re
import string
import random
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# nltk things
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
for corpus_name in ["words", "stopwords", "names"]:
    try:
        nltk.data.find('corpora/%s'%corpus_name)
    except LookupError:
        # nltk.download('punkt')
        print("Downloading: ", corpus_name)
        nltk.download(corpus_name)

from nltk.corpus import words, stopwords, names, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

import networkx as nx

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

# import seaborn as sns
# sns.set_palette("colorblind")
import math
import re
from collections import Counter, OrderedDict
import itertools
from itertools import chain

from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

import torch
# from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


## --------------- Set 1: Functions for Text processing ------------------------
# Divided into two parts

# 1. Functions for data clean up
#     remove numbers, remove punctuations, email ids, urls
# 2. Functions for the similarity metrics


# ----------------------------------------------------------------------------------
# ------------------------------- Functions ----------------------------------------
# ----------------------------------------------------------------------------------

# ------------------------- Functions for data clean up ----------------------------

def is_number(s):
    try:
        float(s)
        return(True)
    except ValueError:
        return(False)

def fix_colons_for_pvs(given_text_as_string):

    # replaces colons between non-whitespace characters, with underscore.
    # other colons are replaced with space

    string_to_return = given_text_as_string
    string_to_return = re.sub("\\s:", " ", string_to_return)
    string_to_return = re.sub(":\\s", " ", string_to_return)

    string_to_return = re.sub(":", "_" ,string_to_return)
    return(string_to_return)

def remove_email_ids(given_text_as_string):
	regex_email_id = "\S*@\S*\s?"
	# email_id_list = re.findall(regex, given_text_as_string)
	given_text_email_ids_removed = re.sub(regex_email_id, " ", given_text_as_string)
	return(given_text_email_ids_removed)

def remove_urls(given_text_as_string):
	regex_url = "(?P<url>https?://[^\s]+)"
	# url_list = re.findall(regex, given_text_as_string)
	given_text_urls_removed = re.sub(regex_url, " ", given_text_as_string)
	return(given_text_urls_removed)

def extract_urls(given_text_as_string):
	regex_url = "(?P<url>https?://[^\s]+)"
	list_of_extracted_urls = re.findall(regex_url, given_text_as_string)
	return(list_of_extracted_urls)

def remove_objects(given_text_as_string):
	regex_object = '<[^>]+>'
	given_text_objects_removed = re.sub(regex_object, " ", given_text_as_string, 0)
	return(given_text_objects_removed)

def remove_punc_and_word_tokenize_RR(given_text_as_string):
	regex_punctuations = r"[^\w\s\:]" # to remove all punctuations except :

	chars_to_remove = ["\n", "\t"] # other similar things to remove
	regex_specific_punctuations = '[' + re.escape(''.join(chars_to_remove)) + ']'

	temp = re.sub(regex_punctuations, " ", given_text_as_string, 0)
	given_text_nopunc = re.sub(regex_specific_punctuations, " ", temp, 0)
	# We split the text based on n number of whitespaces
	given_text_nopunc_as_wordlist = re.split(r' +', given_text_nopunc)
	try: given_text_nopunc_as_wordlist.remove("") # Sometimes there would be, due to our algorithm.
	except: ValueError
	# NOTE: The last two steps are significant, as they mean that we can use as many " "s we need for different reasons, it will be okay!
	return(given_text_nopunc_as_wordlist)

# NOTE: We remove the stopwords from lists, using remove repeatedly to get all instances
	# A tweak: To reduce runtime, we subset the list of stopwords present in our text first
def remove_stopwords_from_wordlist_RR(given_text_as_wordlist):
	# Subset the list of stopwords present in our text
	words_to_remove = list(set(stopwords.words("english")) & set(given_text_as_wordlist))
	for word in words_to_remove:
		while(True):
			try: given_text_as_wordlist.remove(word)
			except: break
	# Now the given_text_as_wordlist do not have any more stopwords
	return(given_text_as_wordlist)

# -------------------- Functions for the similarity metrics ------------------------
# ----------------------------------------------------------------------------------

def jaccard_similarity(vec1, vec2): # input
	if isinstance(vec1, set):
		common_set = set(vec1.keys()) & set(vec2.keys())
		union_set = set(vec1.keys()) | set(vec2.keys())
	elif isinstance(vec1, list):
		common_set = set(vec1) & set(vec2)
		union_set = set(vec1) | set(vec2)

	js = len(common_set) / len(union_set)

	return(js)

def cosine_similarity(vec1, vec2): # input
	common_set = set(vec1.keys()) & set(vec2.keys())
	dot_product = sum([vec1[i] * vec2[i] for i in common_set]) # dot product of the vectors

	# need to divide the dot product of the vectors, by the magnitudes of each vector
	magnitude_vec1 = math.sqrt(sum([vec1[i]**2 for i in list(vec1.keys())]))
	magnitude_vec2 = math.sqrt(sum([vec2[i]**2 for i in list(vec2.keys())]))

	denominator = magnitude_vec1 * magnitude_vec2
	if not denominator:
		return 0.0
	else:
		return float(dot_product) / denominator

def relative_entropy(vec1, vec2): # asymmetric, distance of vec1 wrt vec2
	if isinstance(vec1, set):
		union_set = set(vec1.keys()) | set(vec2.keys())
		# NOTE: plus 0.01 with vecs, to deal with zero counts
		vec1_values_in_order = np.array([vec1[key] for key in union_set]) + 1e-2
		vec2_values_in_order = np.array([vec2[key] for key in union_set]) + 1e-2
		p1 = vec1_values_in_order/np.sum(vec1_values_in_order)
		p2 = vec2_values_in_order/np.sum(vec2_values_in_order)
	else:
		p1 = vec1
		p2 = vec2

	# Laplace smoothing for 0 probs
	# for p_x in [p1, p2]:
	if p1[~(p1 > 0)].shape[0] > 0:
		p1 = p1 + 1/p1.shape[0]/1000
		p1 = p1/np.sum(p1)

	if p2[~(p2 > 0)].shape[0] > 0:
		p2 = p2 + 1/p2.shape[0]/1000
		p2 = p2/np.sum(p2)
		# print(p_x)
	re = np.sum(p1*np.log2(p1/p2))
	return (re)

# max(a,b) = (a + b + |a - b|)/2
# min(a,b) = (a + b - |a - b|)/2
def overlapping_index_pastore_2019(p1, p2):
	# we want the min here.
	min_p1_p2 = (p1 + p2 - np.abs(p1 - p2))/2
	overlapping_index = np.sum(min_p1_p2)
		# simply sum, because the discrete discrete have default dx = 1
	return(overlapping_index)

def distributional_similarity_RR(p1, p2):
	# we want the min here.
	# min_p1_p2 = (p1 + p2 - np.abs(p1 - p2))/2
	overlapping_index = overlapping_index_pastore_2019(p1, p2)

	dis_sim_RR = overlapping_index/(2-overlapping_index)
	return(overlapping_index)

# def sim_score_based_on_re():


# Another option to try:
# Actually, let us use Bhattacharya coeff, instead of the distance metric.
# But the distance metric is also going to be useful, especially in comparing with the RE values.
# A test of the rankings based on information contained in the distributions:
	# If the rankings are roughly the same for KLD and BDis and BC, it would provide great support for my information-based perspective.
# Another thing to consider. USe the operation needed to go from BDis to BCoeff, to go from RE (the distance) to a similarity score

def bhattacharyya_coeff_for_vecs(vec1, vec2): # asymmetric, distance of vec1 wrt vec2
	union_set = set(vec1.keys()) | set(vec2.keys())
	# NOTE: A BIIIG difference from RE
		# no problems with zero counts
	vec1_values_in_order = np.array([vec1[key] for key in union_set])
	vec2_values_in_order = np.array([vec2[key] for key in union_set])
	p1 = vec1_values_in_order/np.sum(vec1_values_in_order)
	p2 = vec2_values_in_order/np.sum(vec2_values_in_order)

	bhattacharyya_coeff = np.sum(np.sqrt(p1*p2))
	return (bhattacharyya_coeff)

def bhattacharyya_distance(vec1, vec2): # asymmetric, distance of vec1 wrt vec2
	union_set = set(vec1.keys()) | set(vec2.keys())
	# NOTE: A BIIIG difference from RE
		# no problems with zero counts
	vec1_values_in_order = np.array([vec1[key] for key in union_set])
	vec2_values_in_order = np.array([vec2[key] for key in union_set])
	p1 = vec1_values_in_order/np.sum(vec1_values_in_order)
	p2 = vec2_values_in_order/np.sum(vec2_values_in_order)

	bhattacharyya_coeff = np.sum(np.sqrt(p1*p2))
	return (-np.log(bhattacharyya_coeff))

# Jensen-Shannon Divergence: The equation I liked -- in Wang and Dong (lol) 2020

# def relative_entropy_old(vec1, vec2): # symmetric
# 	# What was the problem? Adding +1 is too much for zero-mention items as important words are also mentioned very few times.

# 	union_set = set(vec1.keys()) | set(vec2.keys())
# 	# NOTE: plus 1 with vecs, to deal with zero counts
# 	vec1_values_in_order = np.array([vec1[key] for key in union_set]) + 1
# 	vec2_values_in_order = np.array([vec2[key] for key in union_set]) + 1
# 	p1 = vec1_values_in_order/np.sum(vec1_values_in_order)
# 	p2 = vec2_values_in_order/np.sum(vec2_values_in_order)

# 	# we use the symmetric version, which is comparable to Jeffrey's divergence?.
# 	re = ( np.sum(p1*np.log2(p1/p2)) + np.sum(p2*np.log2(p2/p1)) )/2
# 	return (re)

# def relative_entropy_test3(vec1, vec2): # asymmetric, distance of vec1 wrt vec2
# 	union_set = set(vec1.keys()) | set(vec2.keys())
# 	# NOTE: plus 1 with vecs, to deal with zero counts
# 	vec1_values_in_order = np.array([vec1[key] for key in union_set]) + 1
# 	vec2_values_in_order = np.array([vec2[key] for key in union_set]) + 1
# 	p1 = vec1_values_in_order/np.sum(vec1_values_in_order)
# 	p2 = vec2_values_in_order/np.sum(vec2_values_in_order)

# 	# we use the symmetric version, which is comparable to Jeffrey's divergence?.
# 	re = np.sum(p1*np.log2(p1/p2))
# 	return (re)

# def relative_entropy_from_dists(p1, p2):
# 	# we use the symmetric version, which is comparable to Jeffrey's divergence(?).
# 	re = ( np.sum(p1*np.log2(p1/p2)) + np.sum(p2*np.log2(p2/p1)) )/2
# 	return (re)

# ----------------------- Unclassified Functions -----------------------------------
# ----------------------------------------------------------------------------------

WORD = re.compile(r"\w+")
def text_to_vector(text): # From raw text as strings (MEANT ONLY FOR TESTS)
	words = WORD.findall(text)
	# print(words)
	return Counter(words)

def vec_dict_to_prob_dict(vec):
	sum_vec = sum(vec.values())
	p = {key:(vec[key]/sum_vec) for key in vec.keys()}
	return(p)

def softmax(x):
    x = x - np.max(x)
    p = np.exp(x)/np.sum(np.exp(x))
    return(p)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# -----------------------Set 2: Natural Language Processing
## # ----------------------------------------------------------------------------------
# 1. min_word_cutoff filtering
# 2. LSA ( = TF-IDF vectorization + SVD)
# 3. Classifying entries relevant to tuning, based on cosine similarity, and
# 4. Classifying tuning parameters used in process ("likely knobs" in code) based on cosine similarity.


def filter_elog_df_min_word_required(sdf, min_number_of_words_required = 5):
    # -------------------------> MINIMUM WORD REQUIREMENT <-----------------------------
    # ----------------------------------------------------------------------------------

    # Here we create a first knob to control the information used in our models:
        # A minimum number of words in entries for them to be included.
        # Reason: We are using distributional methods that would work better wirh higher number of many words.
        # Another way to look at it: The more information we have, the more reliable estimated data distributions would be.
        # Note: About 50% of the elog entries goes away with 10, less with 5.

        # In summary, using a lower minimum for the number of words in an entry, would be more inclusive (which we want).
        # But this more information would come at a cost of loss of accuracy/reliability of the models.

    incoming_sdf_docs = sdf.shape[0]
    sdf = sdf[sdf.all_text_as_vec.apply(len) > min_number_of_words_required]
    print("# -------------------------> MINIMUM WORD REQUIREMENT <-----------------------")
    print("min_number_of_words_required = ", min_number_of_words_required)
    print("Number of documents in sdf: %i"%sdf.shape[0])
    print("Number of documents removed due to min word requirement: %i" %(incoming_sdf_docs - sdf.shape[0]))
    all_words_in_sdf = list(chain(*sdf.all_text_in_words))
    unique_words_in_sdf = set(all_words_in_sdf)
    print("Number of words in sdf: %i"%len(all_words_in_sdf))
    print("Number of unique words in sdf: %i"%len(unique_words_in_sdf))
    return(sdf)

def lsa_of_set_of_documents_as_texts(all_docs_as_text, num_topics_svd = 100):
    # We receive a pandas series with the text from all documents.
    # Then we calculate a TF-IDF document-term matrix
    # Then we perform LSA on doc-term matrix to get back the three matrices: U, S, VT
        # We can regulate the number of topics (k) for:
        # (1) trancating the S matrix containing the singular values.
        # (2) And convert the doc-term matrix to doc-topic matrices of diffent sizes.

    # ---------------------------------> (1) TF-IDF Vectorize <-------------------------------
    # ----------------------------------------------------------------------------------------

    # Important NOTE:
        # We exclude words that are not mentioned at least 10 times over the 14 years.

    # Also note, if we use the stop_words parameter, it will do their own stop word filtering.
        # But we already did it, so not needed any more.

    time_start = time.time()
    vectorizer = TfidfVectorizer(stop_words='english',
        # max_features= 1000, # keep top 1000 terms
        min_df = 10/all_docs_as_text.shape[0], # takes a propotion, we adapted to specify minimum num of mentions
        smooth_idf=True)

    # Get tf-idf representation of the dataset
    all_docs_as_ti_matrix = vectorizer.fit_transform(all_docs_as_text)
    tfidf_feats = vectorizer.get_feature_names_out()

    print("# ---------------------------------> TF-IDF Vectorize <-----------------------")
    print("features of vectors:", tfidf_feats)
    # print(vectorizer.idf_)
    print("Shape of Document-term matrix:", all_docs_as_ti_matrix.shape)
    print("Time taken = %1.2f seconds"%(time.time() - time_start))


    # --------------------------> (2) SINGULAR VALUE DECOMPOSITION <--------------------------
    # ----------------------------------------------------------------------------------------
    time_start = time.time()
    # using the svds package from scipy.sparse.linalg
    A_from_data = all_docs_as_ti_matrix.T
    Uf, Sf, VTf = svds(A_from_data, k = num_topics_svd) # f for flipped
    print("***NOTE: Scipy sparse svd gives the S in ascending order. So we will flip U, S, V ourselves.")

    print("# --------------------------> SINGULAR VALUE DECOMPOSITION <------------------")
    S = np.diag(Sf[::-1]) # ascending to descending
    U = Uf[:, ::-1] # Now, the columns need to be in reversed order
    VT = VTf[::-1, :] # Now, the rows need to be in reversed order
    # print("SciPy sparse SVD, (U, S, V^T): \n{0},\n{1},\n{2}".format(u, s, vt))
    print("Shapes of U, S, and VT:", U.shape, S.shape, VT.shape)
    print("Time taken = %1.2f seconds"%(time.time() - time_start))
    return(U, S, VT)

def identify_likely_tuning_knobs(sdf, VT, critical_similarity = 0.30):
    print(critical_similarity)
# ----------> i.e., which knobs were discussed? <-------------------
# ----------------------------------------------------------------------------------------
    # Define a threshold for relevance based on cosine similarity
    # We will do fancy things with this threshold if Allah permits.

    # We leave an option for separate threshold for the whole and the piecewise article.
    critical_similarity_wt = critical_similarity # for whole text
    critical_similarity_tt = critical_similarity # for topic texts

    # separate out logs and articles texts from VT.
    VT_all_logs = VT[:, :-28].T
    VT_whole_text = VT[:, -1].reshape([100,1]).T
    VT_topic_texts = VT[:, -28:-1].T

    # ---------------> CALCULATE SIMILARITY BETWEEN ENTRIES AND TUNING ARTICLE <--------------

    ## We calculated two similarity metrics: (a) The cosine similarity and (b) Bhattacharyya Co-efficient
        ## NOTE: But, we use only the COSINE SIMILARITY for our analysis in the notebook.
        ## But BC offers an equally good alternative, among other options. We chose CS for simplicity.

    # ------------------------- Cosine Similarity ---------------------------------------------

    # First, we convert the vectors to unit vectors, denoted by _uv.
    # then, the dot product of the unit vectors is the cosine similarity.
    VT_logs_uv = normalize(VT_all_logs, norm = "l2", axis = 1)
    VT_whole_text_uv = normalize(VT_whole_text, norm = "l2", axis = 1)
    VT_topic_texts_uv = normalize(VT_topic_texts, norm = "l2", axis = 1)

    # Cosine similarity between the ELOG ENTRIES and WHOLE TUNING ARTICLE
        # So, for EACH entry, ONE SIMILARITY VALUE
    cs_VT_logs_wt = VT_logs_uv.dot(VT_whole_text_uv.T)

    print("Cosine sim matrix of all entries with the whole tuning article:\n", cs_VT_logs_wt.shape)

    # Cosine similarity between the ELOG ENTRIES and the 27 PARAMETERS as TOPICS of TUNING ARTICLE
        # So, for EACH entry, an ARRAY of 27 SIMILARITY VALUES
    cs_VT_logs_tt = VT_logs_uv.dot(VT_topic_texts_uv.T)
    print("Cosine sim matrix of all entries with the 27 tuning parameters or topics:\n", cs_VT_logs_tt.shape)

    # Compute Cosine Similarity of the entries with the detailed tuning text. We will use this one to subset entries relevant to the tuning task
    sdf["cs_lsa_wt"] = cs_VT_logs_wt

    # Now the knob-wise similarity --> identify likely knobs
    sdf["likely_knobs_lsa"] = np.empty((len(sdf), 0)).tolist() # empty placeholder
    knob_hits_in_series = np.where(cs_VT_logs_tt > critical_similarity_tt)

    # We get the likely knobs as a series that we organize by doc.
    knob_hits_in_series = pd.DataFrame(np.transpose(knob_hits_in_series), columns = ["doc_ids", "topic_match"])
    knob_hits_grouped_by_docs = knob_hits_in_series.groupby(by = "doc_ids")["topic_match"].apply(list)
    col_index_for_likely_knobs_lsa = np.where(sdf.columns == "likely_knobs_lsa")[0][0]
    sdf.iloc[list(knob_hits_grouped_by_docs.index), col_index_for_likely_knobs_lsa] = knob_hits_grouped_by_docs
    return(sdf)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# ----------------------- Set 3: Network Analysis and Visualizations ---------------
## # ----------------------------------------------------------------------------------

# 1. for networks
# 2. for subsetting entries most relevent to tuning task, based on similarity to the whole tuning article.
# 3. for determining the likely knobs

# 1. Adjacency matrices\
#     -- Adjacency matrices from edge data (0 or 1 for edge weights)\
#     -- Adjacency matrices based on similarity scores between two nodes (continous edge weights between 0 and 1)
#     -- ...
# 2. Community Detection\
#     -- G-N algorithm\
#     -- Spectral Clustering
# Network visualizations (all together)
# Bootstrapping tests

# ----------------------------------------------------------------------------------------
# -------------------------------- Adjacency Matrices ------------------------------------
# ----------------------------------------------------------------------------------------
def return_adj_matrix(list_of_connected_points, num_rows = 27):
	pairs = list(itertools.combinations(list_of_connected_points, r = 2))
	A = np.zeros([num_rows, num_rows])
	for x in pairs:
		A[x] = 1 # e.g., A[2,3] = 1
		A[x[::-1]] = 1 # A[3,2] = 1
	return(A)

def return_adj_matrix_from_sim_scores(list_of_sim_scores):
	num_rows = len(list_of_sim_scores)
	pairs = list(itertools.combinations(range(num_rows), r = 2))
	A = np.zeros([num_rows, num_rows])
	for x, y in pairs:
		A[x, y] = A[y, x] = np.sqrt(list_of_sim_scores[x] * list_of_sim_scores[y])
	return(A)

# Takes a list of weights for nodes and returns edge weights as an adj matrix by multiplying node weights
    # A limiting case is using a threshold to assign 0s or 1s in adj matrix
    # So this approach removes the need to set a threshold.
def return_adj_matrix_by_multiplying_node_weights(node_weights):
    node_weights = node_weights.reshape([node_weights.shape[0], 1])
    adj_matrix = node_weights @ node_weights.T
    np.fill_diagonal(adj_matrix, 0)
    return(adj_matrix)


# ----------------------------------------------------------------------------------------
# -------------------------------- Community detection -----------------------------------
# ----------------------------------------------------------------------------------------

# -----------------------> Girvan-Newman community detection <----------------------------
# ----------------------------------------------------------------------------------------

def most_central_edge_weighted(G):
    centrality = nx.edge_betweenness_centrality(G, weight="weight")
    return max(centrality, key=centrality.get)

def most_central_edge_unweighted(G):
    centrality = nx.edge_betweenness_centrality(G)
    return max(centrality, key=centrality.get)

def communities_and_modularities_girvan_newman(G_x):
    # find communities using girvan-newman algorithm
    communities = list(nx.community.girvan_newman(G_x, most_valuable_edge=most_central_edge_weighted))
    # print(communities[0])

    # Modularity -> measures the strength of division of a network into modules
    modularity_df = pd.DataFrame(
        [
            [k + 1, communities[k], nx.community.modularity(G_x, communities[k], weight ="weight")]
            for k in range(len(communities))
        ],
        columns=["k", "community_elements", "modularity"],
    )
    return(communities, modularity_df)


# -----------------------------> Spectral CLUSTERING <------------------------------------
# ----------------------------------------------------------------------------------------

def community_detection_by_spectral_clustering(G, A, num_clusters, set_seed = 2):
    np.random.seed(set_seed)
    D = np.diag([v for k, v in G.degree(weight = "weight")])
    # D_alt = np.diag(np.sum(A, axis = 0))
    L = D - A

    eigenvalues, eigenvectors = np.linalg.eig(L)

    sorted_indices_of_eigenvalues = np.argsort(eigenvalues)
    # sorted_eigenvalues = eigenvalues[sorted_indices_of_eigenvalues]

    ## ------------- Spectral Clustering using sklearn --------------
    model = SpectralClustering(num_clusters, affinity = "precomputed")

    labels = model.fit_predict(A)
        # returns the labels of clusters for each node
        # NOTE: directly usable with Adj Rand Index and Adj Mutual Info
        # next we convert this form to a list of communities

    communities_spectral_clustering = []
    for i_cluster in range(num_clusters):
        communities_spectral_clustering.append(list(np.where(labels == i_cluster)[0]))
    
    # Sorting it ascending order at all levels --> unique order for the partition
        # nodes in communities are already in order.
        # We just need to sort the communities by first element
    comm_order = np.argsort([community[0] for community in communities_spectral_clustering])
    communities_spectral_clustering = [communities_spectral_clustering[i_comm] for i_comm in comm_order] 
    return(communities_spectral_clustering)

# Why a custom function? networkx returns unsorted communities, we sort by ids which help later in fixing community colors.
def community_detection_by_louvain_algorithm(G, set_seed = 2):
    communities_louvain = nx.community.louvain_communities(G, weight = "weight", seed = set_seed)
    communities_louvain = [list(set(comm)) for comm in communities_louvain]
    len_communities_louvain = {i_comm: len(comm) for comm, i_comm in zip(communities_louvain, range(len(communities_louvain)))}
    len_communities_louvain = {k: v for k, v in sorted(len_communities_louvain.items(), key=lambda item: item[1])}
    list_keys = list(len_communities_louvain.keys())
    communities_louvain = [communities_louvain[elem] for elem in list_keys]

    # Sorting it ascending order at all levels --> unique order for the partition
        # nodes in communities are already in order.
        # We just need to sort the communities by first element
    comm_order = np.argsort([community[0] for community in communities_louvain])
    communities_louvain = [communities_louvain[i_comm] for i_comm in comm_order] 
 
    return(communities_louvain)
# ----------------------------------------------------------------------------------------
# -------------------------------- GRAPH/NETWORK DISTANCES -------------------------------
# ----------------------------------------------------------------------------------------

# -----------------------------> Spectral DISTANCES <-------------------------------------
# ----------------------------------------------------------------------------------------

## We use the laplacian spectral distance based on the graph Laplacian matrices of our networks of interest.
## One reason: consistent with the spectral clustering algorithm we use in community detection
## So we can compare the changes in spectral distances and the communities identified through spectral clustering

def three_spectral_distances(G_list, A_list):
    eigenvalues_A_list = []
    eigenvalues_L_list = []
    eigenvalues_norm_L_list = []

    for G, A in zip(G_list, A_list):
        D = np.diag([v for k, v in G.degree(weight = "weight")])
        # D_alt = np.diag(np.sum(A, axis = 0))
        L = D - A
        D_neg_half = D**(-0.5)
        D_neg_half[D_neg_half == np.inf] = 0
        normalized_L = (D_neg_half @ L) @ D_neg_half

        # Adjacency Spectral Distance
        eigenvalues_A, eigenvectors_A = np.linalg.eig(A)
            # NOTE: This time we need to sort it, unlike spectral clustering.
            # If we truncated (that is, used some but not all eigenvalues), the order (e.g., ascending/descending) would matter too.
            # For Adjacency matrices, in descending order. For Laplacian or Normalized Laplacian matrices, we need to use ascending order.
        sorted_indices_of_eigenvalues_A = np.argsort(eigenvalues_A)
        sorted_eigenvalues_A = eigenvalues_A[sorted_indices_of_eigenvalues_A]
        eigenvalues_A_list.append(sorted_eigenvalues_A)

        # Laplacian Spectral Distance
        eigenvalues_L, eigenvectors_L = np.linalg.eig(L)
            # NOTE: This time we need to sort it, unlike spectral clustering.
            # If we truncated (that is, used some but not all eigenvalues), the order (e.g., ascending/descending) would matter too.
            # For Adjacency matrices, in descending order. For Laplacian or Normalized Laplacian matrices, we need to use ascending order.
        sorted_indices_of_eigenvalues_L = np.argsort(eigenvalues_L)
        sorted_eigenvalues_L = eigenvalues_L[sorted_indices_of_eigenvalues_L]
        eigenvalues_L_list.append(sorted_eigenvalues_L)

        # Normalized Laplacian Spectral Distance
        eigenvalues_norm_L, eigenvectors_norm_L = np.linalg.eig(normalized_L)
            # NOTE: This time we need to sort it, unlike spectral clustering.
            # If we truncated (that is, used some but not all eigenvalues), the order (e.g., ascending/descending) would matter too.
            # For Adjacency matrices, in descending order. For Laplacian or Normalized Laplacian matrices, we need to use ascending order.
        sorted_indices_of_eigenvalues_norm_L = np.argsort(eigenvalues_norm_L)
        sorted_eigenvalues_norm_L = eigenvalues_norm_L[sorted_indices_of_eigenvalues_norm_L]
        eigenvalues_norm_L_list.append(sorted_eigenvalues_norm_L)

    adjacency_spectral_distance = np.sqrt(np.sum((eigenvalues_A_list[0] - eigenvalues_A_list[1])**2))
    laplacian_spectral_distance = np.sqrt(np.sum((eigenvalues_L_list[0] - eigenvalues_L_list[1])**2))
    normalized_laplacian_spectral_distance = np.sqrt(np.sum((eigenvalues_norm_L_list[0] - eigenvalues_norm_L_list[1])**2))
    return(adjacency_spectral_distance, laplacian_spectral_distance, normalized_laplacian_spectral_distance)


# ----------------------------------------------------------------------------------------
# -------------------------------- NETWORK VISUALIZATIONS --------------------------------
# ----------------------------------------------------------------------------------------

# ---------------------------> Draw networks with node labels <---------------------------
# ----------------------------------------------------------------------------------------
    # all nodes in ONE COLOR, a sprint layout hard-coded.

def draw_network_RR(G, ax, seed_to_set = 2):
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed= seed_to_set)  # positions for all nodes
    # pos = nx.spectral_layout(G) # Altenate choices

    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}

    nx.draw_networkx_nodes(G, pos, ax = ax, nodelist=G.nodes, node_color="tab:red", **options)
    # nx.draw_networkx_edges(G, pos, ax = ax, width=1.0, alpha=0.5) # perhaps redundant

    nx.draw_networkx_edges(
		G,
		pos,
        ax = ax,
		edgelist=G.edges,
		width=2,
		alpha=0.5,
		edge_color="tab:blue",
	)

    modified_labels = {element: element for element in G.nodes}
    nx.draw_networkx_labels(G, pos, modified_labels, ax = ax, font_size=10, font_color="whitesmoke")


## NOTE: Need to cite the pages for the following codes.

# --------------------> Draw networks colored by communities <----------------------------
# ----------------------------------------------------------------------------------------
    # Quite a lot of work and functions.
    # Most tricky one was to get the similar communities in the same color across figures


# -------------------->> Function to assign colors to communities <<----------------------
    # We have 10 colors, recycled if num communities > 10
def create_community_node_colors(G, communities):
    number_of_colors = len(communities)
    # print(number_of_colors)
    all_colors = ["pink", "#D4FCB1", "#CDC5FC", "gold", "#BCC6C8", "hotpink", "deepskyblue", "cyan", "olive", "orchid"]
        # [lightgreen, lightpurple, pink, yellow, gray]
    if number_of_colors <= len(all_colors):
        colors = all_colors[:number_of_colors]
    else:
        n_full = int(np.floor(number_of_colors/len(all_colors)))
        n_partial = number_of_colors%len(all_colors)
        colors = all_colors*n_full + all_colors[:n_partial]
    # print(colors)
    node_colors = []
    for node in G:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors

# ----->> Function to create matching communities & colors between two partitions <<------
# NOTE: Found it quite tricky, but the current solution seems kinda simple and elegant.
    # We want to have for two similar communities in the same colors across figures.
    # We have implemented a method to do that here.
    # NOTE: We use the first partition or set 1 of communities as the reference.
    # Then we find similar communities in set 2 and make the colors same for communities with 50% Jaccard similarity.
    # For the rest, we ensure that there are not two non-similar comms with the same color.

def create_matching_community_node_colors_for_two_partitions(G_x, communities_set_1, communities_set_2):
    # Important NOTE again: We use the first set of communities as the reference and match the second set to it.

    sim_matrix_c1_c2 = np.array([[jaccard_similarity(community_1, community_2) for community_2 in communities_set_2] for community_1 in communities_set_1])
    match_list = np.squeeze(np.where(sim_matrix_c1_c2 > 0.5))
    if match_list.ndim == 1: match_list = match_list.reshape(-1,1)
    num_matches = match_list.shape[1]

    all_colors = ["pink", "#D4FCB1", "#CDC5FC", "gold", "#BCC6C8", "hotpink", "deepskyblue", "cyan", "olive", "orchid"]

    required_number_of_colors = len(communities_set_1) + len(communities_set_2)

    if required_number_of_colors <= len(all_colors):
        color_list = all_colors[:required_number_of_colors]
    else:
        n_full = int(np.floor(required_number_of_colors/len(all_colors)))
        n_partial = required_number_of_colors%len(all_colors)
        color_list = all_colors*n_full + all_colors[:n_partial]

    # We initialize the colors in a way that there are no matches
    colors_for_set_1 = color_list[:len(communities_set_1)]
    colors_for_set_2 = color_list[len(communities_set_1) : (len(communities_set_1) + len(communities_set_2))]

    # for each match, copy color from first set to the next
    for match_i in np.arange(num_matches): # index for matches we found.
        comm_i_set1 = match_list[0, match_i]
        comm_i_set2 = match_list[1, match_i]
        # colors_for_set_1[comm_i_set1] = color_list[match_i]
        colors_for_set_2[comm_i_set2] = colors_for_set_1[comm_i_set1]
        color_list.remove(color_list[match_i])

    # Now, we create nodewise colors
    # Starting with the first set
    node_colors_for_set_1 = []
    for node in G_x:
        current_community_index = 0
        for community in communities_set_1:
            if node in community:
                node_colors_for_set_1.append(colors_for_set_1[current_community_index])
                break
            current_community_index += 1

    # Then the second set
    # print(node_colors_for_set_1)
    node_colors_for_set_2 = []
    for node in G_x:
        current_community_index = 0
        for community in communities_set_2:
            if node in community:
                node_colors_for_set_2.append(colors_for_set_2[current_community_index])
                break
            current_community_index += 1
    return(node_colors_for_set_1, node_colors_for_set_2)


# -------->> Function to plot networks with node colors based on communities <<-----------
def visualize_communities_RR(G, communities, ax, seed_to_set = 2, algorithm = None, 
                             node_colors_by_community = None, node_size = 400, to_set_title = 1, layout_type = "spring"):
    if layout_type == "spring":
        pos = nx.spring_layout(G, k = 0.3, iterations = 50, seed = seed_to_set)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    print("Number of communities =", len(communities), communities)

    if node_colors_by_community == None:
        node_colors_by_community = create_community_node_colors(G, communities)
    # print(node_colors_by_community)

    modularity = round(nx.community.modularity(G, communities), len(communities))
    if to_set_title:    
        title = f"{algorithm}\n{len(communities)} communities with modularity = {modularity}"
        ax.set_title(title)
    nx.draw(
        G,
        pos=pos,
        ax = ax,
        node_size = node_size,
        node_color = node_colors_by_community,
        edge_color= "gray",
        with_labels = True,
        font_size = 15,
        font_color = "black",
    )

# -------->> Function to label and list nodes communities <<-----------
    # We receive a list of communities in a network
    # We create community labels and attach each node with its community label and return a list of labels
def return_community_labels_from_community_set(communities):
    num_communities = len(communities)
    labeled_nodes_dict = [{node:i_cluster for node in communities[i_cluster]} for i_cluster in range(num_communities)]
    labeled_nodes_dict = {k: v for d in labeled_nodes_dict for k, v in d.items()}
    labeled_nodes_dict = np.array(sorted(labeled_nodes_dict.items()))
    return(labeled_nodes_dict[:, 1])


# --------------------> NETWORK VISUALIZATION ALL TOGETHER <------------------------------
# ----------------------------------------------------------------------------------------

def plot_network_measures(G_x, adj_matrix_x, title_to_set = None, to_savefig = 0):
    num_cols, num_rows = (3,3)

    fig, axs = plt.subplots(figsize = (16, 10), ncols = num_cols, nrows = num_rows)
    bar_width = 0.5

    ## ---------------> NODES IN NETWORKS <------------------

    # Degree Centrality, WEIGHTED
    ax = axs[0,0]
    to_plot = G_x.degree(weight = "weight")
    labels, values = zip(*dict(to_plot).items())
    indices = labels
    ax.bar(indices, values, bar_width)
    ax.set_xticks(indices, labels, rotation = 90)
    ax.set_xlabel("Topic ID (0-26)")
    ax.set_ylabel("Degree")
    ax.set_title("(a) Degree Centrality (Weighted)")
    # ax.grid(True)

    # Closeness Centrality, UN-WEIGHTED
    # ax = axs[0,1]
    # to_plot = nx.closeness_centrality(G_x)
    # labels, values = zip(*dict(to_plot).items())
    # indices = labels
    # ax.bar(indices, values, bar_width)
    # ax.set_xticks(indices, labels, rotation = 90)
    # ax.set_xlabel("Topic number (0-26)")
    # ax.set_ylabel("Closeness Centrality")

    # Clustering Co-efficient
    ax = axs[0,1]
    to_plot = nx.clustering(G_x, weight = "weight")
    labels, values = zip(*dict(to_plot).items())
    indices = labels
    ax.bar(indices, values, bar_width)
    ax.set_xticks(indices, labels, rotation = 90)
    ax.set_xlabel("Topic ID (0-26)")
    ax.set_ylabel("Clustering coefficient")
    ax.set_title("(b) Clustering coefficient")
    # ax.grid(True)

    # PageRank Centrality, WEIGHTED
    ax = axs[0,2]
    to_plot = nx.pagerank(G_x, weight = "weight")
    labels, values = zip(*dict(to_plot).items())
    indices = labels
    ax.bar(indices, values, bar_width)
    ax.set_xticks(indices, labels, rotation = 90)
    ax.set_xlabel("Node/Parameter ID (0-26)")
    ax.set_ylabel("PageRank centrality")
    ax.set_title("(c) PageRank centrality")
    # ax.grid(True)

    ## ---------------> EDGES IN NETWORKS <------------------

    # EDGE WEIGHTS directly, obviously WEIGHTED...
    edge_weights = nx.get_edge_attributes(G_x, name = "weight")
    num_nodes = len(G_x.nodes)
    sorted_key_list = list(itertools.combinations(np.arange(num_nodes), r = 2))
    edge_weights_fixed = {key: edge_weights.get(key, 0) for key in sorted_key_list}

    # A simple check to see if all mapped correctly
    if (sum(edge_weights.values()) != sum(edge_weights_fixed.values())):
        print("Problem with Edge weights")
    # ax.grid(True)

    ax = axs[1,0]
    ax.plot(edge_weights_fixed.values(), ls = "-", marker = ".")
    ax.set_xlabel("Edge ID (0-350)")
    ax.set_ylabel("Edge weights")
    ax.set_title("(d) Edge weights")

    # EDGE BETWEENNESS Centrality, WEIGHTED
    edge_bet = nx.edge_betweenness_centrality(G_x, weight = "weight")
    num_nodes = len(G_x.nodes)
    sorted_key_list = list(itertools.combinations(np.arange(num_nodes), r = 2))
    edge_bet_fixed = {key: edge_bet.get(key, 0) for key in sorted_key_list}

    # A simple check to see if all mapped correctly
    if (sum(edge_bet.values()) != sum(edge_bet_fixed.values())):
        print("Problem with Edge Betnness")
    # ax.grid(True)

    ax = axs[1,1]
    ax.plot(edge_bet_fixed.values(), ls = "-", marker = ".")

    # to_plot = edge_bet_fixed
    # labels, values = zip(*dict(to_plot).items())
    # indices = labels
    # ax.bar(indices, values, bar_width)

    ax.set_xlabel("Edge ID (0-350)")
    ax.set_ylabel("Edge betweenness centrality")
    ax.set_title("(e) Edge betweenness centrality")
    # ax.grid(True)
    # ax = axs[1,2]
    # draw_network_RR(G_x, ax = ax)

    ## ---------------> COMMUNITIES IN NETWORKS <------------------
        # Two G-N (the best two) and one louvain
        # Perhaps good to include: Spectral, keeping the best G-N only.


    communities, modularity_df = communities_and_modularities_girvan_newman(G_x)

    # Modularity df from G-N algorithms' steps
    ax = axs[1,2]
    modularity_df.plot.bar(
        x="k",
        xlabel ="community index, k",
        ylabel = "Modularity, Q",
        ax = ax,
        color ='#0173b2',
        title ="(f) Modularity Trend for Girvan-Newman (G-N) Community Algorithm")
    # Plot graph with colouring based on communities

    k_with_highest_Q = modularity_df.sort_values(by = "modularity")[::-1].k.iloc[0]
    # k_with_2nd_highest_Q = modularity_df.sort_values(by = "modularity")[::-1].k.iloc[1]
    # ax.grid(True)
    # Plot change in modularity, for G-N algorithm, as the important edges are removed


    ## A SEMI-CLEVER WAY TO MAKE SIMILAR COMMUNITIES THE SAME COLOR
        # We want: Across partitions, identify similar communities and use the same color for them.
        # we have implemented a function to "create_matching_community_node_colors_for_two_partitions"
        # For more than two partitions (or sets of communities), we hold one partition as the reference
        # and match the rest to the reference.
        # An important NOTE: We use the Louvain partition or commuinities as the reference

    communities_GN = communities[k_with_highest_Q - 1]
    communities_louvain = nx.community.louvain_communities(G_x, weight = "weight")
    communities_louvain = [list(set(comm)) for comm in communities_louvain]
    len_communities_louvain = {i_comm: len(comm) for comm, i_comm in zip(communities_louvain, range(len(communities_louvain)))}
    len_communities_louvain = {k: v for k, v in sorted(len_communities_louvain.items(), key=lambda item: item[1])}
    xx = list(len_communities_louvain.keys())
    communities_louvain = [communities_louvain[xx[i_comm]] for i_comm in range(len(communities_louvain))]
    # print("Temp", communities_louvain)
    communities_spectral_clustering = community_detection_by_spectral_clustering(G_x, adj_matrix_x, num_clusters = len(communities_louvain))

    # An important NOTE worth mentioning again: We use the Louvain partition or commuinities as the reference
    node_colors_for_louvain, node_colors_for_spectral_clustering = \
        create_matching_community_node_colors_for_two_partitions(G_x, communities_louvain, communities_spectral_clustering)

    node_colors_for_louvain, node_colors_for_GN = \
        create_matching_community_node_colors_for_two_partitions(G_x, communities_louvain, communities_GN)


    # plot the communities with highest Q acc to G-N algorithm
    ax = axs[2,0]
    visualize_communities_RR(G_x, communities_GN, ax = ax, algorithm = "(g) G-N algorithm, highest modularity", node_colors_by_community = node_colors_for_GN)

    # plot the communities with highest Q acc to G-N algorithm
    # ax = axs[2,1]
    # visualize_communities_RR(G_x, communities[k_with_2nd_highest_Q - 1], ax = ax, algorithm = "G-N")

    # plot the louvain communities
    ax = axs[2,1]
    visualize_communities_RR(G_x, communities_louvain, ax = ax, algorithm = "(h) Louvain algorithm", node_colors_by_community = node_colors_for_louvain)

    # plot the communities detected via Spectral Clustering
    ax = axs[2,2]
    visualize_communities_RR(G_x, communities_spectral_clustering, ax = ax, algorithm = "(i) Spectral clustering", node_colors_by_community = node_colors_for_spectral_clustering)
    # ax.set_title("(i) Community Detection, Spectral clustering")

    plt.suptitle(title_to_set, fontsize = 20, ha = "center")
    plt.tight_layout()
    if (to_savefig): plt.savefig("Figs/Networks_for_" + title_to_set + ".png", dpi = 300)
    plt.show()
    plt.close()


# ----------------------------------------------------------------------------------------
# -------------------------------- BOOTSTRAPPING TESTS -----------------------------------
# ----------------------------------------------------------------------------------------

# NOTE: while the previous plot_network_measures function takes in a network, this function works directly with the dfs and creates the networks

def plot_network_measures_bootstrapping(input_df, title_to_set = None, bootstrapping_level = 0.8, \
                                        num_runs = 10, community_analysis = 0):
    num_cols, num_rows = (3,2)
    fig, axs = plt.subplots(figsize = (16, 7), ncols = num_cols, nrows = num_rows)

    for i_run in range(num_runs):
    # "prime" dfs containing 80% of the data
        set_of_all_options = input_df.index
        num_options_to_choose = int(bootstrapping_level*input_df.index.shape[0])
        input_prime_df = input_df.loc[np.random.choice(set_of_all_options, num_options_to_choose, replace = False)]

        adj_matrix_prime = input_prime_df.likely_knobs_lsa.apply(lambda x: return_adj_matrix(x)).sum() # Create adj matrix from likely knobs
        G_prime = nx.from_numpy_array(adj_matrix_prime) # Adj matrix --> nx network

        ax = axs[0,0]
        to_plot = G_prime.degree(weight = "weight")
        labels, values = zip(*dict(to_plot).items())
        indices = labels
        # ax.bar(indices, values, bar_width)
        ax.plot(indices, values, ls = "--", marker = ".")
        ax.set_xticks(indices, labels, rotation = 90)
        ax.set_xlabel("Topic number (0-26)")
        ax.set_ylabel("Degree")
        ax.grid(True)

        ax = axs[0,1]
        to_plot = nx.closeness_centrality(G_prime)
        labels, values = zip(*dict(to_plot).items())
        indices = labels
        ax.plot(indices, values, ls = "--", marker = ".")
        ax.set_xticks(indices, labels, rotation = 90)
        ax.set_xlabel("Topic number (0-26)")
        ax.set_ylabel("Closeness Centrality")
        ax.grid(True)

        ax = axs[0,2]
        to_plot = nx.pagerank(G_prime, weight = "weight")
        labels, values = zip(*dict(to_plot).items())
        indices = labels
        ax.plot(indices, values, ls = "--", marker = ".")
        ax.set_xticks(indices, labels, rotation = 90)
        ax.set_xlabel("Topic number (0-26)")
        ax.set_ylabel("PageRank Centrality")
        ax.grid(True)

        ax = axs[1,0]
        ax.plot(nx.edge_betweenness_centrality(G_prime, weight = "weight").values(), ls = "--", marker = ".")
        ax.set_xlabel("Edge number")
        ax.set_ylabel("Edge centrality")
        ax.grid(True)

        ax = axs[1,1]
        to_plot = nx.clustering(G_prime, weight = "weight")
        labels, values = zip(*dict(to_plot).items())
        indices = labels
        ax.plot(indices, values, ls = "--", marker = ".")
        ax.set_xticks(indices, labels, rotation = 90)
        ax.set_xlabel("Topic number (0-26)")
        ax.set_ylabel("Clustering Coefficient")
        ax.grid(True)

#     ax = axs[1,2]
#     draw_network_RR(G_prime, ax = ax)

# ## -------------------------- COMMUNITIES IN MODELS ---------------------------
# # -------------------- Run 0 --------------------------------------------------
        if (community_analysis == "girvan-newman"):
            communities, modularity_df = communities_and_modularities_girvan_newman(G_prime)

#     # Plot graph with colouring based on communities
            k_with_highest_Q = modularity_df.sort_values(by = "modularity")[::-1].k.iloc[0]
            k_with_2nd_highest_Q = modularity_df.sort_values(by = "modularity")[::-1].k.iloc[1]
            print(i_run, communities[k_with_highest_Q - 1], "\n", communities[k_with_2nd_highest_Q - 1])
        elif (community_analysis == "louvain"):
            communities_louvain = nx.community.louvain_communities(G_prime, weight = "weight")
            print(i_run, communities_louvain,
                  nx.community.modularity(G_prime, communities_louvain))


#     ax = axs[2,0]
#     visualize_communities_RR(G_prime, communities[k_with_highest_Q - 1], ax = ax)
#     ax = axs[2,1]
#     visualize_communities_RR(G_prime, communities[k_with_2nd_highest_Q - 1], ax = ax)
#     ax = axs[2,2]
#     # Plot change in modularity as the important edges are removed
#     modularity_df.plot.bar(
#         x="k",
#         xlabel ="community index, k",
#         ylabel = "Modularity, Q",
#         ax = ax,
#         color ="royalblue",
#         title ="Modularity Trend for Girvan-Newman Community Algorithm",
#     )

    plt.suptitle(title_to_set)
    plt.tight_layout()
    plt.savefig("Figs/Networks_for_" + title_to_set + ".png", dpi = 300)
    plt.show()
    plt.close()

