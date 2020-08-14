from wsi.slm_interface import SLM
from wsi.semeval_utils import generate_sem_eval_2013_no_tokenization, generate_sem_eval_2010_no_tokenization, \
    evaluate_labeling_2010, evaluate_labeling_2013,get_n_senses_corr
from collections import defaultdict
from wsi.lm_bert import LMBert
from wsi.wsi_clustering import cluster_inst_ids_representatives
from tqdm import tqdm
import logging
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
import os
import numpy as np
from multiprocessing import cpu_count
import csv
import time
import json
import sys

CUDA_LAUNCH_BLOCKING="1"
ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
LOGS = ROOT + 'logs/'
INPUT = '/global/scratch/lucy3_li/bertwsi/reddit_input/'
OUTPUT = '/global/scratch/lucy3_li/bertwsi/reddit_output.json'
CLUSTERS = '/global/scratch/lucy3_li/bertwsi/reddit_clusters/'

def main():
    subreddit = sys.argv[1]
    with open(LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
        d = json.load(infile)
    # get all instances of vocab words in sentences w/ line number and user
    # for each vocab word
    doc = INPUT + subreddit
    inst_id_to_sentence = defaultdict(dict)
    with open(doc, 'r') as infile:
        reader = csv.reader(infile, delimiter=',')
        i = 0
        for row in reader: 
            lh = row[2]
            word = row[3]
            rh = row[4]
            inst_id_to_sentence[word][word + '.' + str(i)] = (lh, word, rh)
            i += 1

    for word in inst_id_to_sentence: 
        bilm.predict_sent_substitute_representatives(inst_id_to_sentence=test_inst_id_to_sentence,
                                                                              wsisettings=settings)

    # run lm.predict_sent_substitute_representatives
    # dict vectorize and tfidf transform representatives
    # match representative to closest neighbor
    # get label of closest neighbor
    # save 

if __name__ == '__main__':
    main()
