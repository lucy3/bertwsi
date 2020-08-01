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

def main(): 
    # for each subreddit
    # get all instances of vocab words in sentences w/ line number and user
    # for each vocab word 
    # run lm.predict_sent_substitute_representatives
    # dict vectorize and tfidf transform representatives
    # match representative to closest neighbor
    # get label of closest neighbor
    # save 

if __name__ == '__main__':
    main()
