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
   '''
   input: dictionary of instance ID to (pre, target, post) 
   output: dictionary of instance ID to sense to weight 
   '''
   word = sys.argv[1]
   settings = DEFAULT_PARAMS._asdict()
   settings['disable_lemmatization'] = True
   settings['patterns'] = [('{pre} {target_predict} {post}', 0.5)]
   settings = WSISettings(**settings)

   lm = LMBert(settings.cuda_device, settings.bert_model,
                max_batch_size=settings.max_batch_size)

   if sys.platform == 'linux':
               os.popen(f"taskset -cp 0-{cpu_count()-1} {os.getpid()}").read() 
   
   with open(LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
       d = json.load(infile)
   
   start = time.time()
   inst_id_to_sentence = {}
   ID = d[word]
   doc = INPUT + str(ID)
   print(word)
   with open(doc, 'r') as infile:
       reader = csv.reader(infile, delimiter=',')
       i = 0
       for row in reader: 
           lh = row[2]
           word = row[3]
           rh = row[4]
           inst_id_to_sentence[word + '.' + str(i)] = (lh, word, rh)
           i += 1
   
   inst_ids_to_representatives = lm.predict_sent_substitute_representatives(inst_id_to_sentence=inst_id_to_sentence,
                                                              wsisettings=settings)

   clusters, statistics = cluster_inst_ids_representatives(
            inst_ids_to_representatives=inst_ids_to_representatives,
            max_number_senses=settings.max_number_senses,min_sense_instances=settings.min_sense_instances,
            disable_tfidf=settings.disable_tfidf,explain_features=False,save_clusters=CLUSTERS + str(ID))
   end = time.time()
   print("TIME:", word, end-start)
   ''' 
   with open(OUTPUT, 'w') as outfile:
       json.dump(inst_id_to_sense, outfile)
   '''

if __name__ == '__main__':
   main()

