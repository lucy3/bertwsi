from wsi.slm_interface import SLM
from wsi.semeval_utils import generate_sem_eval_2013_no_tokenization, generate_sem_eval_2010_no_tokenization, \
    evaluate_labeling_2010, evaluate_labeling_2013,get_n_senses_corr
from collections import defaultdict
from wsi.lm_bert import LMBert
from wsi.wsi_clustering import cluster_inst_ids_representatives, match_inst_id_representatives
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
from itertools import islice

CUDA_LAUNCH_BLOCKING="1"
ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
LOGS = ROOT + 'logs/'
BERTWSI = '/global/scratch/lucy3_li/bertwsi/'
INPUT = BERTWSI + 'reddit_input/'
CLUSTERS = BERTWSI + 'reddit_clusters/'
SENSES = BERTWSI + 'ag_senses/'
OTHER_CLUSTERS = LOGS + 'reddit_centroids/'

def chunks(data):
    size = 3000
    if len(data) > size: 
        it = iter(data)
        for i in range(0, len(data), size):
            yield {k:data[k] for k in islice(it, size)}
    else: 
        yield data

def main():
    subreddit = sys.argv[1]
    print(subreddit)

    settings = DEFAULT_PARAMS._asdict()
    settings['max_number_senses'] = 25
    settings['disable_lemmatization'] = True
    settings['run_name'] = 'reddit_' + subreddit
    settings['patterns'] = [('{pre} {target_predict} {post}', 0.5)]
    settings = WSISettings(**settings)

    lm = LMBert(settings.cuda_device, settings.bert_model,
                            max_batch_size=settings.max_batch_size)
    if sys.platform == 'linux':
        os.popen(f"taskset -cp 0-{cpu_count()-1} {os.getpid()}").read() 

    with open(LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
        d = json.load(infile)
    doc = INPUT + subreddit
    inst_id_to_sentence = defaultdict(dict)
    row_number = 0
    with open(doc, 'r') as infile:
        reader = csv.reader(infile, delimiter=',')
        for row in reader: 
            line_number = row[0]
            curr_user = row[1]
            lh = row[2]
            word = row[3]
            rh = row[4]
            inst_id_to_sentence[word][word + '.' + curr_user + '.' + str(line_number) + '-' + str(row_number)] = (lh, word, rh)
            row_number += 1
    start = time.time()
    outfile = open(SENSES + subreddit, 'w')
    for word in inst_id_to_sentence:
        ID = d[word]
        # filter out emojis since our other model doesn't handle them 
        if not os.path.exists(OTHER_CLUSTERS + str(ID) + '.npy'): continue
        for inst_batch in chunks(inst_id_to_sentence[word]):
            outpath = CLUSTERS + str(ID)
            inst_ids_to_representatives = lm.predict_sent_substitute_representatives(inst_id_to_sentence=inst_batch,
                                                                              wsisettings=settings)
            word_senses = match_inst_id_representatives(inst_ids_to_representatives, save_clusters=outpath)
            
            for instance in word_senses:  
                max_weight = -float("inf")
                max_sense = ''
                senses = word_senses[instance]
                for sense in senses: 
                    if senses[sense] > max_weight: 
                        max_weight = senses[sense]
                        max_sense = sense
                c = instance.split('.')
                word = c[0]
                curr_user = c[-2]
                numbers = c[-1].split('-')
                if instance.startswith('..'): word = '.'
                line_number = numbers[0]
                row_number = numbers[1]
                sense_number = max_sense.split('.')[-1]
                outfile.write(str(line_number) + '_' + curr_user + '\t' + word + '\t' + str(sense_number) + '\n') 
    outfile.close()
    print("TOTAL TIME:", time.time() - start) 


if __name__ == '__main__':
    main()
