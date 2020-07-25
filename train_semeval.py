from wsi.lm_bert import LMBert
from wsi.slm_interface import SLM
from wsi.semeval_utils import generate_sem_eval_2013_no_tokenization, generate_sem_eval_2010_no_tokenization, \
            evaluate_labeling_2010, evaluate_labeling_2013,get_n_senses_corr
from collections import defaultdict
from wsi.wsi_clustering import cluster_inst_ids_representatives
from tqdm import tqdm
import numpy as np
import csv
import time
import json
import os
import logging
from time import strftime
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi import WordSenseInductor
from multiprocessing import cpu_count
import sys

def perform_wsi(ds_name, gen, wsisettings: WSISettings, eval_proc, print_progress=False):
    ds_by_target = defaultdict(dict)

    # this part was rewritten from the original code to handle both train and test split inputs
    # TODO: can keep old gen for test data, but do the same preprocessing for train data 
    for pre, target, post, inst_id in gen:
        lemma_pos = inst_id.rsplit('.', 1)[0]
        ds_by_target[lemma_pos][inst_id] = (pre, target, post)

    inst_id_to_sense = {}
    gen = ds_by_target.items()
    if print_progress:
        gen = tqdm(gen, desc=f'predicting substitutes {ds_name}')
    for lemma_pos, inst_id_to_sentence in gen:
        inst_ids_to_representatives = \
            self.bilm.predict_sent_substitute_representatives(inst_id_to_sentence=inst_id_to_sentence,
                                                              wsisettings=wsisettings)

        # TODO: modify this function to save cluster information 
        # output instead of clusters should be cluster information for matching later 
        clusters, statistics = cluster_inst_ids_representatives(
            inst_ids_to_representatives=inst_ids_to_representatives,
            max_number_senses=wsisettings.max_number_senses,min_sense_instances=wsisettings.min_sense_instances,
            disable_tfidf=wsisettings.disable_tfidf,explain_features=True)

        # TODO: get test set examples 
        # TODO: match test set examples to closest centroid. get clusters

        inst_id_to_sense.update(clusters)
    out_key_path = None
    if wsisettings.debug_dir:
        out_key_path = os.path.join(wsisettings.debug_dir, f'{wsisettings.run_name}-{ds_name}.key')

    if print_progress:
        print(f'writing {ds_name} key file to %s' % out_key_path)

    return eval_proc(inst_id_to_sense, out_key_path)


def main(): 
    # the following is copied from wsi_bert.py
    settings = DEFAULT_PARAMS._asdict()
    settings = WSISettings(**settings)
    lm = LMBert(settings.cuda_device, settings.bert_model,
                            max_batch_size=settings.max_batch_size)
    if sys.platform == 'linux':
                os.popen(f"taskset -cp 0-{cpu_count()-1} {os.getpid()}").read() 
   
    # this part is new 
    # TODO: include the proper arguments 
    perform_wsi() # on semeval 2010

    perform_wsi() # on semeval 2013
    
    

if __name__ == '__main__':
    main()
