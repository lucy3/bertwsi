from wsi.lm_bert import LMBert
from wsi.slm_interface import SLM
from wsi.semeval_utils import generate_sem_eval_2013_no_tokenization, generate_sem_eval_2010_no_tokenization, \
            evaluate_labeling_2010, evaluate_labeling_2013,get_n_senses_corr, 
            generate_semeval2010_train, generate_semeval2013_train
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
import random

SEMEVAL_CLUSTERS = '/global/scratch/lucy3_li/bertwsi/semeval_clusters/' 

def perform_wsi(ds_name, train_gen, test_gen, wsisettings: WSISettings, eval_proc, print_progress=False):
    train_ds_by_target = defaultdict(dict)
    test_ds_by_target = defaultdict(dict)

    for pre, target, post, inst_id in test_gen:
        lemma_pos = inst_id.rsplit('.', 1)[0]
        test_ds_by_target[lemma_pos][inst_id] = (pre, target, post)

    for pre, target, post, inst_id in train_gen: 
        lemma_pos = inst_id.rsplit('.', 1)[0]
        train_ds_by_target[lemma_pos][inst_id] = (pre, target, post)

    inst_id_to_sense = {}
    test_gen = test_ds_by_target.items()
    if print_progress:
        test_gen = tqdm(test_gen, desc=f'predicting substitutes {ds_name}')
    for lemma_pos, test_inst_id_to_sentence in test_gen:
        # TRAIN
        train_iits_all = train_ds_by_target[lemma_pos]
        inst_ids = list(train_iits_all.keys())
        if len(inst_ids) > 500: # cap at 500 examples 
            inst_ids = random.sample(inst_ids, 500)
            train_inst_id_to_sentence = {}
            for inst_id in inst_ids: 
                train_inst_id_to_sentence[inst_id] = train_iits_all[inst_id]
        else: 
            train_inst_id_to_sentence = train_iits_all

        train_inst_ids_to_representatives = \
            bilm.predict_sent_substitute_representatives(inst_id_to_sentence=train_inst_id_to_sentence,
                                                              wsisettings=wsisettings)
 
        lemma = lemma_pos.split('.')[0]
        _, _ = cluster_inst_ids_representatives(
            inst_ids_to_representatives=train_inst_ids_to_representatives,
            max_number_senses=wsisettings.max_number_senses,min_sense_instances=wsisettings.min_sense_instances,
            disable_tfidf=wsisettings.disable_tfidf,explain_features=True,save_clusters=SEMEVAL_CLUSTERS+lemma)
        '''
        # TEST
        test_inst_ids_to_representatives = \
            bilm.predict_sent_substitute_representatives(inst_id_to_sentence=test_inst_id_to_sentence,
                                                              wsisettings=wsisettings)

        # TODO: match test set examples
        clusters = match_inst_id_representatives(test_inst_ids_to_representatives, save_clusters=SEMEVAL_CLUSTERS+lemma)
        
        inst_id_to_sense.update(clusters)
        '''
    out_key_path = None
    if wsisettings.debug_dir:
        out_key_path = os.path.join(wsisettings.debug_dir, f'{wsisettings.run_name}-{ds_name}.key')

    if print_progress:
        print(f'writing {ds_name} key file to %s' % out_key_path)

    return eval_proc(inst_id_to_sense, out_key_path)


def main(): 
    # the following is copied from wsi_bert.py
    settings = DEFAULT_PARAMS._asdict()
    settings['run_name'] = 'eval500'
    settings['patterns'] = [('{pre} {target_predict} {post}', 0.5)] # no ND
    settings = WSISettings(**settings)

    lm = LMBert(settings.cuda_device, settings.bert_model,
                            max_batch_size=settings.max_batch_size)
    if sys.platform == 'linux':
                os.popen(f"taskset -cp 0-{cpu_count()-1} {os.getpid()}").read() 
   
    # this part is new 

    #test_gen = generate_sem_eval_2013_no_tokenization('./resources/SemEval-2013-Task-13-test-data')
    #train_gen = generate_semeval_2013_train('./resources/semeval_2013_train')
    '''
    perform_wsi('SemEval2013', 
            train_gen, 
            test_gen, 
            settings, 
            lambda inst2sense, outkey: 
            evaluate_labeling_2013('./resources/SemEval-2013-Task-13-test-data', inst2sense, outkey), 
            print_progress=False))
    '''

    #test_gen = generate_sem_eval_2010_no_tokenization('./resources/SemEval-2010/test_data')
    train_gen = generate_semeval2010_train('/global/scratch/lucy3_li/ingroup_lang/semeval-2010-task-14/training_data/')
    ds_by_target = defaultdict(dict)
    for pre, target, post, inst_id in train_gen: 
        ds_by_target[lemma_pos][inst_id] = (pre, target, post)
    for target in ds_by_target: 
        print(target, len(ds_by_target[target]))
        i = 0
        for inst_id in ds_by_target[target]: 
            print(ds_by_target[target][inst_id])
            i += 1
            if i > 5: break
        print()
    '''
    perform_wsi('SemEval2010', 
            train_gen, 
            test_gen, 
            settings, 
            lambda inst2sense, outkey:
            evaluate_labeling_2010('./resources/SemEval-2010/evaluation/', inst2sense, outkey),
            print_progress=False)) 
    '''
    
    

if __name__ == '__main__':
    main()
