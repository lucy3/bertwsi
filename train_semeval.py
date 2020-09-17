from wsi.lm_bert import LMBert
from wsi.slm_interface import SLM
from wsi.semeval_utils import generate_sem_eval_2013_no_tokenization, generate_sem_eval_2010_no_tokenization, \
            evaluate_labeling_2010, evaluate_labeling_2013,get_n_senses_corr, \
            generate_semeval2010_train, generate_semeval2013_train
from collections import defaultdict
from wsi.wsi_clustering import cluster_inst_ids_representatives, match_inst_id_representatives
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

max_num_senses = 35
min_sense_instances = 10
SEMEVAL2010_CLUSTERS = '/global/scratch/lucy3_li/bertwsi/semeval_clusters_2010_' + str(max_num_senses) + \
        '_' + str(min_sense_instances) + '/' 
if not os.path.exists(SEMEVAL2010_CLUSTERS):
    os.makedirs(SEMEVAL2010_CLUSTERS)
SEMEVAL2013_CLUSTERS = '/global/scratch/lucy3_li/bertwsi/semeval_clusters_2013_' + str(max_num_senses) + \
        '_' + str(min_sense_instances) + '/'
if not os.path.exists(SEMEVAL2013_CLUSTERS): 
    os.makedirs(SEMEVAL2013_CLUSTERS)
#SEMEVAL2010_CLUSTERS = '/global/scratch/lucy3_li/bertwsi/sc_2010_exp5/' 
#SEMEVAL2013_CLUSTERS = '/global/scratch/lucy3_li/bertwsi/sc_2013_exp5/'  

def perform_wsi(ds_name, bilm, train_gen, test_gen, \
        wsisettings: WSISettings, eval_proc, print_progress=False, seed=0):
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
        lemma = lemma_pos.replace('.', '-') # just in case, for file naming 
        if ds_name == 'SemEval2010': 
            outpath = SEMEVAL2010_CLUSTERS + str(seed) + '_' + lemma
        else: 
            outpath = SEMEVAL2013_CLUSTERS + str(seed) + '_' + lemma
        # TRAIN
        lemma_pos = lemma_pos.replace('.j', '.a') # difference between train/test pos label
        
        train_iits_all = train_ds_by_target[lemma_pos]
        inst_ids = list(train_iits_all.keys())
        num_test = len(test_inst_id_to_sentence)
        '''
        # train on test examples
        train_inst_id_to_sentence = {}
        for inst_id in test_inst_id_to_sentence: 
            train_inst_id_to_sentence[inst_id] = test_inst_id_to_sentence[inst_id]
        # plus additional train examples
        cap = 500
        if num_test < cap: # number of test examples not enough
            if len(inst_ids) > cap - num_test: # go up to the cap
                inst_ids = random.sample(inst_ids, cap - num_test)
            # otherwise grab all training examples
            for inst_id in inst_ids: 
                train_inst_id_to_sentence[inst_id + '0000'] = train_iits_all[inst_id]
        print(lemma_pos, len(train_inst_id_to_sentence))
        '''
        
        if len(inst_ids) > 500: # number of train examples
            inst_ids = random.sample(inst_ids, 500)
            train_inst_id_to_sentence = {}
            for inst_id in inst_ids: 
                train_inst_id_to_sentence[inst_id] = train_iits_all[inst_id]
        else: 
            train_inst_id_to_sentence = train_iits_all
        
        train_inst_ids_to_representatives = \
            bilm.predict_sent_substitute_representatives(inst_id_to_sentence=train_inst_id_to_sentence,
                                                              wsisettings=wsisettings)
        train_senses, statistics = cluster_inst_ids_representatives(
            inst_ids_to_representatives=train_inst_ids_to_representatives,
            max_number_senses=wsisettings.max_number_senses,min_sense_instances=wsisettings.min_sense_instances,
            disable_tfidf=wsisettings.disable_tfidf,explain_features=True,save_clusters=outpath)

        # TEST
        test_inst_ids_to_representatives = \
            bilm.predict_sent_substitute_representatives(inst_id_to_sentence=test_inst_id_to_sentence,
                                                              wsisettings=wsisettings)
        clusters = match_inst_id_representatives(test_inst_ids_to_representatives, save_clusters=outpath)
        '''
        # this is for only evaluating on test set examples
        clusters = {}
        for inst_id in train_senses: 
            if inst_id in test_inst_id_to_sentence: 
                clusters[inst_id] = train_senses[inst_id]
        '''
        inst_id_to_sense.update(clusters)
        
    out_key_path = None
    if wsisettings.debug_dir:
        out_key_path = os.path.join(wsisettings.debug_dir, f'{wsisettings.run_name}-{ds_name}.key')

    if print_progress:
        print(f'writing {ds_name} key file to %s' % out_key_path)

    return eval_proc(inst_id_to_sense, out_key_path)

def examine_data(gen): 
    ds_by_target = defaultdict(dict)
    for pre, target, post, inst_id in gen:
        lemma_pos = inst_id.rsplit('.', 1)[0]
        ds_by_target[lemma_pos][inst_id] = (pre, target, post)
    for target in ds_by_target: 
        print(target, len(ds_by_target[target]))
        i = 0
        for inst_id in ds_by_target[target]: 
            print(ds_by_target[target][inst_id])
            i += 1
            if i > 5: break
        print()

def main():
    s = 0
    random.seed(s)
    # the following is copied from wsi_bert.py
    settings = DEFAULT_PARAMS._asdict()
    settings['max_number_senses'] = max_num_senses # adjustment
    settings['min_sense_instances'] = min_sense_instances
    settings['run_name'] = '500params_' + str(max_num_senses) + '_' + str(min_sense_instances) # 'eval500'
    settings['patterns'] = [('{pre} {target_predict} {post}', 0.5)] # no dynamic patterns
    settings = WSISettings(**settings)

    lm = LMBert(settings.cuda_device, settings.bert_model,
                            max_batch_size=settings.max_batch_size)
    if sys.platform == 'linux':
                os.popen(f"taskset -cp 0-{cpu_count()-1} {os.getpid()}").read() 
   
    # this part is new 
    # semeval 2013 eval_proc has been modified to do single-sense evaluation 
    ''' 
    test_gen = generate_sem_eval_2013_no_tokenization('./resources/SemEval-2013-Task-13-test-data')
    train_gen = generate_semeval2013_train('/global/scratch/lucy3_li/ingroup_lang/logs/ukwac2.txt')
    
    scores2013, corr = perform_wsi('SemEval2013',
            lm,
            train_gen, 
            test_gen, 
            settings, 
            lambda inst2sense, outkey: 
            evaluate_labeling_2013('./resources/SemEval-2013-Task-13-test-data', inst2sense, outkey), 
            print_progress=False, seed=s)
    fnmi = scores2013['all']['FNMI']
    fbc = scores2013['all']['FBC']
    msg = 'SemEval 2013 FNMI %.2f FBC %.2f AVG %.2f' % (fnmi * 100, fbc * 100, np.sqrt(fnmi * fbc) * 100)
    print(msg)
    '''
    test_gen = generate_sem_eval_2010_no_tokenization('./resources/SemEval-2010/test_data')
    train_gen = generate_semeval2010_train('/global/scratch/lucy3_li/ingroup_lang/semeval-2010-task-14/training_data/')
    scores2010, corr = perform_wsi('SemEval2010', 
            lm,
            train_gen, 
            test_gen, 
            settings, 
            lambda inst2sense, outkey:
            evaluate_labeling_2010('./resources/SemEval-2010/evaluation/', inst2sense, outkey),
            print_progress=False, seed=s)

    fscore = scores2010['all']['FScore']
    v_measure = scores2010['all']['V-Measure']

    msg = 'SemEval 2010 FScore %.2f V-Measure %.2f AVG %.2f' % (
            fscore * 100, v_measure * 100, np.sqrt(fscore * v_measure) * 100)
    print(msg)
    
    

if __name__ == '__main__':
    main()
