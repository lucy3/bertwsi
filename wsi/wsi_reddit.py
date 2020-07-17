from .slm_interface import SLM
from wsi.semeval_utils import generate_sem_eval_2013_no_tokenization, generate_sem_eval_2010_no_tokenization, \
    evaluate_labeling_2010, evaluate_labeling_2013,get_n_senses_corr
from collections import defaultdict
from .wsi_clustering import cluster_inst_ids_representatives
from tqdm import tqdm
import logging
from .WSISettings import DEFAULT_PARAMS, WSISettings
import os
import numpy as np

def main(): 
   '''
   input: dictionary of instance ID to (pre, target, post) 
   output: dictionary of instance ID to sense to weight 
   '''
   settings = DEFAULT_PARAMS._asdict()
   settings['disable_lemmatization'] = True
   settings = WSISettings(**settings)

   lm = LMBert(settings.cuda_device, settings.bert_model,
                max_batch_size=settings.max_batch_size)

   dataset = set(['cat'])

   for word in dataset:
       inst_id_to_sense = {} 
       inst_id_to_sentence = {}
       inst_id_to_sentence['cat1'] = ('I played with the', 'cat', 'at the animal shelter')
       inst_id_to_sentence['cat2'] = ('I played with the fluffy', 'cat', 'at the library')
       inst_id_to_sentence['cat3'] = ('I pet the', 'cat', 'at the animal shelter')
       inst_id_to_sentence['cat4'] = ('I plan to adopt a', 'cat', 'from the animal shelter')
 

       lm.predict_sent_substitute_representatives(inst_id_to_sentence=inst_id_to_sentence,
                                                                  wsisettings=wsisettings)

       clusters, statistics = cluster_inst_ids_representatives(
                inst_ids_to_representatives=inst_ids_to_representatives,
                max_number_senses=wsisettings.max_number_senses,min_sense_instances=wsisettings.min_sense_instances,
                disable_tfidf=wsisettings.disable_tfidf,explain_features=False)
       inst_id_to_sense.update(clusters)

if __name__ == '__main__':
   main()

