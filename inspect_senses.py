import json
from collections import defaultdict
import random
import csv

from sklearn.svm import LinearSVC
from scipy import sparse
from joblib import dump, load
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter

ROOT = '/global/scratch/lucy3_li/bertwsi/'
LOGS = '/global/scratch/lucy3_li/ingroup_lang/logs/'
INPUT = ROOT + 'reddit_output.json'
DOCS = ROOT + 'reddit_input/'
CLUSTERS = ROOT + 'reddit_clusters/'

def inspect_reddit(): 
    sense2example = defaultdict(dict) # {word : {sense : [example]} }
    with open(LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
       d = json.load(infile)

    for word in ['python']:
        ID = d[word]
        with open(CLUSTERS + str(ID) + '_senses.json', 'r') as infile: 
            word_senses = json.load(infile)
        for instance in word_senses: 
            max_weight = -float("inf")
            max_sense = ''
            senses = word_senses[instance]
            for sense in senses: 
                if senses[sense] > max_weight: 
                    max_weight = senses[sense]
                    max_sense = sense
            if max_sense not in sense2example[word]: 
                sense2example[word][max_sense] = [instance]
            else: 
                sense2example[word][max_sense].append(instance)

    inst_id_to_sentence = {}

    for word in sense2example:
       ID = d[word]
       doc = DOCS + str(ID)

       with open(doc, 'r') as infile:
           reader = csv.reader(infile, delimiter=',')
           i = 0
           for row in reader: 
               lh = row[2]
               word = row[3]
               rh = row[4]
               inst_id_to_sentence[word + '.' + str(i)] = (lh, word, rh)
               i += 1

    sample_size = 10
    for word in sense2example: 
        print(word)
        print("# of senses:", len(sense2example[word]))
        print("sense cluster sizes:")
        for sense in sense2example[word]: 
            print(sense, '---', len(sense2example[word][sense]))
            print("EXAMPLES:")
            if len(sense2example[word][sense]) <= sample_size: 
               sample = sense2example[word][sense]
            else: 
               sample = random.sample(sense2example[word][sense], sample_size)
            for example in sample: 
                print(inst_id_to_sentence[example])
        print()

def inspect_semeval(): 
    pass

def get_important_substitutes(word):
    """
    This can't actually be used because rep_mat has been tfidf transformed. 
    """
    with open(LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
       d = json.load(infile)

    ID = d[word]

    save_clusters = CLUSTERS + str(ID) 

    with open(save_clusters + '_labels', 'r') as infile: 
        labels = infile.read().strip().split(' ')
    rep_mat = sparse.csr_matrix(np.load(save_clusters + '.npy'))
    dict_vectorizer = load(save_clusters + '_dictvectorizer.joblib') 
    tfidf_transformer = load(save_clusters + '_tfidftransformer.joblib')

    label_count = Counter(labels)
    statistics = []
    if len(label_count) > 1:
        svm = LinearSVC(class_weight='balanced', penalty='l1', dual=False)
        svm.fit(rep_mat, labels)

        coefs = svm.coef_
        top_coefs = np.argpartition(coefs, -10)[:, -10:]
        if top_coefs.shape[0] == 1:
            top_coefs = [top_coefs[0], -top_coefs[0]]

        rep_arr = np.asarray(rep_mat)
        totals_cols = rep_arr.sum(0)

        p_feats = totals_cols / transformed.shape[0]

        for sense_idx, top_coef_sense in enumerate(top_coefs):
            count_reps = label_count[sense_idx]
            # p_sense = count_reps / transformed.shape[0]
            count_sense_feat = rep_arr[np.where(labels == sense_idx)]
            p_sense_feat = count_sense_feat.sum(0) / transformed.shape[0]

            pmis_proxy = p_sense_feat / (p_feats + 0.00000001)
            best_features_pmi_idx = np.argpartition(pmis_proxy, -10)[-10:]

            closest_instance = best_instance_for_sense[sense_idx][1]
            best_features = [dict_vectorizer.feature_names_[x] for x in top_coef_sense]
            best_features_pmi = [dict_vectorizer.feature_names_[x] for x in best_features_pmi_idx]

            # logging.info(f'sense #{sense_idx+1} ({label_count[sense_idx]} reps) best features: {best_features}')
            statistics.append((count_reps, best_features, best_features_pmi, closest_instance))

    for idx, (rep_count, best_features,best_features_pmi, best_instance_id) in enumerate(statistics):
        best_instance = ds_by_target[lemma_pos][best_instance_id]
        nice_print_instance = f'{best_instance[0]} -{best_instance[1]}- {best_instance[2]}'
        print(f'Sense {idx}, # reps: {rep_count}, best feature words: {", ".join(best_features)}.')
        print(f', best feature words(PMI): {", ".join(best_features_pmi)}.')
        print(f' closest instance({best_instance_id}):\n---\n{nice_print_instance}\n---\n')
    
def main(): 
    #inspect_reddit()

    get_important_substitutes('python')

if __name__ == '__main__': 
    main()
