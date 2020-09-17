import json
from collections import defaultdict
import random
import csv

ROOT = '/global/scratch/lucy3_li/bertwsi/'
LOGS = '/global/scratch/lucy3_li/ingroup_lang/logs/'
INPUT = ROOT + 'reddit_output.json'
DOCS = ROOT + 'reddit_input/'
CLUSTERS = ROOT + 'reddit_clusters/'

def inspect_reddit(): 
    sense2example = defaultdict(dict) # {word : {sense : [example]} }
    with open(LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
       d = json.load(infile)

    for word in ['scheme', 'pits']:
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

def main(): 
    inspect_reddit()

if __name__ == '__main__': 
    main()
