"""
Reads in input file for a word
and outputs a csv of pre,target,post 
"""

ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
LOGS = ROOT + 'logs/'
INPUT_LOGS = ROOT + 'logs/'
OUTPUT = '/global/scratch/lucy3_li/bertwsi/reddit_input/'

from transformers import BasicTokenizer
from nltk.tokenize import sent_tokenize
import json
import csv

def format_all_examples(word):
    with open(INPUT_LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
        d = json.load(infile)
    ID = d[word]
    outfile = open(OUTPUT + str(ID), 'w') 
    writer = csv.writer(outfile, delimiter=',')
    tokenizer = BasicTokenizer(do_lower_case=True)
    curr_user = None
    with open(OUTPUT + word + '/part-00000', 'r') as infile: 
        line_number = 0
        for line in infile: 
            contents = line.strip()
            if contents.startswith('USER1USER0USER'): 
                curr_user = contents
            else:
                sent_tok = sent_tokenize(contents)
                for sent in sent_tok: 
                    tokens = tokenizer.tokenize(sent.strip())
                    if word in tokens: 
                        i = tokens.index(word)
                        lh = ' '.join(tokens[:i])
                        rh = ' '.join(tokens[i+1:])
                        writer.writerow([line_number, curr_user, lh, word, rh])
                        break
            line_number += 1
    outfile.close()
 

def format_n_examples(words): 
    with open(INPUT_LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
        d = json.load(infile)
    tokenizer = BasicTokenizer(do_lower_case=True)
    for word in words:
        ID = d[word]
        doc = INPUT_LOGS + 'vocabs/docs/' + str(ID)
        curr_user = None
        line_number = 0
        outfile = open(OUTPUT + str(ID), 'w') 
        writer = csv.writer(outfile, delimiter=',')
        with open(doc, 'r') as infile: 
            for line in infile: 
                contents = line.strip()
                if contents.startswith('USER1USER0USER'): 
                    curr_user = contents
                else:
                    sent_tok = sent_tokenize(contents)
                    for sent in sent_tok: 
                        tokens = tokenizer.tokenize(sent.strip())
                        if word in tokens: 
                            i = tokens.index(word)
                            lh = ' '.join(tokens[:i])
                            rh = ' '.join(tokens[i+1:])
                            writer.writerow([line_number, curr_user, lh, word, rh])
                            break
                line_number += 1
        outfile.close()

def main(): 
    #format_n_examples(['ow', 'haul', 'transmission', 'the', 'dial'])
    format_all_examples('add')

if __name__ == '__main__': 
    main()
