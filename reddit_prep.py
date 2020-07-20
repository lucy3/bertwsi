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

words = ['ow', 'haul', 'transmission', 'the', 'dial']
for word in words: 
	with open(INPUT_LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
		d = json.load(infile)
	ID = d[word]
	doc = INPUT_LOGS + 'vocabs/docs/' + str(ID)
	tokenizer = BasicTokenizer(do_lower_case=False)
	sentences = []
	curr_user = None
	line_number = 0
	outfile = open(OUTPUT + word, 'w') 
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
