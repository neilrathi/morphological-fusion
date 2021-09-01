# basic imports
import re, os, io, time, csv, statistics, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, help='directory where files are located')
parser.add_argument('--lang', type=str, help='name of language')
parser.add_argument('--langtwo', type=str, help='name of language, two characters')
args = parser.parse_args()

# filepath
filepath = args.filepath
lang = args.lang
lang2 = args.langtwo

# files
wikifile = f'{filepath}{lang}/{lang2}wiki-20181001-corpus.xml.txt'
morphfile = f'{filepath}{lang}/{lang}'
freqfile = f'{filepath}{lang}/{lang}_counts.txt'

def get_counts(wikifile):
	lang_dict = {}
	for line in open(wikifile):
		if not line:
			continue
		for word in line.split():
			if word not in lang_dict:
				lang_dict[word] = 1
			else:
				lang_dict[word] += 1
	return lang_dict

def feat_counts(wikifile, morphfile):
	freq_dict = {}
	with open(morphfile, 'r') as f:
		morph = csv.reader(f, delimiter = '\t')
		lang_dict = get_counts(wikifile)
		for row in morph:
			if not row:
				continue
			if row[1] in lang_dict:
				if row[2] not in freq_dict:
					freq_dict[row[2]] = lang_dict[row[1]]
				else:
					freq_dict[row[2]] += lang_dict[row[1]]
	return freq_dict

freq_dict = feat_counts(wikifile, morphfile)

with open(freqfile, 'w+') as f:
	for key in freq_dict:
		f.write(f'{key}\t{freq_dict[key]}\n')