import re
import os
import pickle
import numpy as np

from commons import show_progress
from gensim.corpora import wikicorpus
from gensim.utils import tokenize
from urllib.parse import unquote
from itertools import chain

unk_ent_name, unk_ent_id, unk_ent_wiki_id = 'unk', 0, -1
ent_path = 'entities.pickle'


def gen_entity_mentions_map(wiki_path, name2wikiid, ent_max):

	#print('new! ')
	
	hyp_pattern = re.compile(r'<a[^>]*href=\"([^\">]+)\"[^>]*>([^>]+)</a>', re.DOTALL | re.UNICODE)
	
	ignored_names = set()
	wikiid2mentions = {}
	
	with open(wiki_path, 'r', encoding='utf8') as inf:
	
		for i, line in enumerate(inf):
		
			if i % 100000 == 0:
				show_progress(percent=i/(755*10**5))
				#return wikiid2mentions, ignored_names
		
			hyp_matches = re.finditer(hyp_pattern, line)

			for link in hyp_matches:

				name = unquote(link.groups()[0])   # wikipedia url id
				mention = link.groups()[1]
				
				if name in name2wikiid:
					wikiid = name2wikiid[name]
					if wikiid not in wikiid2mentions:
						wikiid2mentions[wikiid] = []
					wikiid2mentions[wikiid].append(mention)
				#else:
				#	ignored_names.add(name)
	
	show_progress(percent=1.0, done=True)
	
	return wikiid2mentions, ignored_names


def load_entity_mentions_map(ent_path, name2wikiid, men_pickle_path='mentions.pickle', ent_max=10**7):

	if os.path.exists(men_pickle_path):
		
		with open(men_pickle_path, 'rb') as inf:
			wikiid2mentions = pickle.load(inf)
			ignored_names = pickle.load(inf)

		print(f'loaded mentions from {men_pickle_path}')   
		
	else:
			
		wikiid2mentions, ignored_names = gen_entity_mentions_map(ent_path, name2wikiid, ent_max)

		with open(men_pickle_path, 'ab') as outf:
			pickle.dump(wikiid2mentions, outf)
			pickle.dump(ignored_names, outf)

		print(f'mentions processed and saved to {men_pickle_path}')

	
	return wikiid2mentions, ignored_names
	

def gen_entity_mention_vec(wikiid2mentions, id2wikiid, wikiid2name, emb_model, emb_size, alpha=0.2):
	
	#print('new!')
	
	ent_men_emb = np.memmap('ent_men_emb.bin', 
                           dtype='float32', mode='w+',
                           shape=(len(id2wikiid), emb_size))
						   
	ent_men_emb[unk_ent_id] = np.zeros((emb_size,))

	for eid in id2wikiid:
	
		if eid % 500 == 0:
			show_progress(percent=eid/len(id2wikiid))
			
		if eid == unk_ent_id:
			continue
		
		wikiid = id2wikiid[eid]
		
		tokens = tokenize(wikiid2name[wikiid], deacc=True)
		
		if wikiid in wikiid2mentions:
			mentions = wikiid2mentions[wikiid]
			tokens = chain(tokens, tokenize(' '.join(mentions), deacc=True))
		
		token_counts = {}
		
		for token in tokens:
			if token not in emb_model:
				continue
			if token in token_counts:
				token_counts[token] += 1
			else:
				token_counts[token] = 1
				
		if len(token_counts) == 0:
			ent_men_emb[eid] = np.random.uniform(-1, 1, emb_size)
			continue
	
		word_vecs = np.zeros((len(token_counts), emb_size))
		word_weights = np.zeros((len(token_counts), ))
		
		for i, token in enumerate(token_counts):
			word_vecs[i] = emb_model[token]
			word_weights[i] = token_counts[token]
		
		word_weights = (word_weights ** alpha) / word_weights.sum()
		
		ent_men_emb[eid] = np.average(word_vecs, axis=0, weights=word_weights)
		
	show_progress(percent=1.0, done=True)
	
	return ent_men_emb

	