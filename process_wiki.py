import re
import os
import pickle

from commons import show_progress
from gensim.corpora import wikicorpus
from urllib.parse import unquote


unk_ent_name, unk_ent_id, unk_ent_wiki_id = 'UNK_E', 0, -1
ent_path = 'entities.pickle'


def is_special_page(title):
	return False
	

def extract_doc_tag(line):
	global unk_ent_wiki_id, unk_ent_name

	doc_pattern = re.compile(r'<doc[^>]+id=\"([0-9]+)\"[^>]+title=\"([^>\"]+)\"[^>]*>', re.DOTALL | re.UNICODE)

	doc_tag = re.match(doc_pattern, line)

	if doc_tag:
		doc_id, doc_title = tuple(doc_tag.groups())
		wiki_id = int(doc_id)
		
		if wiki_id == 3946:
			print(line)
			return 0
		if not is_special_page(doc_title):
			return wiki_id, doc_title

	return unk_ent_wiki_id, unk_ent_name
	

def gen_entity_id_maps(wiki_path, ent_max):

	global unk_ent_name, unk_ent_id, unk_ent_wiki_id
	name2wikiid, wikiid2name = {}, {}
	id2wikiid, wikiid2id = {}, {},

	print(f'-- looking for entity articles in {wiki_path} corpus...', end=' ', flush=True)

	name2wikiid[unk_ent_name] = unk_ent_wiki_id
	wikiid2name[unk_ent_wiki_id] = unk_ent_name
	id2wikiid[unk_ent_id] = unk_ent_wiki_id
	wikiid2id[unk_ent_wiki_id] = unk_ent_id
	
	doc_count = 1

	with open(wiki_path, 'r', encoding='utf8') as inf:

		for line in inf:

			wiki_id, doc_title = extract_doc_tag(line)

			if wiki_id >= 0:

				name2wikiid[doc_title] = wiki_id
				wikiid2name[wiki_id] = doc_title
				id2wikiid[doc_count] = wiki_id
				wikiid2id[wiki_id] = doc_count
				doc_count += 1
				
				if doc_count % 10000 == 0:
					show_progress(percent=doc_count/ent_max)

				if doc_count == ent_max:
					break

	show_progress(percent=1.0)
	ent_size = doc_count
	
	print(f'done!\n   {doc_count} entities loaded!')
	
	return name2wikiid, wikiid2name, id2wikiid, wikiid2id


def gen_entity_mentions_map(wiki_path, name2wikiid, wikiid2id, ent_max):

	hyp_pattern = re.compile(r'<a[^>]*href=\"([^\">]+)\"[^>]*>([^>]+)</a>', re.DOTALL | re.UNICODE)
	
	ignored_names= set()
	wikiid2mentions = {}
	
	with open(wiki_path, 'r', encoding='utf8') as inf:
	
		for line in inf:
		
			if len(wikiid2mentions) % 100 < 10:
				show_progress(percent=len(wikiid2mentions)/ent_max)
		
			clean_text = wikicorpus.filter_wiki(line)
			hyp_matches = re.finditer(hyp_pattern, line)


			for link in hyp_matches:

				name = unquote(link.groups()[0])   # wikipedia url id
				mention = link.groups()[1]
				
				if name in name2wikiid:
					wikiid = name2wikiid[name]
					if wikiid not in wikiid2mentions:
						wikiid2mentions[wikiid] = []
					wikiid2mentions[wikiid].append(mention)
				else:
					ignored_names.add(name)
	
	return wikiid2mentions

def load_entity_id_maps(ent_pickle_path='entites.pickle', ent_path='enwiki_full.txt', ent_max=10**7):

	if os.path.exists(ent_pickle_path):
		
		with open(ent_pickle_path, 'rb') as inf:
			name2wikiid = pickle.load(inf)
			wikiid2name = pickle.load(inf)
			id2wikiid = pickle.load(inf)
			wikiid2id = pickle.load(inf)

		print(f'loaded entities from {ent_pickle_path}')   
		
	else:
			
		name2wikiid, wikiid2name, id2wikiid, wikiid2id = gen_entity_id_maps(ent_path, ent_max)

		with open(ent_pickle_path, 'wb') as outf:
			pickle.dump(name2wikiid, outf)
			pickle.dump(wikiid2name, outf)
			pickle.dump(id2wikiid, outf)
			pickle.dump(wikiid2id, outf)

		print(f'entites processed and saved to {ent_pickle_path}')

	
	return name2wikiid, wikiid2name, id2wikiid, wikiid2id