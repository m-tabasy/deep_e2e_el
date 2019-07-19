import numpy as np
import re


unk_word, unk_word_id = 'UNK_W', 0
iob_o, iob_b, iob_i = 0, 1, 2

def split_aida(aida_path='aida-yago2-dataset.tsv'):

	train = 'aida-train.tsv'
	testa, testb = 'aida-testa.tsv', 'aida-testb.tsv'
	current = open(train, 'w')

	with open(aida_path, 'r') as f:
		for line in f:

			if line.find('947testa CRICKET') > 0:
				current.close()
				current = open(testa, 'w')   

			if line.find('1163testb SOCCER') > 0:
				current.close()
				current = open(testb, 'w')  

			current.write(line)
		current.close()

	print(f'aida splitted to {train}, {testa}, {testb}')
  
  
def process_token(token, lower=False):
	
	if token in ['.', ',', '-']:
		return None
		
	if lower:
		token = token.lower()
		
	if token in ['\n']:
		token = '</s>'
	
	if token in ['to', 'of']:
		token = token.capitalize()
		
	if len(token) > 1:
		token = re.sub(r'[0-9]', '#', token)
	
	return token
		
		
def process_tag(wiki_id):
	return int(wiki_id)
	

def extract_word_ent(line, word2id, wiki2id):

	if line.count('\t') <= 4:
		token, wiki_id = line.split('\t')[0].strip(), -1
	else:
		token, iob, mention, yago2, wiki_url, wiki_id, *freebase_mid = line.split('\t')
					

	if process_token(token) in word2id:
		word_id = word2id[process_token(token)]
		
	elif process_token(token, lower=True) in word2id:
		word_id = word2id[process_token(token, lower=True)]
		
	elif len(token) > 2:				# UNK
		word_id = unk_word_id
		
	else:					# skip punctuations... which map to no vector
		word_id = None
			
	if process_tag(wiki_id) in wiki2id:
		ent_id = wiki2id[process_tag(wiki_id)]
	else:
		ent_id = None
			
	return word_id, ent_id

	
def gen_el_data(aida_path, word2id, wiki2id, e2v):
	
	f = open(aida_path, 'r');
	f.readline()
	eof = False
	
	while not eof:
		tokens = []
		tags = []
		
		f.readline()
		f.readline()
		line = f.readline()
		
		while not line.startswith('-DOCSTART-'):
		
			word_id, ent_id = extract_word_ent(line, word2id, wiki2id)
			
			if word_id is not None and ent_id is not None:
				tokens.append(word_id)
				tags.append(e2v[ent_id])
			
			line = f.readline()
			if line == '':
				eof = True
				break
		
		yield np.array(tokens), np.array(tags)
		
	f.close()	


def extract_word_ent(line):

	if line.count('\t') < 2:
		token  = line.split('\t')[0].strip()
		iob, wiki_id = iob_o, -1
	elif line.count('\t') <= 4:
		token, iob, *_  = line.split('\t')
		wiki_id = -1
	else:
		token, iob, mention, yago2, wiki_url, wiki_id, *freebase_mid = line.split('\t')
		
	iob = process_iob(iob)
	wiki_id = int(wiki_id)
					
	return token, iob, wiki_id
	
	
def gen_el_data_vecs(aida_path, word_vecs, wiki2id, ent_vecs):
	
	f = open(aida_path, 'r');
	f.readline()
	eof = False
	
	while not eof:
		tokens = []
		iobs = []
		mentions = []
		
		f.readline()
		f.readline()
		line = f.readline()
		
		while not line.startswith('-DOCSTART-'):
		
			token, iob, wiki_id = extract_word_ent(line)
			
			if token not in word_vecs:
				token = 'unk'						#
			
			if wiki_id in wiki2id:
				ent_id = wiki2id[wiki_id]
			else:
				ent_id = 0							#
			
			tokens.append(word_vecs[token])
			iobs.append(iob)
			mentions.append(ent_vecs[ent_id])
			
			line = f.readline()
			if line == '':
				eof = True
				break
		
		yield np.array(tokens), np.array(iobs), np.array(mentions)
		
	f.close()	


def process_iob(iob):
	if iob in [iob_b, iob_i, iob_o]:
		return iob
	if iob.lower() == 'b':
		return iob_b
	elif iob.lower() == 'i':
		return iob_i
	else:
		return iob_o


def extract_word_iob(line, word2id):

	if line.count('\t') < 2:
		token, iob = line.split('\t')[0].strip(), iob_o
	else:
		token, iob, mention, *extras = line.split('\t')
					

	if process_token(token) in word2id:
		word_id = word2id[process_token(token)]
		
	elif process_token(token, lower=True) in word2id:
		word_id = word2id[process_token(token, lower=True)]
		
	elif len(token) > 2:				# UNK
		word_id = unk_word_id
		
	else:					# skip punctuations... which map to no vector
		word_id = None

	iob = process_iob(iob)
			
	return word_id, iob


def gen_md_data(aida_path, word2id):

	f = open(aida_path, 'r');
	f.readline()
	eof = False
	
	while not eof:
		tokens = []
		tags = []
		
		f.readline()
		f.readline()
		line = f.readline()
		
		while not line.startswith('-DOCSTART-'):
		
			word_id, iob = extract_word_iob(line, word2id)
			
			if word_id is not None:
				tokens.append(word_id)
				tags.append(iob)
			
			line = f.readline()
			if line == '':
				eof = True
				break
		
		yield np.array(tokens), np.array(tags)
		
	f.close()		
	

def gen_mentions(aida_path):

	f = open(aida_path, 'r');
	
	for line in f:
		if line.count('\t') > 4:

			token, iob, mention, yago2, wiki_url, wiki_id, *freebase_mid = line.split('\t')
			yield token, int(wiki_id)
		
	f.close()		
	
	
def extract_word_iob_wikiid(line, word2id):

	if line.count('\t') <= 4:
		token, wiki_id, iob = line.split('\t')[0].strip(), -1, iob_o
	else:
		token, iob, mention, yago2, wiki_url, wiki_id, *freebase_mid = line.split('\t')
					

	if process_token(token) in word2id:
		word_id = word2id[process_token(token)]
		
	elif process_token(token, lower=True) in word2id:
		word_id = word2id[process_token(token, lower=True)]
		
	elif len(token) > 2:				# UNK
		word_id = unk_word_id
		
	else:					# skip punctuations... which map to no vector
		word_id = None
			
	return word_id, process_tag(wiki_id), process_iob(iob)
	

def get_gold_tags(wids, iobs, wikiids):

	gold = {}

	start, end = -1, -1

	for idx, tag in enumerate(iobs):

		if tag in [iob_o, iob_b] and start > 0:
			gold[f'{start}:{end}'] = wikiids[start]
			start, end = -1, -1

		if tag == iob_b:    # iob_b
			start, end = idx, idx
		if tag == iob_i:
			end = idx

	return gold
	
	
def gen_doc_with_golds(aida_path, word2id):
	
	f = open(aida_path, 'r')
	
	f.readline()
	eof = False
	
	while not eof:
		word_ids = []
		iobs = []
		wiki_ids = []
		
		f.readline()
		f.readline()
		line = f.readline()
		
		while not line.startswith('-DOCSTART-'):
		
			word_id, wiki_id, iob = extract_word_iob_wikiid(line, word2id)
			
			if word_id is not None:
				word_ids.append(word_id)
				wiki_ids.append(wiki_id)
				iobs.append(iob)
			
			line = f.readline()
			if line == '':
				eof = True
				break
		

		yield np.array(word_ids), get_gold_tags(word_ids, iobs, wiki_ids)
		
		
	f.close()	
	

def extract_token_iob_wikiid(line):

	if line.count('\t') <= 4:
		token, wiki_id, iob = line.split('\t')[0].strip(), -1, iob_o
	else:
		token, iob, mention, yago2, wiki_url, wiki_id, *freebase_mid = line.split('\t')
					
	return token, process_tag(wiki_id), process_iob(iob)
	
	
def gen_tokens_with_golds(aida_path):
	
	f = open(aida_path, 'r')
	
	f.readline()
	eof = False
	
	while not eof:
		tokens = []
		iobs = []
		wiki_ids = []
		
		f.readline()
		f.readline()
		line = f.readline()
		
		while not line.startswith('-DOCSTART-'):
		
			token, wiki_id, iob = extract_token_iob_wikiid(line)
			
			tokens.append(token)
			wiki_ids.append(wiki_id)
			iobs.append(iob)
			
			line = f.readline()
			if line == '':
				eof = True
				break

		yield np.array(tokens), get_gold_tags(None, iobs, wiki_ids)
		
	f.close()	
	
	
def check_aida_entities():
	# train_ids = set()

	# with open('aida-train.tsv', 'r') as f:
	  # for line in f:
		# if line.count('\t') > 4:
		  # token, iob, mention, yago2, wiki_url, wiki_id, *freebase_mid = line.split('\t')
		  # train_ids.add(int(wiki_id))

	# testa_ids = set()
	# new_ids_a = set()

	# with open('aida-testa.tsv', 'r') as f:
	  # for line in f:
		# if line.count('\t') > 4:
		  # token, iob, mention, yago2, wiki_url, wiki_id, *freebase_mid = line.split('\t')
		  # testa_ids.add(int(wiki_id))
		  # if int(wiki_id) not in train_ids:
			# new_ids_a.add(int(wiki_id))

	# print(f'there are {len(new_ids_b)} new entities from total {len(testb_ids)} in test b: {100*len(new_ids_a)//len(testa_ids)}%')
	# testb_ids = set()
	# new_ids_b = set()

	# with open('aida-testb.tsv', 'r') as f:
	  # for line in f:
		# if line.count('\t') > 4:
		  # token, iob, mention, yago2, wiki_url, wiki_id, *freebase_mid = line.split('\t')
		  # testb_ids.add(int(wiki_id))
		  # if int(wiki_id) not in train_ids:
			# new_ids_b.add(int(wiki_id))

	print(f'there are {len(new_ids_b)} new entities from total {len(testb_ids)} in test b: {100*len(new_ids_b)//len(testb_ids)}%')
		