from file_io import *

import string
PUNCTS = string.punctuation + '”“¿?.�✔✅⤵➕➖⛔✍⃣.-'
PUNCTS = list(PUNCTS) # punctuations

import spacy
nlp_en = spacy.load('en_core_web_md')

from spacy.lang.vi import Vietnamese # https://github.com/trungtv/vi_spacy
nlp_vi = Vietnamese()

def count_vocab_size(input_file = 'dataset/collected_data.json'):

    dataset = read_list_from_jsonl_file(input_file)
    vocab_en_dict = {}
    vocab_vi_dict = {}

    for item in dataset:

        source = item['source']
        doc_en = nlp_en(source)
        for token in doc_en:
            word = token.text.strip()
            if (word == '' or word in PUNCTS): continue
            
            if (word not in vocab_en_dict):
                vocab_en_dict[word] = 1
            else:
                vocab_en_dict[word] += 1
        
        target = item['target']
        doc_vi = nlp_vi(target)
        for token in doc_vi:
            word = token.text.strip()
            if (word == '' or word in PUNCTS): continue
            
            if (word not in vocab_vi_dict):
                vocab_vi_dict[word] = 1
            else:
                vocab_vi_dict[word] += 1
                
    print('vocab_en_dict: ', len(vocab_en_dict))
    vocab_en_dict = dict(sorted(vocab_en_dict.items(), key=lambda item: item[1], reverse = True))
    write_list_to_json_file('dataset/vocab_en.json', vocab_en_dict, file_access = 'w')
    
    print('vocab_vi_dict: ', len(vocab_vi_dict))
    vocab_vi_dict = dict(sorted(vocab_vi_dict.items(), key=lambda item: item[1], reverse = True))
    write_list_to_json_file('dataset/vocab_vi.json', vocab_vi_dict, file_access = 'w')
    
#...............................................................................            
if __name__ == "__main__":
    count_vocab_size()