from wiki_core import *
from file_io import *

import gc
import re
from random import randrange
import sys
sys.setrecursionlimit(10**6)

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

wikidata_api = Wikidata()
wikipedia_api = Wikipedia()

def write_index(index, file = 'dataset/index.txt'):
    write_to_text_file(file, index)

def check_index(index, file = 'dataset/index.txt'):

    index_list = []

    try:
        index_list = read_list_from_text_file(file)
    except:
        pass
    
    if (index in index_list): return True
    return False
    
def collect_single(output_file = 'dataset/collected_data.json'):

   
    result_dict = {}
    wikidata_id = ''
    
    while(True):

        number_id = randrange(9900001) # there are over 99 millions items actually
        wikidata_id = 'Q' + str(number_id)

        # avoid scanning again
        if (check_index(wikidata_id) == True): continue;
        
        item_dict = {}
        try: 
            item_dict = wikidata_api.get_item_by_id(wikidata_id)
        except: 
            pass

        if not item_dict:
            write_index(wikidata_id)
            continue
        
        print('wikidata_id: ', item_dict['wikidata_id'])
        sitelinks = item_dict['sitelinks']

        en_cat, vi_cat = '', ''
        try:
            en_cat = sitelinks['enwiki']
            vi_cat = sitelinks['viwiki']
        except:
            pass

        # continue conditions
        if (en_cat == '' or vi_cat == ''): continue;
        if (en_cat == 'Category:' or vi_cat == 'Category:'): continue;

        # break conditions
        if (en_cat != '' and vi_cat != '' and 'Category:' in en_cat):
            result_dict['wikidata_id'] = item_dict['wikidata_id']
            result_dict['source'] = en_cat.replace('Category:','')
            result_dict['target'] = vi_cat.replace('Thể loại:','')
            break
        
        write_index(wikidata_id) # write index
        
    print('en_cat: ', en_cat)
    print('vi_cat: ', vi_cat)
    print('---------------------------------------')
    print('---------------------------------------')

    write_single_dict_to_jsonl_file(output_file, result_dict)
    write_index(wikidata_id)
    gc.collect()

def collect_multi(max_workers = 16, limit = 1000000, output_file = 'dataset/collected_data.json'):

    number_list = [i for i in range(1, limit + 1)]

    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        results = executor.map(collect_single, [output_file]*len(number_list), timeout = 600)

#........................................
if __name__ == "__main__":
    collect_multi()
 
    #collect_single()