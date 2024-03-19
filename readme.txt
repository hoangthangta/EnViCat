--------------------------------
Helsinki-NLP/opus-mt-en-vi

# python seq2seq.py --mode "train" --model_name "Helsinki-NLP/opus-mt-en-vi" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix ""

# python seq2seq.py --mode "test" --model_name "Helsinki-NLP/opus-mt-en-vi" --model_path "opus-mt-en-vi\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix ""

--------------------------------

--------------------------------
google/mt5-base

# python seq2seq.py --mode "train" --model_name "google/mt5-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: "

# python seq2seq.py --mode "test" --model_name "google/mt5-base" --model_path "google_mt5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "summarize: "

--------------------------------

# python seq2seq.py --mode "train" --model_name "lewtun/tiny-random-mt5" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: "

--------------------------------
# get all categories by Wikidata Query Service

SELECT ?item ?itemLabel ?sitelink
WHERE
{
  ?item wdt:P31 wd:Q4167836. 

  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } 
}
LIMIT 10
--------------------------------

https://huggingface.co/VietAI/envit5-translation

facebook/m2m100_418M
mbart
VietAI/envit5-translation
mt5
Helsinki-NLP/opus-mt-en-vi
