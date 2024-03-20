# Introduction
This repo is to translate Wikipedia categories from English to Vietnamese.

# Models
* Helsinki-NLP/opus-mt-en-vi
* VietAI/envit5-translation (https://huggingface.co/VietAI/envit5-translation)
* mbart
* mt5
* facebook/m2m100_418M

# Train models

### Helsinki-NLP/opus-mt-en-vi

python seq2seq.py --mode "train" --model_name "Helsinki-NLP/opus-mt-en-vi" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix ""

python seq2seq.py --mode "test" --model_name "Helsinki-NLP/opus-mt-en-vi" --model_path "opus-mt-en-vi\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix ""

### VietAI/envit5-translation

python seq2seq.py --mode "train" --model_name "VietAI/envit5-translation" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix ""

python seq2seq.py --mode "test" --model_name "VietAI/envit5-translation" --model_path "envit5-translation\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix ""

### facebook/mbart-large-50

python seq2seq.py --mode "train" --model_name "facebook/mbart-large-50" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix ""

python seq2seq.py --mode "test" --model_name "facebook/mbart-large-50" --model_path "google_mt5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix ""

### google/mt5-base (bad results)

python seq2seq.py --mode "train" --model_name "google/mt5-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: "

python seq2seq.py --mode "test" --model_name "google/mt5-base" --model_path "google_mt5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "summarize: "

### facebook/m2m100_418M
python seq2seq.py --mode "train" --model_name "facebook/m2m100_418M" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix ""

python seq2seq.py --mode "test" --model_name "facebook/m2m100_418M" --model_path "m2m100_418M\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix ""

# Contact
* tahoangthang@gmail.com
* https://www.tahoangthang.com
