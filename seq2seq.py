import argparse
import datasets
import evaluate
import glob
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import random
import re
import torch

from datasets import Dataset, load_dataset, concatenate_datasets
from file_io import *
from huggingface_hub import HfFolder
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score, recall_score, precision_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils import *

nltk.download("punkt")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('CUDA: ', torch.cuda.is_available())

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

def preprocess_function(sample, padding="max_length"):
    
    # Add prefix for T5 models
   
    inputs = [source_prefix + item for item in sample[source_column]]
    print('inputs: ', inputs[0])

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    
    # Tokenize labels
    labels = tokenizer(text_target=sample[target_column], max_length=max_target_length, padding=padding, truncation=True)
    
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss. 
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# helper function to postprocess text
def postprocess_text(preds, labels):

    print('-----------------------')
    print('labels: ', labels[0])
    print('preds: ', preds[0])
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # RougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):

    preds, labels = eval_preds

    # Replace -100 in the labels as we can't decode them.
    #if isinstance(preds, tuple): preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces = True)
    #print('decoded_preds[0]: ', decoded_preds[0])
    
    # Replace -100 in the labels as we can't decode them.
    #print('tokenizer.pad_token_id: ', tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #print('decoded_labels[0]: ', decoded_labels[0])

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    #predictions = ["hello there", "general kenobi"]
    #references = ["hello there", "general kenobi"]
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    print('rouge_result: ', rouge_result)
    
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    print('bleu_result: ', bleu_result['bleu'])
    
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    print('meteor_result: ', meteor_result['meteor'])
    
    result = {}
    result["rouge1"] = rouge_result['rouge1']
    result["rouge2"] = rouge_result['rouge2']
    result["rougeL"] = rouge_result['rougeL']
 
    result["bleu"] = bleu_result['bleu']
    result["meteor"] = meteor_result['meteor']

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result


def train(train_set, val_set, test_set, tokenizer, model, model_name = 'facebook/bart-base', max_source_length = 64, max_target_length = 8, epochs = 40, batch_size = 4, source_column = 'source', target_column = 'target'):
    
    # Load dataset 
    train_df = pd.DataFrame(train_set)
    val_df = pd.DataFrame(val_set)
    test_df = pd.DataFrame(test_set)
    
    '''if ('m2m100' in model_name):
        tokenizer.src_lang = "en"'''

    # The maximum total input sequence length after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    #tokenized_inputs = concatenate_datasets([train_set, test_set]).map(lambda x: tokenizer(x[source_column], truncation=True), batched=True, remove_columns=train_set.column_names)
    #print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    #tokenized_targets = concatenate_datasets([train_set, test_set]).map(lambda x: tokenizer(x[target_column], truncation=True), batched=True, remove_columns=train_set.column_names)
    #max_target_length = max([len(str(x)) for x in tokenized_targets["input_ids"]])

    #print('Tokenized targets: ', tokenized_targets)
    #print(f"Max target length: {max_target_length}")

    tokenized_train_dataset = train_set.map(preprocess_function, batched=True, remove_columns=train_set.column_names)
    print(f"Keys of tokenized dataset: {list(tokenized_train_dataset.features)}")

    tokenized_val_dataset = val_set.map(preprocess_function, batched=True, remove_columns=val_set.column_names)
    print(f"Keys of tokenized dataset: {list(tokenized_val_dataset.features)}")
    
    # Load model from the hub
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    #model = model.to(device)

    # Ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    
    # Data collator for boosting training speed
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
        )

    # Hugging Face repository id
    repository_id = ''
    try:
        repository_id = f"{model_name.split('/')[1]}"
    except:
        repository_id = f"{model_name}"

    # fp16
    fp16_value = False 
    if (torch.cuda.is_available() == True and 't5' not in model_name): fp16_value = True
    
    # Define training args
    training_args = Seq2SeqTrainingArguments(
        gradient_accumulation_steps = 4,
        #gradient_checkpointing=True,
        output_dir=repository_id,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        fp16 = fp16_value,  # "cuda" device only
        #learning_rate=3e-4,
        num_train_epochs=epochs,
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="epoch", 
        # logging_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="eval_meteor",
        # push to hub parameters
        report_to="tensorboard",
        generation_max_length = max_target_length,
        #push_to_hub=True,
        #hub_strategy="every_save",
        #hub_model_name=repository_id,
        #hub_token=HfFolder.get_token(),
        )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        )

    trainer.train()
    trainer.evaluate()

def test(dataset, model_name, model, tokenizer, input_file = 'dataset/test.json', \
                     batch_size = 4, max_len = 32, min_len = 1, source_column = 'source', target_column = 'target', decode_pred = False):

    
    if (len(dataset) == 0): # load dataset if not given
        dataset = read_list_from_jsonl_file(input_file)
    
    # Add source_prefix
    for item in dataset:
        item[source_column] = source_prefix + item[source_column]
    
    pred_list = []
    for i in range(0, len(dataset), batch_size):

        n_batch = 0
        if (len(dataset)%batch_size != 0): n_batch = len(dataset)//batch_size + 1
        else: n_batch = len(dataset)//batch_size

        sys.stdout.write('Infer batch: %d/%d \t Model: %s \r' % (i//batch_size + 1, n_batch, model_name))
        sys.stdout.flush()
        
        subset = dataset[i:i + batch_size]
        texts = [item[source_column] for item in subset]

        inputs = tokenizer(texts, padding = "max_length", truncation = True, max_length = max_len, \
                           return_tensors = 'pt').to(device)
                           
        outputs = []
        with torch.no_grad():
            # use greedy algorithm
            outputs = model.generate(**inputs, max_length = max_len, min_length = min_len, \
                                         num_beams = 4, do_sample = False, return_dict_in_generate = True, output_scores = True)  

        preds = tokenizer.batch_decode(outputs.sequences, skip_special_tokens = True)
        
        preds = [[x] for x in preds]
        pred_list += preds

    pred_list = [[x for x in pred][0].strip() for pred in pred_list] # use strip() to remove spaces
    label_list = [item[target_column] for item in dataset]

    # decode prediction
    if (decode_pred == True):
        pred_list = [decode_vi(pred) for pred in pred_list]
    
    # Use simple post-processing
    decoded_preds, decoded_labels = postprocess_text(pred_list, label_list)
    
    # Calculate metrics
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    print('rouge_result: ', rouge_result)
    
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    print('bleu_result: ', bleu_result['bleu'])
    
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    print('meteor_result: ', meteor_result['meteor'])
    
    result = {}
    result["rouge1"] = rouge_result['rouge1']
    result["rouge2"] = rouge_result['rouge2']
    result["rougeL"] = rouge_result['rougeL']
 
    result["bleu"] = bleu_result['bleu']
    result["meteor"] = meteor_result['meteor']
    
    # Write predictions to file
    for item, pred in zip(dataset, pred_list):
        item['prediction'] =  pred
    
    repository_id = ''
    try:
        repository_id = f"{model_name.split('/')[1]}"
    except:
        repository_id = f"{model_name}"
        
    dataset.append(result)
    write_list_to_jsonl_file('dataset/test_pred_' + repository_id + '.json', dataset, 'w')
        
    print('result: ', result)
    return result

def main(args):
    if (args.mode == 'train'):
        train_set = datasets.load_dataset('json', data_files = args.train_path, split="train")
        test_set = datasets.load_dataset('json', data_files = args.test_path, split="train")
        val_set = datasets.load_dataset('json', data_files = args.val_path, split="train")

        print(f"Train dataset size: {len(train_set)}")
        print(f"Test dataset size: {len(test_set)}")
        print(f"Val dataset size: {len(val_set)}")

        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model.to(device)
        
        train(train_set, val_set, test_set, tokenizer, model, model_name = args.model_name, max_source_length = args.max_source_length, max_target_length = args.max_target_length, epochs = args.epochs, batch_size = args.batch_size)
    
    elif (args.mode == 'test'):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        model.to(device)
        model.eval()
        
        decode_pred = (args.decode_pred == 1)
        
        test([], args.model_name, model, tokenizer, input_file = args.test_path, batch_size = args.test_batch_size, max_len = args.max_source_length, min_len = args.min_target_length, decode_pred = decode_pred)

#...............................................................................            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--model_name', type=str, default='facebook/bart-base') # or test
    parser.add_argument('--train_path', type=str, default='dataset/train.json') 
    parser.add_argument('--test_path', type=str, default='dataset/test.json')
    parser.add_argument('--val_path', type=str, default='dataset/val.json')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--max_source_length', type=int, default=32)
    parser.add_argument('--max_target_length', type=int, default=32)
    parser.add_argument('--min_target_length', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='bart-base\checkpoint-xxx')
    parser.add_argument('--source_prefix', type=str, default='summarize: ') # only for T5 models
    parser.add_argument('--source_column', type=str, default='source') 
    parser.add_argument('--target_column', type=str, default='target') 
    parser.add_argument('--decode_pred', type=int, default=0) 
    
    args = parser.parse_args()
    
    global source_prefix
    source_prefix = args.source_prefix
    
    #global model_name
    #model_name = args.model_name
    
    global tokenizer
    if ('m2m100' in args.model_name):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case = False, add_prefix_space = True, src_lang='en', tgt_lang='vi')
    elif ('mbart' in args.model_name):
         tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case = False, add_prefix_space = True, src_lang='en_XX', tgt_lang='vi_VN')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case = False, add_prefix_space = True)
    
    global source_column
    source_column = args.source_column
    
    global target_column
    target_column = args.target_column
    
    global max_source_length
    max_source_length = args.max_source_length
    
    global max_target_length
    max_target_length = args.max_target_length
    
    main(args)
    
# python seq2seq.py --mode "train" --model_name "Helsinki-NLP/opus-mt-en-vi" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 16 --source_prefix "" --source_column "source" --target_column "target"
# python seq2seq.py --mode "test" --model_name "Helsinki-NLP/opus-mt-en-vi" --model_path "opus-mt-en-vi\checkpoint-1738" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target"
        
# python seq2seq.py --mode "train" --model_name "google-t5/t5-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: " --source_column "source" --target_column "target_encoded"        
# python seq2seq.py --mode "test" --model_name "google-t5/t5-base" --model_path "t5-base\checkpoint-2250" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "summarize: " --source_column "source" --target_column "target" --decode_pred 1