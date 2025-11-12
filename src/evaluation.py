import jsonlines
import json
import re
import argparse
from os import path as osp
import os
from tqdm import tqdm


def keep_only_letters(s: str) -> str:
    """
    Keep only English alphabet letters and '-' from the input string.
    """
    return re.sub(r'[^A-Za-z\-]', '', s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()
    
    
    if args.type=='text_prompt_qa':
        all_pred = list()
        all_gold = list()
        
        single_pred = list()
        single_gold = list()
        multi_pred = list()
        multi_gold = list()
        
        image_to_mode = dict()
        
        with jsonlines.open('data/benchmark/text_prompt_qa.jsonl') as f:
            for item in f.iter():
                image_path = osp.join('data', item['image_path'])
                image_to_mode[image_path] = 1 if len(item['gt'])==1 else 2
        
        with jsonlines.open(args.input) as f:
            for item in f.iter():
                response = str(item['response'])
                
                gold_set = set([str(x).lower() for x in item['answer']])
                
                preds = response.split(',')
                pred_set = set()
                for pred in preds:
                    pred = keep_only_letters(pred).lower()
                    pred_set.add(pred)
                
                if image_to_mode[item['img_path']] == 1:
                    single_pred.append(pred_set)
                    single_gold.append(gold_set)
                else:
                    multi_pred.append(pred_set)
                    multi_gold.append(gold_set)
                
                all_pred.append(pred_set)
                all_gold.append(gold_set)
        
        tp = fp = fn = 0
        for pred_set, gold_set in zip(single_pred, single_gold):
            tp += len(pred_set & gold_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        p = round(p * 100, 2)
        r = round(r * 100, 2)
        f1 = round(f1 * 100, 2)
        print('Single')
        print("P={:.2f} R={:.2f} F1={:.2f}".format(p, r, f1))
        
        tp = fp = fn = 0
        for pred_set, gold_set in zip(multi_pred, multi_gold):
            tp += len(pred_set & gold_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        p = round(p * 100, 2)
        r = round(r * 100, 2)
        f1 = round(f1 * 100, 2)
        print('Multi')
        print("P={:.2f} R={:.2f} F1={:.2f}".format(p, r, f1))
        
        tp = fp = fn = 0
        for pred_set, gold_set in zip(all_pred, all_gold):
            tp += len(pred_set & gold_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        p = round(p * 100, 2)
        r = round(r * 100, 2)
        f1 = round(f1 * 100, 2)
        print('Total')
        print("P={:.2f} R={:.2f} F1={:.2f}".format(p, r, f1))
    
    elif args.type=='multimodal_prompt_qa':
        question_to_mode = dict()
        with jsonlines.open('data/benchmark/multimodal_prompt_qa.jsonl') as f:
            for item in f.iter():
                mode = 1 if len(item['gt'])==1 else 2
                for qa in item['QA']:
                    question_to_mode[qa['question']] = mode
        
        def extract_better_response(text: str) -> str:
            """
            Extracts 'A', 'B', or 'Tie' from a line like 'Better Response: A'
            """
            match = re.search(r"Better Response:\s*(A|B|Tie)", text, re.IGNORECASE)
            if match:
                return match.group(1)
            else:
                print(text)
                raise ValueError("Could not find a valid better response in the input.")
        from collections import defaultdict
        total_counter = defaultdict(int)
        
        single_counter = defaultdict(int)
        multi_counter = defaultdict(int)
        
        total_cnt = 0
        single_cnt = 0
        multi_cnt = 0
        with jsonlines.open(args.input) as f:
            for item in f.iter():
                total_cnt += 1
                result = item['result']
                result = extract_better_response(result)
                total_counter[result] += 1
                
                if question_to_mode[item['original_question']]==1:
                    single_counter[result] += 1
                    single_cnt += 1
                else:
                    multi_counter[result] += 1
                    multi_cnt += 1
        
        assert total_cnt == single_cnt + multi_cnt 
        
        print("Single")
        score = (single_counter['B'] + 0.5 * single_counter['Tie']) / single_cnt * 100
        score = round(score, 2)
        print("{:.2f}".format(score))
        
        print("Multi")
        score = (multi_counter['B'] + 0.5 * multi_counter['Tie']) / multi_cnt * 100
        score = round(score, 2)
        print("{:.2f}".format(score))
        
        print("Total")
        score = (total_counter['B'] + 0.5 * total_counter['Tie']) / total_cnt * 100
        score = round(score, 2)
        print("{:.2f}".format(score))
        
    else:
        raise ValueError("Not Supported Evaluation Type")