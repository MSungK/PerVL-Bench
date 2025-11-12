import jsonlines
import json
import re
import argparse
import sys
from os import path as osp
import os
from tqdm import tqdm
from collections import defaultdict
sys.path.append('src')
from utils import Agent
from prompt import eval_system_prompt, eval_user_prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--shard_num', type=int, default=1, help='Total number of shards')
    parser.add_argument('--shard_index', type=int, default=0, help='Index of this shard (0-based)')
    args = parser.parse_args()
    
    file_name = osp.splitext(osp.basename(args.input))[0]
    # type_name = osp.dirname(args.input).split('/')[-1]
    
    # target_path = osp.join('vp_result', type_name, file_name, f'{args.shard_index}.jsonl')
    target_path = osp.join('eval_result', file_name, f'{args.shard_index}.jsonl')

    with jsonlines.open(args.input) as f:
        items = [item for item in f.iter()]
    
    q_to_a = dict()
    
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    writer = open(target_path, 'w')
    
    with jsonlines.open('data/benchmark/multimodal_prompt_qa.jsonl') as f:
        for item in f.iter():
            for qa in item['QA']:
                q_to_a[qa['question']] = qa['answer']
    
    assert len(items) == len(q_to_a)
    
    total = len(items)
    shard_size = (total + args.shard_num - 1) // args.shard_num  # ceil division
    start = args.shard_index * shard_size
    end = min(start + shard_size, total)
    items = items[start:end]
    
    print(f"Saved in {target_path}")
    
    for item in tqdm(items):
        original_question = item['original_question']
        question = item['question']
        history = item['history']
        response = item['response']
        all_history = ""
        
        for k, v in history.items():
            if k not in item['gt']: continue
            all_history += f'<{k}>\n{v}\n\n'
        
        gpt_answer = q_to_a[original_question]
        
        agent = Agent(
            model_info='gpt-5-mini'
        )
        prompt = eval_user_prompt.format(
            all_history=all_history,
            original_question=original_question,
            question=question,
            gpt_answer=gpt_answer,
            model_answer=response
        )
        messages = [
            {
                'role': 'system',
                'content': eval_system_prompt
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
        result = agent.client.chat.completions.create(
            model='gpt-5-mini',
            messages=messages,
            temperature=1.0,
        )
        result = result.choices[0].message.content
        writer.write(json.dumps({
            'original_question': original_question,
            'question': question,
            'history': history,
            'gpt_answer': gpt_answer,
            'model_answer': response,
            'result': result
        }) + '\n')
        writer.flush()
    writer.close()