import os
import json
import jsonlines
from tqdm import tqdm
import argparse
import base64
from os import path as osp
import sys
sys.path.append('src')
from utils import Agent, name_to_path, DeepInfra_Agent
from prompt import text_system_prompt, text_user_prompt


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-concept interaction generation")
    parser.add_argument('--shard_num', type=int, default=1, help='Total number of shards')
    parser.add_argument('--shard_index', type=int, default=0, help='Index of this shard (0-based)')
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    if args.model not in ['gpt-5-mini', 'gpt-5-nano', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'google/gemma-3-27b-it', 'google/gemma-3-12b-it', 'google/gemma-3-4b-it']:
        raise ValueError("Not Supported Model Type")
    
    concept_to_path = name_to_path()
    
    with jsonlines.open('data/benchmark/text_prompt_qa.jsonl') as f:
        dataset = list()
        for item in f.iter():
            dataset.append(item)
    
    # Shard the list manually without using datasets
    total = len(dataset)
    shard_size = (total + args.shard_num - 1) // args.shard_num  # ceil division
    start = args.shard_index * shard_size
    end = min(start + shard_size, total)
    dataset = dataset[start:end]
    
    target_path = f'output/text_prompt_qa/{args.model}/{args.shard_index}.jsonl'
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    f = open(target_path, 'w')
    
    print(f'Saved in {target_path}')
    
    for data in tqdm(dataset):
        history = data['history']
        QAs = data['QA']
        gt = data['gt']
        img_path = osp.join('data', data['image_path'])
        concepts = data['concepts']
        
        content = list()
        
        content.append({
            'type': 'text', 'text': '### Query image\n\n',
        })
        content.append({
            'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{encode_image(img_path)}"}
        })
        content.append({
            'type': 'text', 'text': '### Concepts\n\n',
        })
        
        for concept, textual_memory in history.items():
            content.append({
                'type': 'text', 'text': f'<{concept}>\n',
            })
            content.append({
                'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{encode_image(concept_to_path[concept])}"}
            })
            content.append({
                'type': 'text', 'text': f'{textual_memory}\n\n',
            })
        
        content.append({
            'type': 'text', 'text': text_user_prompt
        })
        
        
        for qa in QAs:
            
            question = qa['question']
            answer = qa['answer']
            
            content.append({
                'type': 'text', 'text': f'###Question\n{question}\n'
            })
            
            messages = [
                {
                    'role': 'system',
                    'content': text_system_prompt
                },
                {
                    'role': 'user',
                    'content': content
                }
            ]
            
            if 'gemma' in args.model:
                agent = DeepInfra_Agent()
            else:
                agent = Agent(
                    model_info=args.model
                )
            
            if args.model == 'gemini-2.5-flash':
                response = agent.client.chat.completions.create(
                    model=args.model,
                    reasoning_effort='low',
                    messages=messages,
                    temperature=1.0
                )
            else:
                response = agent.client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=1.0,
                )
            
            response = response.choices[0].message.content
            
            content.pop()
            f.write(json.dumps({
                'img_path': img_path,
                'question': question,
                'answer': answer,
                'response': response,
            }) + '\n')
            f.flush()
    f.close()
