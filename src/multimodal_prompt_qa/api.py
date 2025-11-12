import json
import jsonlines
from tqdm import tqdm
import argparse
import base64
from os import path as osp
import os
import sys
sys.path.append('src')
from prompt import multimodal_system_prompt, multimodal_user_prompt
from utils import Agent, name_to_path, DeepInfra_Agent


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-concept interaction generation")
    parser.add_argument('--shard_num', type=int, default=1, help='Total number of shards')
    parser.add_argument('--shard_index', type=int, default=0, help='Index of this shard (0-based)')
    parser.add_argument('--prompt_type', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    if args.model not in ['gpt-5-mini', 'gpt-5-nano', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'google/gemma-3-27b-it', 'google/gemma-3-12b-it', 'google/gemma-3-4b-it']:
        raise ValueError("Not Supported Model Type")
    
    if args.prompt_type not in ['circle', 'point', 'rectangle', 'scribble']:
        raise ValueError("Not Supported Prompt Type")
    
    concept_to_path = name_to_path()
    
    with jsonlines.open(f'data/benchmark/multimodal_prompt_qa_{args.prompt_type}.jsonl') as f:
        dataset = list()
        for item in f.iter():
            dataset.append(item)
    
    # Shard the list manually without using datasets
    total = len(dataset)
    shard_size = (total + args.shard_num - 1) // args.shard_num  # ceil division
    start = args.shard_index * shard_size
    end = min(start + shard_size, total)
    dataset = dataset[start:end]
    
    target_path = f'output/multimodal_prompt_qa_{args.prompt_type}/{args.model}/{args.shard_index}.jsonl'
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    f = open(target_path, 'w')
    
    print(f'Saved in {target_path}')
    
    image_root = f'data/vp_images/{args.prompt_type}'
    
    for data in tqdm(dataset):
        image_path = osp.join(image_root, f"{data['image_id']}.png")
        assert osp.exists(image_path)
        question = data['question']
        history = data['history']
        
        content = list()
        
        content.append({
            'type': 'text', 'text': '### Query image\n\n',
        })
        content.append({
            'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{encode_image(image_path)}"}
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
            'type': 'text', 'text': multimodal_user_prompt
        })
        content.append({
            'type': 'text', 'text': f'###Question\n{question}\n'
        })
        
        messages = [
            {
                'role': 'system',
                'content': multimodal_system_prompt
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
        data['response'] = response
        
        f.write(json.dumps(
            data
        ) + '\n')
        f.flush()
    f.close()
