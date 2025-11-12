import json
import jsonlines
from tqdm import tqdm
import argparse
from os import path as osp
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
sys.path.append('src')
from utils import name_to_path
from prompt import multimodal_system_prompt, multimodal_user_prompt


def encode_image(image_path):
    return f'file:///{image_path}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    if args.prompt_type not in ['circle', 'point', 'rectangle', 'scribble']:
        raise ValueError("Not Supported Prompt Type")
    
    concept_to_path = name_to_path()
    
    with jsonlines.open(f'data/benchmark/multimodal_prompt_qa_{args.prompt_type}.jsonl') as f:
        dataset = list()
        for item in f.iter():
            dataset.append(item)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", cache_dir='./'
    )
    processor = AutoProcessor.from_pretrained(args.model, cache_dir='./')
    
    target_path = f'output/multimodal_prompt_qa_{args.prompt_type}/{args.model.split('/')[-1]}/{args.shard_index}.jsonl'
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
            'type': 'image', 'image': encode_image(image_path)
        })
        content.append({
            'type': 'text', 'text': '### Concepts\n\n',
        })
        for concept, textual_memory in history.items():
            content.append({
                'type': 'text', 'text': f'<{concept}>\n',
            })
            content.append({
                'type': 'image', 'image': encode_image(concept_to_path[concept])
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
        
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        data['response'] = response
        
        f.write(json.dumps(
            data
        ) + '\n')
        f.flush()

    f.close()
