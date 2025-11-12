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
from prompt import text_system_prompt, text_user_prompt


def encode_image(image_path):
    return f'file://{image_path}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--topk', type=int, required=True)
    args = parser.parse_args()
    
    concept_to_path = name_to_path()
    
    with jsonlines.open('data/benchmark/text_prompt_qa.jsonl') as f:
        dataset = list()
        for item in f.iter():
            dataset.append(item)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", cache_dir='./'
    )
    processor = AutoProcessor.from_pretrained(args.model, cache_dir='./')
    
    target_path = f"output/text_prompt_qa/{args.model.split('/')[-1]}.jsonl"
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    f = open(target_path, 'w')
    
    print(f'Saved in {target_path}')
    
    for data in tqdm(dataset):
        history = data['history']
        QAs = data['QA']
        gt = data['gt']
        img_path = osp.join('data', data['image_path'])
        # print(img_path)
        # print(os.getcwd())
        # exit()
        content = list()
        
        content.append({
            'type': 'text', 'text': '### Query image\n\n',
        })
        content.append({
            'type': 'image', 'image': encode_image(img_path)
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
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            content.pop()
            f.write(json.dumps({
                'img_path': img_path,
                'question': question,
                'answer': answer,
                'response': response,
            }) + '\n')
            f.flush()
    f.close()
