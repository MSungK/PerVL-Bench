from openai import OpenAI
import base64
import re
import json
from typing import List, Dict, Any, Optional
from glob import glob
from collections import defaultdict
from os import path as osp
import google.generativeai as genai
import os


def name_to_path():
    db_path = 'data/database'
    files = glob(f'{db_path}/**/*.png', recursive=True) + glob(f'{db_path}/**/*.jpg', recursive=True)
    mapping = defaultdict(bool)
    for file in files:
        name = osp.dirname(file).split('/')[-1]
        if mapping[name] != False: continue
        mapping[name] = file
    return mapping


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class Agent:
    def __init__(self, model_info='gpt-5-mini'):
        if 'gpt' in model_info:
            self.client = OpenAI(
                api_key=os.environ['openai_api_key']
            )
        
        elif 'gemini' in model_info:
            self.client = OpenAI(
                api_key=os.environ['genai_api_key'],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            raise NotImplementedError


class DeepInfra_Agent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ['deepinfra_api_key'],
            base_url="https://api.deepinfra.com/v1/openai"
        )