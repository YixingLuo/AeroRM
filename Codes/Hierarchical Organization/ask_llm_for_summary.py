import json
import re

import argparse
import json
import os
from threading import Thread
import re
import random
import jieba
from rouge_chinese import Rouge
import torch
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
)

from template import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

def ask_llm_for_summary_first_inner(cluster_data, GPU):
    """
    Call the large model to generate a summary name and description for each cluster
    :param cluster_data: Sample data for each cluster, in the form of a list containing dictionaries
    :return: Returns the summary name and description for the cluster
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="auto", type=str)
    parser.add_argument('--base_model', default="Qwen2.5-7B-Instruct", type=str)
    parser.add_argument('--lora_model', default="", type=str)
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="qwen", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--rag', action='store_true')
    parser.add_argument('--gpus', default="7", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--model_max_length', type=int, default=8192)
    args = parser.parse_args()
    if args.only_cpu is True:
        args.gpus = ""
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    load_type = torch.float16
    cuda_name = 'cuda:' + GPU
    if torch.cuda.is_available():
        device = torch.device(cuda_name)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    if args.template_name.startswith("llama3"):  # pad token=EOS token
        tokenizer.add_special_tokens({'pad_token': '<end_of_text>'})
    
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        # device_map='auto',
        device_map={"": int(os.environ.get("LOCAL_RANK", GPU))},
        trust_remote_code=True,
    )
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()

    # Chat
    prompt_template = get_conv_template(args.template_name)

    print("Start inference.")

    max_new_tokens=1024
    do_sample=True
    num_beams=1
    repetition_penalty=1.0
    
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        pad_token_id = tokenizer.eos_token_id,
        temperature=0.7
    )

    MAX_LENGTH = 8000
    cluster_data_sorted = sorted(cluster_data, key=lambda data: len(data['Requirements']))
    formatted_data = []
    current_length = 0

    for index, data in enumerate(cluster_data_sorted):
        new_requirements = data['Requirements']
        new_names = data['name']
        new_keywords = data['Keywords']
        new_domains = data['Domains']

        formatted_item = f"Data {index+1} [name: {new_names}; keywords: {new_keywords}; domains: {new_domains}; requirements: {new_requirements}]"

        new_length = len(str(formatted_item))

        if current_length + new_length > MAX_LENGTH:
            break

        formatted_data.append(formatted_item)
        current_length += new_length
    
    if not formatted_data:
        first_data = cluster_data_sorted[0]
        first_requirements = first_data['Requirements']
        first_names =  first_data['name']
        first_keywords = first_data['Keywords']
        first_domains =  first_data['Domains']

        other_length = len(str(first_names)) + len(str(first_keywords)) + len(str(first_domains))

        remaining_length = MAX_LENGTH - other_length

        truncated_requirements = first_requirements[:remaining_length]

        formatted_data.append(f"Data {index+1} [name：{first_names}; keywords：{first_keywords}; domains：{first_domains}; requirements：{truncated_requirements}]")

    prompts = f"""
        ### Instruction ###
        You are an expert in the domain of aerospace software. Your task is to summarize the name and description of a cluster of reusable requirements artifacts based on the following metadata. Be specific and avoid overgeneralization. Flag clusters with domain conflicts or heterogeneous semantics.
        ### Input ####
        {";".join(formatted_data)}
        ### Output Format ###
            cluster_name: <cluster_name>
            description: <cluster_description>
            anomaly_falg: <anomaly_flag>
    """
    
    generated_texts = []

    model_input=[[prompts, '']]
    prompts = prompt_template.get_prompt(messages=model_input)
    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs_tokens['input_ids'].to(device)
    gen_sequence = model.generate(input_ids=input_ids, **generation_kwargs)
    prompt_len = len(input_ids[0])
    outputs = model.generate(input_ids=input_ids, **generation_kwargs)
    for gen_sequence in outputs:
        prompt_len = len(input_ids[0])
        gen_sequence = gen_sequence[prompt_len:]
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=False)
        stop_str = prompt_template.stop_str
        find_stop = False
        if type(stop_str) == type([1]):
            for stop_item in stop_str:
                pos = gen_text.find(stop_item)
                if pos != -1:
                    gen_text = gen_text[:pos]
                    find_stop = True
                    break
        else:
            pos = gen_text.find(stop_str)
            if pos != -1:
                gen_text = gen_text[:pos]
                find_stop = True
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

        sentences = re.split(r'(?<=[。！？])', generated_texts[-1])
        sentences = [s.strip() for s in sentences if s.strip()]
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        generated_texts[-1] = ''.join([s + ('。' if s[-1] not in '。！？' else '') for s in unique_sentences])

    print("generated_texts: ", generated_texts[-1])
    
    return generated_texts[-1]


def ask_llm_for_summary_recursion_inner(cluster_data, GPU):
    """
    Call the large model to generate a summary name and description for each cluster
    :param cluster_data: Sample data for each cluster, in the form of a list containing dictionaries
    :return: Returns the summary name and description for the cluster
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="auto", type=str)
    parser.add_argument('--base_model', default="Qwen2.5-7B-Instruct", type=str)
    parser.add_argument('--lora_model', default="", type=str)
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="qwen", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--rag', action='store_true')
    parser.add_argument('--gpus', default='7', type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--model_max_length', type=int, default=8192)
    args = parser.parse_args()
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    load_type = torch.float16
    cuda_name = 'cuda:' + GPU
    if torch.cuda.is_available():
        device = torch.device(cuda_name)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    if args.template_name.startswith("llama3"):
        tokenizer.add_special_tokens({'pad_token': '<end_of_text>'})
    
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        # device_map='auto',
        device_map={"": int(os.environ.get("LOCAL_RANK", GPU))},
        trust_remote_code=True,
    )
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()

    # Chat
    prompt_template = get_conv_template(args.template_name)
    
    print("Start inference.")

    max_new_tokens=1024
    do_sample=True
    num_beams=1
    repetition_penalty=1.0
    
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        pad_token_id = tokenizer.eos_token_id,
        temperature=0.7
    )

    MAX_LENGTH = 8000
    cluster_data_sorted = sorted(cluster_data, key=lambda data: len(data['cluster_description']))
    formatted_data = []
    current_length = 0

    for index, data in enumerate(cluster_data_sorted):
        new_name = data['cluster_name']
        new_description = data['cluster_description']

        formatted_item = f"Data {index+1} [name：{new_name}; description：{new_description}]"

        new_length = len(str(formatted_item))

        if current_length + new_length > MAX_LENGTH:
            break

        formatted_data.append(formatted_item)
        current_length += new_length
    
    if not formatted_data:
        first_data = cluster_data_sorted[0]
        first_name = first_data['cluster_name']
        first_description =  first_data['cluster_description']

        remaining_length = MAX_LENGTH - len(str(first_name))

        truncated_description = first_description[:remaining_length] 

        formatted_data.append(f"Data {index+1} [name：{first_name}; description：{truncated_description}]")

    prompts = f"""
        ### Instruction ###
        You are an expert in the domain of aerospace software. Your task is to summarize the name and description of a cluster of reusable requirements artifacts based on the following metadata. Be specific and avoid overgeneralization. Flag clusters with domain conflicts or heterogeneous semantics.
        ### Input ####
        {";".join(formatted_data)}
        ### Output Format ###
            cluster_name: <cluster_name>
            description: <cluster_description>
            anomaly_falg: <anomaly_flag>
    """

    generated_texts = []

    model_input=[[prompts, '']]
    prompts = prompt_template.get_prompt(messages=model_input)
    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs_tokens['input_ids'].to(device)
    gen_sequence = model.generate(input_ids=input_ids, **generation_kwargs)
    prompt_len = len(input_ids[0])
    outputs = model.generate(input_ids=input_ids, **generation_kwargs)
    for gen_sequence in outputs:
        prompt_len = len(input_ids[0])
        gen_sequence = gen_sequence[prompt_len:]
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=False)
        stop_str = prompt_template.stop_str
        find_stop = False
        if type(stop_str) == type([1]):
            for stop_item in stop_str:
                pos = gen_text.find(stop_item)
                if pos != -1:
                    gen_text = gen_text[:pos]
                    find_stop = True
                    break
        else:
            pos = gen_text.find(stop_str)
            if pos != -1:
                gen_text = gen_text[:pos]
                find_stop = True
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

        sentences = re.split(r'(?<=[。！？])', generated_texts[-1])
        sentences = [s.strip() for s in sentences if s.strip()]
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        generated_texts[-1] = ''.join([s + ('。' if s[-1] not in '。！？' else '') for s in unique_sentences])

    print("generated_texts: ", generated_texts[-1])
    return generated_texts[-1]

def ask_llm_for_summary_first_inner_glm4(cluster_data, GPU):
    """
    Call the large model to generate a summary name and description for each cluster
    :param cluster_data: Sample data for each cluster, in the form of a list containing dictionaries
    :return: Returns the summary name and description for the cluster
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="auto", type=str)
    parser.add_argument('--base_model', default="glm4-9b", type=str)
    parser.add_argument('--lora_model', default="", type=str)
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="qwen", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--rag', action='store_true')
    parser.add_argument('--gpus', default="7", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--model_max_length', type=int, default=8192)
    args = parser.parse_args()
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    load_type = torch.float16
    cuda_name = 'cuda:' + GPU
    if torch.cuda.is_available():
        device = torch.device(cuda_name)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    if args.template_name.startswith("llama3"):
        tokenizer.add_special_tokens({'pad_token': '<end_of_text>'})
    
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        # device_map='auto',
        device_map={"": int(os.environ.get("LOCAL_RANK", GPU))},
        trust_remote_code=True,
    )
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()

    # Chat
    prompt_template = get_conv_template(args.template_name)
    
    print("Start inference.")

    max_new_tokens=1024
    do_sample=True
    num_beams=1
    repetition_penalty=1.0
    
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        pad_token_id = tokenizer.eos_token_id,
        temperature=0.7
    )

    MAX_LENGTH = 8000
    cluster_data_sorted = sorted(cluster_data, key=lambda data: len(data['Requirements']))
    formatted_data = []
    current_length = 0

    for index, data in enumerate(cluster_data_sorted):
        new_requirements = data['Requirements']
        new_names = data['name']
        new_keywords = data['Keywords']
        new_domains = data['Domains']

        formatted_item = f"Data {index+1} [name：{new_names}; keywords：{new_keywords}; domains：{new_domains}; requirements：{new_requirements}]"

        new_length = len(str(formatted_item))

        if current_length + new_length > MAX_LENGTH:
            break

        formatted_data.append(formatted_item)
        current_length += new_length
    
    if not formatted_data:
        first_data = cluster_data_sorted[0]
        first_requirements = first_data['Requirements']
        first_names =  first_data['name']
        first_keywords = first_data['Keywords']
        first_domains =  first_data['Domains']

        other_length = len(str(first_names)) + len(str(first_keywords)) + len(str(first_domains))

        remaining_length = MAX_LENGTH - other_length

        truncated_requirements = first_requirements[:remaining_length]

        formatted_data.append(f"Data {index+1} [name：{first_names}; keywords：{first_keywords}; domains：{first_domains}; requirements：{truncated_requirements}]")

    prompts = f"""
        ### Instruction ###
        You are an expert in the domain of aerospace software. Your task is to summarize the name and description of a cluster of reusable requirements artifacts based on the following metadata. Be specific and avoid overgeneralization. Flag clusters with domain conflicts or heterogeneous semantics.
        ### Input ####
        {";".join(formatted_data)}
        ### Output Format ###
            cluster_name: <cluster_name>
            description: <cluster_description>
            anomaly_falg: <anomaly_flag>
    """
    
    generated_texts = []

    model_input=[[prompts, '']]
    prompts = prompt_template.get_prompt(messages=model_input)
    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs_tokens['input_ids'].to(device)
    gen_sequence = model.generate(input_ids=input_ids, **generation_kwargs)
    prompt_len = len(input_ids[0])
    outputs = model.generate(input_ids=input_ids, **generation_kwargs)
    for gen_sequence in outputs:
        prompt_len = len(input_ids[0])
        gen_sequence = gen_sequence[prompt_len:]
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=False)
        stop_str = prompt_template.stop_str
        find_stop = False
        if type(stop_str) == type([1]):
            for stop_item in stop_str:
                pos = gen_text.find(stop_item)
                if pos != -1:
                    gen_text = gen_text[:pos]
                    find_stop = True
                    break
        else:
            pos = gen_text.find(stop_str)
            if pos != -1:
                gen_text = gen_text[:pos]
                find_stop = True
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

        sentences = re.split(r'(?<=[。！？])', generated_texts[-1])
        sentences = [s.strip() for s in sentences if s.strip()]
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        generated_texts[-1] = ''.join([s + ('。' if s[-1] not in '。！？' else '') for s in unique_sentences])

    print("generated_texts: ", generated_texts[-1])
    
    return generated_texts[-1]


def ask_llm_for_summary_recursion_inner_glm4(cluster_data, GPU):
    """
    Call the large model to generate a summary name and description for each cluster
    :param cluster_data: Sample data for each cluster, in the form of a list containing dictionaries
    :return: Returns the summary name and description for the cluster
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="auto", type=str)
    parser.add_argument('--base_model', default="glm4-9b", type=str)
    parser.add_argument('--lora_model', default="", type=str)
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="qwen", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--rag', action='store_true')
    parser.add_argument('--gpus', default='7', type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--model_max_length', type=int, default=8192)
    args = parser.parse_args()
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    load_type = torch.float16
    cuda_name = 'cuda:' + GPU
    if torch.cuda.is_available():
        device = torch.device(cuda_name)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    if args.template_name.startswith("llama3"):
        tokenizer.add_special_tokens({'pad_token': '<end_of_text>'})
    
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        # device_map='auto',
        device_map={"": int(os.environ.get("LOCAL_RANK", GPU))},
        trust_remote_code=True,
    )
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()

    # Chat
    prompt_template = get_conv_template(args.template_name)
    
    print("Start inference.")

    max_new_tokens=1024
    do_sample=True
    num_beams=1
    repetition_penalty=1.0
    
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        pad_token_id = tokenizer.eos_token_id,
        temperature=0.7
    )

    MAX_LENGTH = 8000
    cluster_data_sorted = sorted(cluster_data, key=lambda data: len(data['cluster_description']))
    formatted_data = []
    current_length = 0

    for index, data in enumerate(cluster_data_sorted):
        new_name = data['cluster_name']
        new_description = data['cluster_description']

        formatted_item = f"Data {index+1} [name：{new_name}; description：{new_description}]"

        new_length = len(str(formatted_item))

        if current_length + new_length > MAX_LENGTH:
            break

        formatted_data.append(formatted_item)
        current_length += new_length
    
    if not formatted_data:
        first_data = cluster_data_sorted[0]
        first_name = first_data['cluster_name']
        first_description =  first_data['cluster_description']

        remaining_length = MAX_LENGTH - len(str(first_name))

        truncated_description = first_description[:remaining_length] 

        formatted_data.append(f"Data {index+1} [name：{first_name}; description：{truncated_description}]")

    prompts = f"""
        ### Instruction ###
        You are an expert in the domain of aerospace software. Your task is to summarize the name and description of a cluster of reusable requirements artifacts based on the following metadata. Be specific and avoid overgeneralization. Flag clusters with domain conflicts or heterogeneous semantics.
        ### Input ####
        {";".join(formatted_data)}
        ### Output Format ###
            cluster_name: <cluster_name>
            description: <cluster_description>
            anomaly_falg: <anomaly_flag>
    """

    generated_texts = []

    model_input=[[prompts, '']]
    prompts = prompt_template.get_prompt(messages=model_input)
    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs_tokens['input_ids'].to(device)
    gen_sequence = model.generate(input_ids=input_ids, **generation_kwargs)
    prompt_len = len(input_ids[0])
    outputs = model.generate(input_ids=input_ids, **generation_kwargs)
    for gen_sequence in outputs:
        prompt_len = len(input_ids[0])
        gen_sequence = gen_sequence[prompt_len:]
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=False)
        stop_str = prompt_template.stop_str
        find_stop = False
        if type(stop_str) == type([1]):
            for stop_item in stop_str:
                pos = gen_text.find(stop_item)
                if pos != -1:
                    gen_text = gen_text[:pos]
                    find_stop = True
                    break
        else:
            pos = gen_text.find(stop_str)
            if pos != -1:
                gen_text = gen_text[:pos]
                find_stop = True
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

        sentences = re.split(r'(?<=[。！？])', generated_texts[-1])
        sentences = [s.strip() for s in sentences if s.strip()]
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        generated_texts[-1] = ''.join([s + ('。' if s[-1] not in '。！？' else '') for s in unique_sentences])

    print("generated_texts: ", generated_texts[-1])
    return generated_texts[-1]