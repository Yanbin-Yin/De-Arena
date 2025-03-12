import os
# Set the multiprocess method to 'spawn' to avoid CUDA initialization issues
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# os.environ["CUDA_VISIBLE_DEVICES"] = "7,4"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
import json
import random
import argparse
from tqdm import tqdm
import fire
import uuid
import torch
from vllm import SamplingParams
import gc
from vllm import LLM
from openai import OpenAI
import anthropic 
from decimal import Decimal
import re
import numpy as np
from utils import existing_model_paths
from tokencost import calculate_completion_cost, calculate_prompt_cost
import time
from decimal import Decimal
import sys
import copy
import itertools
total_completion_cost = 0
total_prompt_cost = 0

def save_to_jsonl(data, filename):
    """Saves a Python data structure to a .jsonl file."""
    with open(filename, 'w') as f:
        f.write(json.dumps(data) + '\n')

def load_records(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def update_voting_records(model, response_A_name, response_B_name, won, question_id, data_id, dimension, split=0):
    """Updates the voting records with a new voting result."""
    if split!=0:
        records_path = f"judgements_step5_incre_o1_{dimension}/{model}/voting_records_{split}.jsonl"
    else:
        records_path = f"judgements_step5_incre_o1_{dimension}/{model}/voting_records.jsonl"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(records_path), exist_ok=True)

    # Load existing records or create an empty list if the file does not exist
    if os.path.exists(records_path):
        try:
            records = load_records(records_path)[0]
        except:
            records = []
    else:
        records = []

    # Append a new record to the list of records
    new_record = {
        "response_A": response_A_name,
        "response_B": response_B_name,
        "Won": won,
        "question_id": question_id,
        "data_id": data_id
    }
    records.append(new_record)  # Ensure this is a flat append operation

    # Save updated records back to the JSONL file
    save_to_jsonl(records, records_path)

def format_prompt(model_name, prompt, tokenizer=None):
    if "vicuna" in model_name.lower():
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
    elif "llama2-13b-chat" in model_name.lower() or "llama2-7b-chat" in model_name.lower():
        return f"<s>[INST] {prompt} [/INST] {{ model_answer }}</s>"
    elif "openchat-3.5" in model_name.lower():
        return f"You are a helpful assistant. GPT4 Correct User: {prompt} GPT4 Correct Assistant:"
    elif "koala-13b" in model_name.lower():
        text = f"BEGINNING OF CONVERSATION: USER: {prompt} GPT:"
        return text
    elif "openassistant-pythia-12b" in model_name.lower():
        text = f"<|prompter|>{prompt}<|endoftext|><|assistant|>"
        return text
    return prompt

# Function to run the HuggingFace model with specific settings (temperature, max_tokens)
def run_hf_model(prompts, judge_name, tokenizer, engine, temperature=0.7, max_tokens=15):
    # Set the max tokens based on the judge name
    if judge_name == "athene-70b" or judge_name == "gemma-2-2b-it" or judge_name == "gemma-1.1-2b-it" or judge_name == "llama2-13b-chat" or judge_name == "gemma-1.1-7b-it":
        max_new_tokens = 16
    else:
        max_new_tokens = 15
    if judge_name == "koala-13b" or judge_name == "openassistant-pythia-12b":
        prompts = [format_prompt(judge_name,prompt) for prompt in prompts]
    # Generate responses based on the sampling parameters
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    outputs = engine.generate(prompts, sampling_params=sampling_params)
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    return responses

# Function to run OpenAI models
def run_openai_model(prompts, model_name, client, temperature=0.7, max_tokens=15):
    # Handle model selection for OpenAI models
    if "3.5-turbo-0125" in model_name: 
        model_name = "gpt-3.5-turbo-0125"
    elif "4-1106" in model_name: 
        model_name = "gpt-4-1106-preview"
    elif "gpt-4o-mini" in model_name:
        model_name = "gpt-4o-mini-2024-07-18"
    elif "ChatGPT-4o-latest" in model_name:
        model_name = "chatgpt-4o-latest"
    elif "gpt-4-turbo-2024-04-09" in model_name: 
        model_name = "gpt-4-turbo-2024-04-09"
    elif "gpt-4o-2024-05-13" in model_name:
        model_name = "gpt-4o-2024-05-13"
    elif "gpt-4o-2024-08-06" in model_name:
        model_name = "gpt-4o-2024-08-06"
    elif "o1-mini" in model_name:
        model_name = "o1-mini-2024-09-12"
        responses = []
        # Modify each prompt to ask the model to evaluate dataset quality
        for prompt in prompts:
            # Call OpenAI API with the modified quality evaluation prompt
            text = ""
            while not text.strip():  # 当文本为空时继续循环
                # 调用 OpenAI API
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                
                # 提取和存储响应
                text = completion.choices[0].message.content
            responses.append(str(text))
        
        return responses
    elif "o1-preview" in model_name:
        model_name = "o1-preview-2024-09-12"
        responses = []
        # Modify each prompt to ask the model to evaluate dataset quality
        for prompt in prompts:
            # Call OpenAI API with the modified quality evaluation prompt
            text = ""
            while not text.strip():  # 当文本为空时继续循环
                # 调用 OpenAI API
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                
                # 提取和存储响应
                text = completion.choices[0].message.content
            responses.append(str(text))
        
        return responses
    responses = []
    
    # Modify each prompt to ask the model to evaluate dataset quality
    for prompt in prompts:
        # Call OpenAI API with the modified quality evaluation prompt
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract and store the response
        text = completion.choices[0].message.content
        responses.append(str(text))
    
    return responses

def run_claude_model(prompts, client, model_name="claude-3-opus", max_tokens=2048):
    if model_name=="claude-3.5-sonnet":
        model_name="claude-3-5-sonnet-20240620"
    elif model_name=="claude-3-opus":
        model_name="claude-3-opus-20240229"
    elif model_name=="claude-3-sonnet":
        model_name="claude-3-sonnet-20240229"
    elif model_name=="claude-3-haiku":
        model_name="claude-3-haiku-20240307"
    elif model_name=="claude-1":
        model_name="claude-instant-1.2"
    elif model_name=="claude-2.0":
        model_name="claude-2.0"
    else:
        model_name="claude-2.1"
    responses = []
    
    for prompt in prompts:
        message = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = ''.join([block.text for block in message.content])
        responses.append(response_text)
    
    return responses

# Function to load the appropriate model (OpenAI, Anthropic, or HuggingFace)
def load_model(model_name,tensor_parallel_size, enforce_eager=True):
    model_info = existing_model_paths.get(model_name)
    from download_40_models import model_hf_name_dict
    tokenizer_str = model_hf_name_dict.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    # Return OpenAI or Anthropic models if applicable
    if model_info == "OPENAI":
        return None, "OPENAI"
    
    if model_info == "Claude":
        return None, "Anthropic"

    # Set attention backend for specific models
    # if "gemma-2" in model_name.lower():
    #     os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

    ## Load other API models


    # Check if the model path exists and load the vLLM model
    if os.path.exists(model_info):
        print(f"vLLM model detected, loading from: {model_info}")
        print(f"vLLM Tokenizer detected, loading from: {tokenizer_str}")
        vllm_model = LLM(model=model_info, tokenizer=tokenizer_str, gpu_memory_utilization=0.85, tensor_parallel_size=tensor_parallel_size, enforce_eager=True)
            
        if hasattr(vllm_model, "to"):
            vllm_model.to("cuda")
        else:
            print("The model does not support `to` method.")
        return None, vllm_model  

# Function to load JSONL files line by line
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            try:
                json_data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            yield json_data
    return json_data

# Function to get a question and its reference answer from a JSONL file
def get_question_with_reference(path, prompt_id):
    questions = load_jsonl(path)
    for question in questions:
        if question['question_id'] == prompt_id:
            if 'reference' in question and len(question['reference']) == 0:
                return question['turns'][0], ""
            else:
                return question['turns'][0], question.get('reference', [""])[0]
    return None, ""

# Function to fetch responses for a given model from a JSONL file
def fetch_responses(path,model):
    directory = f"{path}/{model}.jsonl"
    with open(directory, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Process the JSON to clean up and format correctly
    data = data.strip().replace('}\n{', '},{')
    data = f'[{data}]'  # Add square brackets to make it a valid JSON array
    try:
        data = json.loads(data)
    except:
        print(directory)
        exit()
    id2response = {lines['question_id']:lines['response'] for lines in data}
    return id2response

# Function to generate a pairwise judgment prompt
def judge_prompt_pairwise(question, answer_a, answer_b):
    prompt = (
        "[System]\n"
        'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. You should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.\n\n'
        "[User Question]\n"
        f"{question}\n\n"
        "[The Start of Assistant A's Answer]\n"
        f"{answer_a}\n"
        "[The End of Assistant A's Answer]\n\n"
        "[The Start of Assistant B's Answer]\n"
        f"{answer_b}\n"
        "[The End of Assistant B's Answer]\n\n"
        '[The Verdict(only contains one model identifier)]\n'
    )
    return prompt

# Function to generate a reference-based judgment prompt
def judge_prompt_pair_reference(question, answer_a, answer_b, ref_answer):
    prompt = (
        "[System]\n"
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. You should compare both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n"
        "[User Question]\n"
        f"{question}\n\n"
        "[The Start of Reference Answer]\n"
        f"{ref_answer}\n"
        "[The End of Reference Answer]\n\n"
        "[The Start of Assistant A's Answer]\n"
        f"{answer_a}\n"
        "[The End of Assistant A's Answer]\n\n"
        "[The Start of Assistant B's Answer]\n"
        f"{answer_b}\n"
        "[The End of Assistant B's Answer]\n\n"
        '[The Verdict(only contains one model identifier)]\n'
    )
    return prompt

# Function to determine the winner between two responses based on judge output
def determine_winner(judge_response, model_a, model_b):
    if "[[A]]" in judge_response and "[[B]]" not in judge_response:
        winner = model_a
    elif "[[B]]" in judge_response and "[[A]]" not in judge_response:
        winner = model_b
    else:
        if "[A]" in judge_response and "[B]" not in judge_response:
            winner = model_a
        elif "[B]" in judge_response and "[A]" not in judge_response:
            winner = model_b
        else:
            winner = "Tie"
    return winner

# Function to save judgment data into a JSONL file
def save_judgement(judge_name, data, dimension, split=0):
    """Save judgement data to a JSONL file."""
    if split != 0:
        path = f"judgements_step5_incre_o1_{dimension}/{judge_name}/judgements_{split}.jsonl"
    else:
        path = f"judgements_step5_incre_o1_{dimension}/{judge_name}/judgements.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')

# Function to save the cost estimation to a JSONL file
def save_cost_estimation(judge_name, cost):
    """Save judgement cost to a JSONL file."""
    path = f"judgement_cost/{judge_name}/cost.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert Decimal to float for any Decimal entries in the cost dictionary
    for key in cost:
        if isinstance(cost[key], Decimal):
            cost[key] = float(cost[key])
            
    with open(path, 'a') as f:
        f.write(json.dumps(cost) + '\n')

# Function to save time estimation data into a JSONL file
def save_time_estimation(judge_name, cost):
    """Save judgement time usage to a JSONL file."""
    path = f"judgement_time/{judge_name}/time.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert Decimal to float for any Decimal entries in the cost dictionary
    for key in cost:
        if isinstance(cost[key], Decimal):
            cost[key] = float(cost[key])
            
    with open(path, 'a') as f:
        f.write(json.dumps(cost) + '\n')

def resume_check(combination_models, initial_question_ids, model, dimension, split=0):
    """Updates the voting records with a new voting result."""
    if split != 0:
        records_path = f"judgements_step5_incre_o1_{dimension}/{model}/voting_records_{split}.jsonl"
    else:
        records_path = f"judgements_step5_incre_o1_{dimension}/{model}/voting_records.jsonl"

    if os.path.exists(records_path):
        try:
            records = load_records(records_path)[0]
        except:
            records = []
        pair2count = {}
        for record in records:
            pair = (record["response_A"], record["response_B"])
            pair2count[pair] = pair2count.get(pair, 0) + 1
        new_combination_models = []
        for res_A, res_B in combination_models:
            pair = (res_A, res_B)
            if pair2count.get(pair, 0) < len(initial_question_ids):
                new_combination_models.append(pair)
        return new_combination_models
    else:
        return combination_models

# Main function to run the judging trials, handling multiple models
def run_judging_trials(model_name="command-r-v01", tensor_parallel_size=1, split=0):
    print(model_name)
    model = model_name
    tokenizer, judge_model = load_model(model, tensor_parallel_size)
    client = None
    print(judge_model)
    if judge_model == "OPENAI":
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"))
    if judge_model == "Anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
    new_model_names = ["claude-3.5-sonnet-20241022"] #["ministral-8b-it", "smollm2-1.7b", "qwen2.5-1.5b", "llama-3.2-1b-it"]
    global total_completion_cost, total_prompt_cost
    model_names = list(existing_model_paths.keys())
    for dimension in ["mt_bench", "math_algebra_v1", "math_geometry_v1", "math_probability_v1", "reasoning_logical_v1", "reasoning_social_v1", "science_chemistry_v1", "science_physics_v1", "science_biology_v1"]:
        #for dimension in ["math_probability_v1", "reasoning_logical_v1", "reasoning_social_v1", "science_chemistry_v1"]:
        if dimension == "science_chemistry_v1":
            model_names.remove("gemini-1.5-pro-001")

        start_time = time.time()
        if dimension == "mt_bench":
            initial_question_ids = list(range(81, 161))
            path = "mt_bench_questions.jsonl"
        elif dimension == "math_algebra_v1":
            initial_question_ids = [296, 48, 292, 323, 129, 342, 133, 301, 37, 270, 385, 363, 356, 112, 285, 66, 26, 149, 16, 30, 55, 163, 97, 31, 339, 126, 263, 396, 103, 92, 161, 188, 111, 257, 282, 125, 393, 398, 122, 293, 236, 219, 341, 154, 261, 23, 381, 348, 99, 24, 195, 250, 255, 335, 326, 260, 325, 305, 86, 229, 197, 391, 153, 361, 265, 46, 215, 226, 190, 123, 117, 81, 231, 284, 281, 25, 207, 128, 331, 93, 316, 277, 57, 362, 354, 223, 47, 165, 307, 18, 157, 303, 74, 45, 104, 189, 210, 294, 248, 134]
            path = "math_algebra_questions_v1_selected.jsonl"
        elif dimension == "math_geometry_v1":
            initial_question_ids = [172, 140, 210, 69, 222, 136, 79, 151, 26, 373, 208, 96, 258, 29, 144, 324, 66, 244, 128, 237, 239, 360, 358, 88, 241, 104, 168, 235, 94, 183, 70, 300, 156, 76, 125, 127, 261, 243, 152, 307, 147, 47, 54, 110, 339, 80, 181, 395, 196, 95, 179, 116, 264, 12, 178, 223, 218, 377, 41, 101, 114, 219, 174, 327, 30, 213, 175, 375, 63, 98, 267, 357, 7, 124, 231, 379, 126, 272, 315, 370, 323, 296, 118, 68, 180, 135, 10, 304, 157, 121, 249, 160, 28, 229, 346, 215, 4, 238, 159, 103]
            path = "math_geometry_questions_v1_selected.jsonl"
        elif dimension == "math_probability_v1":
            initial_question_ids = [268, 169, 176, 118, 132, 377, 23, 167, 393, 211, 226, 301, 83, 27, 280, 182, 210, 64, 242, 338, 347, 238, 147, 29, 255, 225, 174, 8, 329, 260, 107, 249, 380, 276, 35, 270, 170, 128, 399, 376, 322, 294, 388, 195, 379, 398, 265, 279, 121, 360, 163, 361, 141, 153, 51, 41, 54, 34, 221, 4, 188, 162, 177, 101, 359, 241, 313, 246, 285, 325, 358, 144, 125, 137, 111, 365, 44, 138, 305, 357, 288, 286, 59, 394, 72, 348, 273, 319, 203, 293, 277, 333, 206, 40, 371, 228, 32, 334, 150, 158]
            path = "math_probability_questions_v1_selected.jsonl"
        elif dimension == "reasoning_logical_v1":
            initial_question_ids = [492, 200, 205, 60, 197, 236, 223, 220, 210, 93, 79, 209, 487, 224, 383, 229, 238, 398,
                                    44, 494, 227, 328, 47, 237, 283, 354, 216, 300, 374, 340, 270, 287, 222, 189, 486, 369,
                                    239, 489, 208, 285, 213, 304, 56, 217, 180, 120, 158, 476, 244, 100, 127, 360, 274, 225,
                                    235, 280, 196, 313, 408, 448, 187, 362, 207, 462, 295, 162, 305, 282, 119, 78, 23, 215,
                                    202, 416, 198, 370, 434, 261, 95, 199, 311, 191, 292, 479, 201, 410, 273, 323, 483, 243,
                                    99, 399, 28, 123, 440, 214, 490, 350, 182, 252]
            path = "reasoning_logical_v1_selected.jsonl"
        elif dimension == "reasoning_social_v1":
            initial_question_ids = [80, 457, 0, 246, 426, 394, 97, 312, 488, 405, 268, 122, 257, 434, 215, 141, 89, 85, 57,
                                    379, 459, 341, 8, 315, 418, 251, 388, 496, 182, 332, 431, 365, 22, 358, 339, 12, 136,
                                    433, 289, 129, 207, 112, 361, 491, 235, 310, 410, 91, 278, 474, 72, 438, 445, 222, 87,
                                    412, 302, 36, 54, 391, 367, 345, 415, 401, 212, 259, 192, 77, 389, 30, 178, 460, 336,
                                    346, 273, 38, 465, 5, 347, 162, 27, 458, 486, 451, 258, 356, 284, 297, 402, 173, 262,
                                    468, 393, 413, 333, 76, 42, 291, 183, 198]
            path = "reasoning_social_v1_selected.jsonl"
        elif dimension == "code_cpp_v1":
            initial_question_ids = [258, 388, 265, 257, 302, 339, 385, 317, 498, 482, 496, 278, 290, 255, 338, 382, 244,
                                    321, 329, 363, 273, 249, 334, 260, 284, 359, 292, 368, 394, 151, 436, 370, 313, 356,
                                    224, 296, 349, 369, 283, 310, 474, 452, 190, 390, 403, 286, 322, 437, 335, 411, 490,
                                    279, 81, 352, 242, 387, 285, 301, 364, 391, 367, 287, 331, 340, 366, 438, 397, 213, 319,
                                    303, 221, 358, 362, 271, 395, 470, 374, 330, 263, 298, 327, 445, 276, 299, 404, 288,
                                    428, 350, 376, 36, 281, 30, 269, 208, 418, 325, 422, 103, 243, 275]
        elif dimension == "code_java_v1":
            initial_question_ids = [230, 205, 198, 159, 251, 246, 258, 242, 304, 170, 150, 165, 253, 215, 265, 473, 217,
                                    387, 108, 388, 284, 351, 254, 47, 212, 419, 119, 493, 391, 264, 340, 177, 156, 239, 275,
                                    399, 209, 245, 372, 196, 240, 206, 241, 272, 164, 189, 324, 248, 216, 481, 121, 416,
                                    112, 317, 195, 408, 313, 79, 273, 402, 207, 203, 326, 120, 181, 122, 200, 161, 355, 249,
                                    132, 348, 255, 278, 221, 287, 134, 244, 202, 327, 400, 204, 199, 92, 222, 291, 237, 354,
                                    107, 218, 259, 227, 174, 343, 247, 263, 257, 173, 185, 224]
        elif dimension == "code_python_v1":
            initial_question_ids = [431, 449, 169, 170, 247, 278, 174, 298, 343, 313, 285, 155, 443, 400, 181, 226, 242,
                                    306, 344, 362, 318, 168, 259, 202, 264, 203, 310, 189, 212, 387, 481, 332, 167, 353,
                                    270, 325, 186, 171, 231, 335, 249, 237, 293, 136, 230, 263, 201, 452, 114, 277, 267,
                                    195, 172, 173, 191, 461, 250, 224, 281, 101, 493, 185, 334, 300, 219, 215, 194, 217,
                                    110, 183, 322, 192, 495, 280, 157, 364, 448, 339, 241, 311, 225, 324, 198, 128, 182, 74,
                                    235, 213, 323, 482, 208, 187, 389, 229, 107, 108, 193, 262, 314, 294]
        elif dimension == "science_biology_v1":
            initial_question_ids = [376, 286, 347, 440, 145, 464, 16, 95, 362, 487, 402, 165, 46, 71, 267, 171, 91, 169, 239, 409, 137, 350, 358, 316, 255, 300, 144, 127, 240, 206, 22, 401, 133, 102, 490, 471, 280, 434, 58, 277, 56, 472, 306, 104, 265, 379, 15, 408, 69, 353, 81, 290, 213, 325, 312, 289, 254, 178, 450, 368, 332, 210, 187, 130, 86, 77, 357, 53, 411, 194, 142, 27, 149, 297, 79, 50, 134, 264, 320, 459, 496, 340, 88, 361, 313, 63, 241, 26, 13, 25, 273, 191, 183, 159, 469, 360, 139, 47, 429, 470]
        elif dimension == "science_chemistry_v1":
            initial_question_ids = [453, 145, 96, 9, 377, 183, 40, 436, 118, 441, 85, 159, 390, 401, 228, 380, 76, 141, 425, 351, 189, 55, 13, 239, 150, 37, 498, 142, 479, 124, 391, 421, 359, 149, 308, 100, 35, 157, 64, 405, 438, 26, 423, 230, 223, 18, 243, 104, 300, 310, 28, 170, 251, 176, 32, 332, 174, 255, 7, 3, 215, 162, 420, 109, 186, 119, 43, 214, 130, 52, 370, 84, 381, 8, 173, 73, 133, 156, 106, 59, 434, 432, 246, 79, 42, 114, 382, 38, 158, 191, 488, 409, 1, 22, 477, 431, 71, 62, 143, 225]
            path = "science_chemistry_v1_selected.jsonl"
        else:
            raise NotImplementedError(f"Dimension {dimension} not yet implemented.")

        responses_dict = dict()
        # Fetch responses for each model
        for mode in model_names:
            if dimension == "mt_bench":
                responses_dict[mode] = fetch_responses(f"{dimension}_responses", mode)
            else:
                responses_dict[mode] = fetch_responses(f"{dimension}_selected_responses", mode)
            for ele in initial_question_ids:
                if ele not in responses_dict[mode]:
                    print(f"Skipping {mode} because {ele} not in!")
                    raise Exception

        # Select specific models for the judging trials
        model = copy.deepcopy(model_name)
        pair_models = copy.deepcopy(model_names)
        pair_models.remove(model_name)
        combination_models = []
        for new_model_name in new_model_names:
            for model_name_1 in model_names:
                if new_model_name != model_name_1:
                    combination_models.append((model_name_1, new_model_name))

        new_combination_models = resume_check(combination_models, initial_question_ids, model_name, dimension, split)
        print(f"After Resume Check, {len(combination_models)} pairs are reduced into {len(new_combination_models)} pairs.")
        combination_models = new_combination_models

        # Iterate over combinations of model pairs for comparison
        for model_a,model_b in tqdm(combination_models):
            responses_a = responses_dict[model_a]
            responses_b = responses_dict[model_b]
            print(model_a,model_b)

            batch_size = 80  # Set batch size for processing
            num_batches = (len(initial_question_ids) + batch_size - 1) // batch_size  # Calculate the number of batches

            for batch_idx in tqdm(range(num_batches)):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(initial_question_ids))
                prompts = list()
                swapped_prompts = list()
                question_ids = list()

                # Create prompts and swapped prompts for comparison
                for idx in range(start_idx, end_idx):
                    question_id = initial_question_ids[idx]
                    question, reference = get_question_with_reference(path, question_id)
                    response_a = responses_a[question_id]
                    response_b = responses_b[question_id]
                    if reference !="":
                        prompt = judge_prompt_pair_reference(question, response_a,response_b,reference)
                        swapped_prompt = judge_prompt_pair_reference(question, response_b,response_a,reference)
                    else:
                        prompt = judge_prompt_pairwise(question, response_a,response_b)
                        swapped_prompt = judge_prompt_pairwise(question, response_b,response_a)                       
                    prompts.append(prompt)
                    swapped_prompts.append(swapped_prompt)
                    question_ids.append(question_id)

                try:
                    # Adjust logic based on the type of judge_model
                    if judge_model == 'OPENAI':  # For OpenAI models
                        judge_responses = run_openai_model(prompts, model, client)
                        swapped_judge_responses = run_openai_model(swapped_prompts, model, client)
                    elif judge_model == "Anthropic":  # For Anthropic models (e.g., Claude)
                        # Placeholder for Anthropic, no operation for now
                        judge_responses = run_claude_model(prompts, client, model_name)
                        swapped_judge_responses = run_claude_model(swapped_prompts, client, model_name)
                    else:  # For other HuggingFace (HF) models
                        judge_responses = run_hf_model(prompts, model, tokenizer, judge_model)
                        swapped_judge_responses = run_hf_model(swapped_prompts, model, tokenizer, judge_model)
                except Exception as e:
                    print(f"Error evaluating model pair ({model_a}, {model_b}) with judge {model}: {e}")
                    continue  # Skip to the next model pair if there's an error

                cnt = 0
                # Process responses and determine winners
                for response, swapped_response in zip(judge_responses, swapped_judge_responses):
                    winner = determine_winner(response, model_a, model_b)
                    swapped_winner = determine_winner(swapped_response, model_b, model_a)
                    final_winner = winner if winner == swapped_winner else "TIE"
                    data_id = str(uuid.uuid4())
                    update_voting_records(model, model_a, model_b, final_winner, question_ids[cnt], data_id, dimension, split)

                    judgement_data = {
                                'data_id': data_id,
                                'question_id': question_ids[cnt],
                                'model_A': model_a,
                                'model_B': model_b,
                                'prompt': prompts[cnt],
                                'judge_response': response
                            }
                    save_judgement(model, judgement_data, dimension, split)
                    cnt+=1

    end_time = time.time()
    duration = end_time - start_time
    cost_data = {
                    'judge_name': model,
                    'completion_cost': total_completion_cost,
                    'prompt_cost': total_prompt_cost,
                    'total_cost': total_completion_cost + total_prompt_cost,
                    'duration': duration
                }
    save_time_estimation(model, cost_data)  # Save the time estimation
    save_cost_estimation(model, cost_data)  # Save the cost estimation
    # break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all models with specified parameters.')
    
    parser.add_argument('--path', type=str, default='math_questions.jsonl', help='Path to the input file')
    parser.add_argument('--model_name', type=str, default="vicuna-33b", help='Comma-separated list of model names')
    parser.add_argument('--dimension', type=str, default="math_algebra", help='new dimension names')
    parser.add_argument('--tensor_parallel_size', type=int, default=2, help='Tensor parallel size')
    parser.add_argument('--split', type=int, default=0, help='Tensor parallel size')
    
    args = parser.parse_args()

    fire.Fire(run_judging_trials(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size))
