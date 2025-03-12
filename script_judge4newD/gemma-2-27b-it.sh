#!/bin/bash
#SBATCH --job-name=math_j2.2
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


path="code_python_v1.jsonl"
q_set="code_python_v1"

model_name="gemma-2-27b-it"
CUDA_VISIBLE_DEVICES=0,1 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} --split 1 &

model_name="koala-13b"
CUDA_VISIBLE_DEVICES=2,3 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} --split 1 &

model_name="mistral-7b-instruct-1"
CUDA_VISIBLE_DEVICES=4 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="mistral-7b-instruct-2"
CUDA_VISIBLE_DEVICES=5 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="openchat-3.5-0106"
CUDA_VISIBLE_DEVICES=6 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="starling-lm-7b-alpha"
CUDA_VISIBLE_DEVICES=7 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

wait