#!/bin/bash
#SBATCH --job-name=math_j4.2
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


path="code_python_v1.jsonl"
q_set="code_python_v1"

model_name="qwen2-72b-instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 4 --dimension ${q_set} --split 1 &

model_name="yi-1.5-34b-chat"
CUDA_VISIBLE_DEVICES=4,5,6,7 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 4 --dimension ${q_set} --split 1 &

wait