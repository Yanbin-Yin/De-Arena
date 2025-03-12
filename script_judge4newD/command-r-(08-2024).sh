#!/bin/bash
#SBATCH --job-name=math_j5.2
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


path="code_python_v1.jsonl"
q_set="code_python_v1"

model_name="athene-70b"
CUDA_VISIBLE_DEVICES=0,1,2,3 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 4 --dimension ${q_set} --split 2 &

model_name="command-r-(08-2024)"
CUDA_VISIBLE_DEVICES=4,5 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} --split 2 &

model_name="gemma-2-27b-it"
CUDA_VISIBLE_DEVICES=6,7 python judge_responses_for_QS.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} --split 2 &

wait