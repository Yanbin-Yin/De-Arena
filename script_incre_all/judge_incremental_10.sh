#!/bin/bash
#SBATCH --job-name=m10.1
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


q_set='mt_bench'
path="mt_bench_questions.jsonl"

model_name="qwen2.5-1.5b"
CUDA_VISIBLE_DEVICES=0 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="llama-3.2-1b-it"
CUDA_VISIBLE_DEVICES=1 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="nemotron-70b"
CUDA_VISIBLE_DEVICES=2,3,4,5 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 4 --dimension ${q_set} &

wait