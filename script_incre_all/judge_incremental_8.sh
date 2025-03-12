#!/bin/bash
#SBATCH --job-name=m8.1
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


q_set='mt_bench'
path="mt_bench_questions.jsonl"

model_name="llama2-13b-chat"
CUDA_VISIBLE_DEVICES=0,1 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} &

model_name="llama2-7b-chat"
CUDA_VISIBLE_DEVICES=2 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="command-r-(04-2024)"
CUDA_VISIBLE_DEVICES=3,4 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} &

model_name="command-r-(08-2024)"
CUDA_VISIBLE_DEVICES=5,6 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} &

model_name="qwen1.5-4b-chat"
CUDA_VISIBLE_DEVICES=7 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

wait