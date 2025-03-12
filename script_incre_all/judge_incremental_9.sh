#!/bin/bash
#SBATCH --job-name=m9.1
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


q_set='mt_bench'
path="mt_bench_questions.jsonl"

model_name="gemma-7b-it"
CUDA_VISIBLE_DEVICES=0 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="qwen1.5-72b-chat"
CUDA_VISIBLE_DEVICES=1,2,3,4 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 4 --dimension ${q_set} &

model_name="llama-3.2-3b-it"
CUDA_VISIBLE_DEVICES=5 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="ministral-8b-it"
CUDA_VISIBLE_DEVICES=6 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="smollm2-1.7b"
CUDA_VISIBLE_DEVICES=7 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

wait