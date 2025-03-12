#!/bin/bash
#SBATCH --job-name=m1.1
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


q_set='mt_bench'
path="mt_bench_questions.jsonl"

model_name="athene-70b"
CUDA_VISIBLE_DEVICES=0,1,2,3 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 4 --dimension ${q_set} &

model_name="gemma-2-9b-it-simpo"
CUDA_VISIBLE_DEVICES=4 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="yi-1.5-34b-chat"
CUDA_VISIBLE_DEVICES=5,6 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} &

model_name="llama3-8b-instruct"
CUDA_VISIBLE_DEVICES=7 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

wait