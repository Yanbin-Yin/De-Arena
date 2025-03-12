#!/bin/bash
#SBATCH --job-name=m4.1
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


q_set='mt_bench'
path="mt_bench_questions.jsonl"

model_name="llama-3-70b-instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 4 --dimension ${q_set} &

model_name="meta-llama-3.1-70b-instruct"
CUDA_VISIBLE_DEVICES=4,5,6,7 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 4 --dimension ${q_set} &

wait