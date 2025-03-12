#!/bin/bash
#SBATCH --job-name=m3.1
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


q_set='mt_bench'
path="mt_bench_questions.jsonl"

model_name="openchat-3.5"
CUDA_VISIBLE_DEVICES=0 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="openchat-3.5-0106"
CUDA_VISIBLE_DEVICES=1 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="vicuna-13b"
CUDA_VISIBLE_DEVICES=2,3 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} &

model_name="openassistant-pythia-12b"
CUDA_VISIBLE_DEVICES=4,5 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 2 --dimension ${q_set} &

model_name="starling-lm-7b-beta"
CUDA_VISIBLE_DEVICES=6 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

model_name="zephyr-7b-beta"
CUDA_VISIBLE_DEVICES=7 python judge_responses_for_newD_incre_all.py --path ${path} --model_name ${model_name} --tensor_parallel_size 1 --dimension ${q_set} &

wait