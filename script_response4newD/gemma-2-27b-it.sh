#!/bin/bash
#SBATCH --job-name=arena_p2.1
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


model_name="gemma-2-27b-it"
path="science_physics_v1.jsonl"
output_dir="science_physics_v1"
CUDA_VISIBLE_DEVICES=0,1 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 2 &

model_name="koala-13b"
CUDA_VISIBLE_DEVICES=2,3 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 2 &

model_name="mistral-7b-instruct-1"
CUDA_VISIBLE_DEVICES=4 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="mistral-7b-instruct-2"
CUDA_VISIBLE_DEVICES=5 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="openchat-3.5-0106"
CUDA_VISIBLE_DEVICES=6 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="starling-lm-7b-alpha"
CUDA_VISIBLE_DEVICES=7 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

wait