#!/bin/bash
#SBATCH --job-name=s7.2
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


path="science_chemistry_v1_selected.jsonl"
output_dir="science_chemistry_v1_selected_responses"

model_name="llama2-13b-chat"
CUDA_VISIBLE_DEVICES=0,1 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 2 &

model_name="llama2-7b-chat"
CUDA_VISIBLE_DEVICES=2 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="command-r-(04-2024)"
CUDA_VISIBLE_DEVICES=3,4 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 2 &

model_name="qwen1.5-4b-chat"
CUDA_VISIBLE_DEVICES=5 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="meta-llama-3.1-8b-instruct"
CUDA_VISIBLE_DEVICES=6,7 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 2 &

wait