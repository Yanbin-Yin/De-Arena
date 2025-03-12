#!/bin/bash
#SBATCH --job-name=s1.2
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


path="science_chemistry_v1_selected.jsonl"
output_dir="science_chemistry_v1_selected_responses"

model_name="llama3-8b-instruct"
CUDA_VISIBLE_DEVICES=0 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="llama-3.2-3b-it"
CUDA_VISIBLE_DEVICES=1 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="openchat-3.5"
CUDA_VISIBLE_DEVICES=3 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="openassistant-pythia-12b"
CUDA_VISIBLE_DEVICES=4,5 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 2 &

model_name="starling-lm-7b-beta"
CUDA_VISIBLE_DEVICES=6 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="zephyr-7b-beta"
CUDA_VISIBLE_DEVICES=7 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

wait