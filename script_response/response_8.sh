#!/bin/bash
#SBATCH --job-name=s8.2
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


path="science_chemistry_v1_selected.jsonl"
output_dir="science_chemistry_v1_selected_responses"

model_name="qwen1.5-32b-chat"
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 4 &

wait