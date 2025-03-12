#!/bin/bash
#SBATCH --job-name=arena_p4.1
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


model_name="qwen2-72b-instruct"
path="science_physics_v1.jsonl"
output_dir="science_physics_v1"
python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 8 &

wait