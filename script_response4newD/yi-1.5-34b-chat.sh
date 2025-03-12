#!/bin/bash
#SBATCH --job-name=arena_p5.2
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --exclude=mbz-h100-009


model_name="yi-1.5-34b-chat"
path="science_physics_v1.jsonl"
output_dir="science_physics_v1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 4 &

#path="science_chemistry_v1.jsonl"
#output_dir="science_chemistry_v1"
#CUDA_VISIBLE_DEVICES=4,5,6,7 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 4 &

wait