#!/bin/bash
#SBATCH --job-name=s5.2
#SBATCH --partition=mbz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112



path="science_chemistry_v1_selected.jsonl"
output_dir="science_chemistry_v1_selected_responses"

model_name="vicuna-33b"
CUDA_VISIBLE_DEVICES=0,1 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 2 &

model_name="vicuna-7b"
CUDA_VISIBLE_DEVICES=2 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="gemma-1.1-2b-it"
CUDA_VISIBLE_DEVICES=3 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="gemma-1.1-7b-it"
CUDA_VISIBLE_DEVICES=4 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="gemma-2-2b-it"
CUDA_VISIBLE_DEVICES=5 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="gemma-2b-it"
CUDA_VISIBLE_DEVICES=6 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

model_name="gemma-7b-it"
CUDA_VISIBLE_DEVICES=7 python run_response_math.py --output_dir ${output_dir} --path ${path} --model_name ${model_name} --tensor_parallel_size 1 &

wait