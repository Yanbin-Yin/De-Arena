export CUDA_VISIBLE_DEVICES=3
python run_response.py --output_dir "mt_bench_responses" --model_name  "qwen2.5-3b,mistral-7b-instruct-2,gemma-2-9b-it-simpo,google-gemma-2-9b-it,llama-3.1-tulu-8b,zephyr-7b-beta,vicuna-7b"  --path "mt_bench_questions.jsonl" --openai_api "" --tensor_parallel_size 1