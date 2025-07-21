
train:
	mlx_lm.lora --model mlx-community/Qwen3-8B-8bit --train --data data --iters 100 --batch-size 1 --num-layers 4 --learning-rate 1e-4

run:
	.venv/bin/python infer.py "Who started primalcom?"

infer:
	.venv/bin/python infer.py
	