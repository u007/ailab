
train:
	mlx_lm.lora --model mlx-community/Qwen3-8B-8bit --train --data data/ --iters 100 --batch-size 1 --num-layers 4 --learning-rate 1e-4

run:
	mlx_lm.generate --model mlx-community/Qwen3-8B-8bit --adapter-path adapters --prompt "Who started Primalcom?" --max-tokens 50 --temp 0.1 --top-p 0.9

infer:
	mlx_lm.generate --model mlx-community/Qwen3-8B-8bit --adapter-path adapters --prompt "What is the capital of Japan?" --max-tokens 50 --temp 0.1 --top-p 0.9
	