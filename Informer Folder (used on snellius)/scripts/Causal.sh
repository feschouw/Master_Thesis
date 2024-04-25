### M

#python -u main_informer.py --model informer --data Causal --features M --patience 100 --train_epochs 30 --seq_len 1 --label_len 1 --pred_len 200 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --learning_rate 0.01

 python -u main_informer.py --model informer --data Causal --data_path data_causal_1.csv --features M --seq_len 96 --label_len 96 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2

# python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 96 --label_len 96 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

# python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 168 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5

# python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5

# python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 720 --label_len 336 --pred_len 720 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5

# ### S

# python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3

# python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 96 --label_len 96 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

# python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 336 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

# python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 336 --label_len 168 --pred_len 336 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5

# python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 336 --label_len 336 --pred_len 720 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5
