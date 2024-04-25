seq_len=336 #NOTE THAT THE DEFAULT FOR TRANSFORMERS IS 96

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_${seq_len}_96 \
  --model Linear \
  --data custom \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_${seq_len}_192 \
  --model Linear \
  --data custom \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_${seq_len}_336 \
  --model Linear \
  --data custom \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_${seq_len}_720 \
  --model Linear \
  --data custom \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --des 'Exp' \
  --itr 1