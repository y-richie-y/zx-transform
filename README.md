# Zx Transform
Graph simplification with transformers

## Example Usage
```sh
python train.py --dump_path '/home/richie/dumped' --save_periodic 0 --fp16 true --amp 2 --accumulate_gradients 1 --clip_grad_norm 5 --n_enc_heads 8 --n_dec_heads 8 --n_enc_hidden_layers 1 --n_dec_hidden_layers 1 --dropout 0 --norm_attention false --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --optimizer 'adam,lr=0.0001' --batch_size 64 --batch_size_eval 128 --epoch_size 300000 --max_epoch 10000 --num_workers 1 --export_data false --reload_data '' --reload_size '-1' --batch_load false --env_name pyzx --tasks simplify --env_base_seed '-1' --eval_size 10000 --windows false --eval_verbose 0 --beam_eval 1 --stopping_criterion 'valid_simplify_beam_acc,60' --validation_metrics valid_simplify_beam_acc --exp_name zx_exp --enc_emb_dim 512 --dec_emb_dim 512 --n_enc_layers 6 --n_dec_layers 6 --max_len 450 --max_output_len 300 --min_qubits 10 --max_qubits 10 --min_depth 15 --max_depth 15

```
