## Transfer Learning for Predicting Virus-Host Protein Interactions for Novel Virus Sequences ##
Jack Lanchantin, Tom Weingarten, Arshdeek Sekhon, Clint Miller, Yanjun Qi <br/>

**SARS-CoV-2 PPI**
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --data_root ./data/data/ -tr yang_ppi/train.json -va yang_ppi/test.json -te  HVPPI/test.json -v vocab.data -s 1024 -hs 512 -l 12  -o results  --lr 0.00001 --dropout 0.1 --epochs 200 --attn_heads 8 --activation 'gelu' --task biogrid  --emb_type 'conv' --overwrite  --batch_size 4 --grad_ac_steps 4

**ZHOU PPI**
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --data_root ./data/data/ -tr zhou_ppi/h1n1/human/train.json  -va zhou_ppi/h1n1/human/test.json -v vocab.data -s 1024 -hs 512 -l 12  -o results --lr 0.00001 --dropout 0.1 --epochs 20000 --attn_heads 8 --activation 'gelu' --task ppi --emb_type 'conv' --overwrite  --batch_size 8 --grad_ac_steps 2 --name '' 

**BARMAN PPI**
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --data_root ./data/data/ -tr barman_ppi/train1.json  -va barman_ppi/test1.json -v vocab.data -s 1600 -hs 512 -l 12  -o results  --lr 0.00001 --dropout 0.1 --epochs 200 --attn_heads 8 --activation 'gelu' --task ppi  --emb_type 'conv' --overwrite  --batch_size 4 --grad_ac_steps 4

**DeNovo SLIM PPI**
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --data_root ./data/data/ -tr DeNovo/train.json  -va DeNovo/test.json -v vocab.data -s 1024 -hs 512 -l 12  -o results --lr 0.00001 --dropout 0.1 --epochs 20000 --attn_heads 8 --activation 'gelu' --task ppi --emb_type 'conv' --overwrite  --batch_size 8 --grad_ac_steps 2 --name '' --saved_bert ./results/multi.bert.bsz_16.layers_12.size_512.heads_8.drop_10.lr_1e-05.saved_bert.torch/best_model.pt
