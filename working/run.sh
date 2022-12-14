#when first run
# pip install tqdm transformers sentencepiece
# pip install iterative-stratification==0.1.7


# find all configs in configs/

# set your gpu id
gpus=0
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi mmn task on the same machine
master_addr=127.0.0.3
master_port=29511

# ------------------------ need not change -----------------------------------
# config_file=configs/$config\.yaml
# output_dir=outputs/$config

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port main.py