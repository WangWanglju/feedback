# ====================================================
# CFG
# ====================================================
class CFG:
    wandb=False
    competition='FB3'
    EXP_NAME='pseudo'
    debug=False
    apex=True
    print_freq=20
    num_workers=16
    #model
    model="microsoft/deberta-v3-base"
    pooling='pooling'  # weighted_pooling, pooling
    multi_sample_dropout=True
    gradient_checkpointing=True
    mixout = 0.
    #training
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=5
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=12
    max_len=512
    weight_decay=0.01
    llrd=0.9
    fgm = False
    #swa
    swa = False
    swa_start = 4
    swa_learning_rate = 1e-6
    anneal_epochs=2 
    anneal_strategy='cos'
    #awp
    awp = False
    awp_eps = 1e-2
    awp_lr = 1e-5
    nth_awp_start_epoch = 1

    use_prior_wd = False
    gradient_accumulation_steps=1
    max_grad_norm=1000
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=4
    trn_fold=[0,1,2,3]
    train=True