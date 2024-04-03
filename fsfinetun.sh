EXP_DIR=exps/voc
BASE_TRAIN_DIR=${EXP_DIR}/base_train
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

fewshot_seed=01
num_shot=1
epoch=500
lr_drop1=300
lr_drop2=450
FS_FT_DIR=${EXP_DIR}/seed${fewshot_seed}_${num_shot}shot
mkdir ${FS_FT_DIR}

python -u main.py \
    --dataset_file voc_base1 \
    --backbone resnet101 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 2 \
    --category_codes_cls_loss \
    --resume ${BASE_TRAIN_DIR}/checkpoint.pth \
    --fewshot_finetune \
    --fewshot_seed ${fewshot_seed} \
    --num_shots ${num_shot} \
    --epoch ${epoch} \
    --lr_drop_milestones ${lr_drop1} ${lr_drop2} \
    --warmup_epochs 50 \
    --save_every_epoch ${epoch} \
    --eval_every_epoch ${epoch} \
    --output_dir ${FS_FT_DIR} \
2>&1 | tee ${FS_FT_DIR}/log.txt
