#!/bin/bash

cd ../..

# custom config
DATA=data
TRAINER=FTVGG16

CFG=vgg16
SHOTS=16
WEP=100

DATASET=$1
EP=$2

for SEED in 1
do
for LR in 1e-5
do
for WD in 1e-2
do
    DIR=output/${DATASET}/shots${SHOTS}/${TRAINER}/${CFG}_EP${EP}_LR${LR}_WD${WD}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    CUDA_VISIBLE_DEVICES=0 python train_ft.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    OPTIM.MAX_EPOCH ${EP} \
    OPTIM.WARMUP_EPOCH ${WEP} \
    OPTIM.WEIGHT_DECAY ${WD} \
    OPTIM.LR ${LR} \
    TRAIN.CHECKPOINT_FREQ 100 \
    TEST.FINAL_MODEL best_val \
    DATASET.SUBSAMPLE_CLASSES all
    # fi
done
done
done