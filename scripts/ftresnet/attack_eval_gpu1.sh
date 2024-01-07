#!/bin/bash
cd ../..
DATA=data
TRAINER=FTRESNET50
SHOTS=16
CFG=resnet50

DATASET=$1

adv_attack='ZOsignSGD' # [SimBA, ZOsignSGD, SPSA, Bandit]
SAVE_DIR='output/svhn/BLACKVIP/vit-mae-base_vit_b16/adversarial_imgs_'
sample="100"

# SAVE_DIR=data/imagenet/images/Adv_Sample_${sample}/${TRAINER}/${adv_attack}
# TEST.ADVERSARIAL_ATTACK $adv_attack \
# --image-path data/imagenet/images/Sample_10/n01443537/ILSVRC2012_val_00030217.JPEG \
# --advimage-path output/adv_images/BLACKVIP/adv_image_label_1.png \
# TEST.START_CLASS 100 \
# TEST.SAVE_ADVIMG ${SAVE_DIR}  \
# TEST.SAMPLE $sample \
# TEST.EARLY_STOPPING False \
# MODEL.TORCHVISION ${MODEL} \
# --load-epoch 5000 \
# --reverse-test "TORCHVISION" \
# --adv-folder \
# --prompt-attack \
# --eval-only \

for SEED in 1
do
    CUDA_VISIBLE_DEVICES=1 python train_attack.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}_attack.yaml \
    --output-dir output/svhn/shots16/FTRESNET50/resnet50_EP10000_LR1e-5_WD1e-2/seed1 \
    --image-path data/svhn/test/sampled_10/2/11536.jpg \
    --eval-only \
    --image-folder data/svhn/sampled_test/sampled_10 \
    --target-folder data/svhn/sampled_test/adv_visualization/${TRAINER}/vis_${adv_attack} \
    TEST.ADVERSARIAL_ATTACK $adv_attack \
    TEST.SAVE_ADVIMG ${SAVE_DIR}_${adv_attack} \
    TEST.INFER_FOLDER True \
    TEST.NO_BLACKVIP True \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    #fi
done