#!/usr/bin/env bash

GPU=$1

BATCH_SIZE=$2

DATASET=$3

DATA_PATH=$4

MODEL=$5

DISTRIBUTION=$6

ROUND=$7

EPOCH=$8

LR=$9

OPT=${10}

CI=${11}

LOCALN=${12}

D2D_USER_NUM=${13}

Topology_type=${14}

Model_update_times=${15}

label_divided=${16}

train_data=${17}

python ./mainn.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--client_optimizer $OPT \
--lr $LR \
--ci $CI \
--localN $LOCALN \
--d2d_user_num $D2D_USER_NUM \
--topology_type $Topology_type \
--model_update_times $Model_update_times \
--label_divided_num $label_divided \
--train_data_dir $train_data