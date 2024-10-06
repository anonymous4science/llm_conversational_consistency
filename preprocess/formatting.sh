#!/usr/bin/env bash

dataset='gsm8k' # medqa | openbookqa | gsm8k

data_format='sharegpt'  # alpaca | sharegpt

project_path='Path/to/Datasets'

if [ "$dataset" = "medqa" ]; then
    data_path="$project_path/MedQA-USMLE/questions/US"
    split='train'  # train | dev | test
elif [ "$dataset" = "openbookqa" ]; then
    data_path="$project_path/openbookqa"
    split='test'  # train | validation | test
elif [ "$dataset" = "gsm8k" ]; then
    data_path="$project_path/gsm8k"
    split="train"  # train | test
else
    echo "Error: Unknown dataset '$dataset'"
    exit 1
fi

python formatting.py \
--dataset $dataset --data_path $data_path --split $split \
--data_format $data_format \
--output_path $data_path