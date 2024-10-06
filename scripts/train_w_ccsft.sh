#!/usr/bin/env bash

dataset='medqa' # medqa | openbookqa | gsm8k
dataset_path='/Path/to/MedQA-USMLE'

data_format='sharegpt'  # alpaca | sharegpt
lr=1e-4 # 1e-4 for llama and mistral
batch_size=8
consistency_criterion='ot'

alpha=0.1

model='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'

# model='unsloth/mistral-7b-instruct-v0.3-bnb-4bit'

# model='unsloth/gemma-2-9b-it-bnb-4bit'

if [ "$dataset" = "medqa" ]; then
    data_path="${dataset_path}/questions/US"
    split='train'  # train | dev | test
    if [[ "$model" == *"4bit"* ]]; then
        output_path="./outputs/medqa/4bit/consistentLLMs/${data_format}/${model}/${consistency_criterion}_${lr}_${batch_size}"
    else
        output_path="./outputs/medqa/consistentLLMs/${data_format}/${model}/${consistency_criterion}"
    fi
elif [ "$dataset" = "gsm8k" ]; then
    data_path=${dataset_path}
    split="train"  # train | test
    if [[ "$model" == *"4bit"* ]]; then
        output_path="./outputs/gsm8k/4bit/consistentLLMs/${data_format}/${model}/${consistency_criterion}_${lr}_${batch_size}"
    else
        output_path="./outputs/gsm8k/consistentLLMs/${data_format}/${model}/${consistency_criterion}"
    fi
elif [ "$dataset" = "openbookqa" ]; then
    data_path="${dataset_path}"
    split="train"  # train | test
    if [[ "$model" == *"4bit"* ]]; then
        output_path="./outputs/openbookqa/4bit/consistentLLMs/${data_format}/${model}/${consistency_criterion}_${lr}_${batch_size}"
    else
        output_path="./outputs/openbookqa/consistentLLMs/${data_format}/${model}/${consistency_criterion}"
    fi
else
    echo "Error: Unknown dataset '$dataset'"
    exit 1
fi

mkdir -p ${output_path}

python csft.py \
--dataset $dataset --data_path $data_path --split $split \
--data_format $data_format \
--model $model \
--alpha $alpha \
--output_path $output_path \
--lr ${lr} \
--batch_size ${batch_size} \
--consistency_criterion ${consistency_criterion} > ${output_path}/console_log.txt 2>&1