#!/usr/bin/env bash

dataset='gsm8k' # medqa | openbookqa | gsm8k

data_format='sharegpt'  # alpaca | sharegpt

project_path='/Path/to/Datasets'

#model_path=''

#model='unsloth/Meta-Llama-3.1-8B-Instruct'
#model='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'

#model='unsloth/mistral-7b-instruct-v0.3'
#model='unsloth/mistral-7b-instruct-v0.3-bnb-4bit'

#model='unsloth/gemma-2-9b-it'
#model='unsloth/gemma-2-9b-it-bnb-4bit'

#model='unsloth/Qwen2.5-7B-Instruct'
#model='unsloth/Qwen2.5-7B-Instruct-bnb-4bit'

model_path='/Path/to/Checkpoints'
model='checkpoint-1000'

model_str=$(echo "$model_path" | awk -F'/' '{print $(NF-2)}')

if [[ "$model" == *"unsloth"* ]]; then
    output_dir='outputs'
else
    output_dir='outputs_tworound_inference'
fi

if [[ $model_path =~ alpha_([0-9]+\.[0-9]+|[0-9]+) ]]; then
    alpha="${BASH_REMATCH[1]}"
else
    alpha=''
fi

if [ "$dataset" = "medqa" ]; then
    data_path="$project_path/MedQA-USMLE/questions/US"
    split='test'  # train | dev | test
    identity='random'
    if [[ "$model" == *"4bit"* ]]; then
        if [ -n "$alpha" ]; then
            output_path="./$output_dir/medqa/4bit/${identity// /_}/$data_format/$model_str/$model/alpha_$alpha"
        else
            output_path="./$output_dir/medqa/4bit/${identity// /_}/$data_format/$model_str/$model"
        fi
    else
        if [ -n "$alpha" ]; then
            output_path="./$output_dir/medqa/${identity// /_}/$data_format/$model_str/$model/alpha_$alpha"
        else
            output_path="./$output_dir/medqa/${identity// /_}/$data_format/$model_str/$model"
        fi
    fi
elif [ "$dataset" = "openbookqa" ]; then
    data_path="$project_path/openbookqa"
    split='test'  # train | validation | test
    identity='random'
    if [[ "$model" == *"4bit"* ]]; then
        if [ -n "$alpha" ]; then
            output_path="./$output_dir/openbookqa/4bit/${identity// /_}/$data_format/$model_str/$model/alpha_$alpha"
        else
            output_path="./$output_dir/openbookqa/4bit/${identity// /_}/$data_format/$model_str/$model"
        fi
    else
        if [ -n "$alpha" ]; then
            output_path="./$output_dir/openbookqa/${identity// /_}/$data_format/$model_str/$model/alpha_$alpha"
        else
            output_path="./$output_dir/openbookqa/${identity// /_}/$data_format/$model_str/$model"
        fi
    fi
elif [ "$dataset" = "gsm8k" ]; then
    data_path="$project_path/gsm8k"
    split="test"  # train | test
    identity='random'
    if [[ "$model" == *"4bit"* ]]; then
        if [ -n "$alpha" ]; then
            output_path="./$output_dir/gsm8k/4bit/${identity// /_}/$data_format/$model_str/$model/alpha_$alpha"
        else
            output_path="./$output_dir/gsm8k/4bit/${identity// /_}/$data_format/$model_str/$model"
        fi
    else
        if [ -n "$alpha" ]; then
            output_path="./$output_dir/gsm8k/${identity// /_}/$data_format/$model_str/$model/alpha_$alpha"
        else
            output_path="./$output_dir/gsm8k/${identity// /_}/$data_format/$model_str/$model"
        fi
    fi
else
    echo "Error: Unknown dataset '$dataset'"
    exit 1
fi

identity=''

if [ -n "$model_path" ]; then
  model=$model_path/$model
fi

mkdir -p ${output_path}

python infer.py \
--dataset $dataset --data_path $data_path --split $split \
--data_format $data_format \
--identity $identity \
--model $model \
--output_path $output_path > ${output_path}/console_log.txt 2>&1