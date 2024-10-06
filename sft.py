# https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=vITh0KVJ10qX

import os
import math
import argparse

import sys
# sys.path.insert(0, '/scratch/hh1811/projects/unsloth-main')

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth import is_bfloat16_supported

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

import torch

from csft_trainer import MySFTTrainer, CSFTTrainer

import pdb


def get_argument():
    parser = argparse.ArgumentParser(description='Description of your program')

    # dataset
    parser.add_argument('--dataset', default='medqa', required=True, type=str,
                        help='dataset name')
    parser.add_argument('--data_path', default='./MedQA-USMLE/questions/US', required=True, type=str,
                        help='dataset path')
    parser.add_argument('--split', default='train', required=True, type=str,
                        help='train | dev or validation | test')

    parser.add_argument('--data_format', default='alpaca', required=True, type=str,
                        help='alpaca | sharegpt')

    # model
    parser.add_argument('--seed', default=3407, type=int,
                        help='random seed for the model')
    parser.add_argument('--model', default='unsloth/Meta-Llama-3.1-8B', type=str,
                        help='model to finetune')
    parser.add_argument('--max_seq_length', default=2048, type=int,
                        help='Choose any! We auto support RoPE Scaling internally!')
    parser.add_argument('--dtype', default=None, type=str,
                        help='None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+')
    parser.add_argument('--load_in_4bit', default=False, type=bool,
                        help='Use 4bit quantization to reduce memory usage. Can be False.')

    # output
    parser.add_argument('--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=10, type=int, help='number of epochs')

    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':

    args = get_argument()

    # if 'codellama' in args.model:
    #     assert args.dataset == 'gsm8k', 'CodeLlama only supports dataset gsm8k.'
    # if 'gsm8k' in args.dataset:
    #     assert 'codellama' in args.model, 'Dataset gsm8k only supports CodeLlama model.'

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if '4bit' in args.model:
        args.load_in_4bit = True
    else:
        args.load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=args.seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    # pdb.set_trace()
    # dataset['instruction'][0]: str
    # dataset['input'][0]: str
    # dataset['output'][0]: str
    # dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    if args.dataset in ['medqa', 'squad_v2', 'gsm8k', 'openbookqa']:
        data_file = os.path.join(args.data_path, args.split + '_' + args.data_format + '_sft.jsonl')
        dataset = load_dataset('json', data_files={args.split: data_file})
    else:
        raise ValueError(f'Dataset {args.dataset} not supported.')

    print(f"=============> PID: {os. getpid()}")

    dataset = dataset[args.split]

    if args.data_format == 'alpaca':
        alpaca_prompt = """
           ### Instruction:
           {}

           ### Input:
           {}

           ### Response:
           {}"""

        EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


        def formatting_alpaca_prompts_func(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs):
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
                texts.append(text)

            return {"text": texts}


        # dataset['text'][0]: str
        dataset = dataset.map(formatting_alpaca_prompts_func, batched=True)
    elif args.data_format == 'sharegpt':
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
        )


        def formatting_sharegpt_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in
                     convos]

            return {"text": texts}


        # dataset['text'][0]: str
        dataset = dataset.map(formatting_sharegpt_prompts_func, batched=True)
    else:
        raise NotImplementedError(f'Data format {args.data_format} not supported.')

    total_steps = args.num_epoch * math.ceil(len(dataset) / args.batch_size)
    # trainer = SFTTrainer(
    trainer = MySFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=total_steps,
            save_steps=1000,
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=os.path.join(args.output_path, 'outputs'),
        ),
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained(os.path.join(args.output_path, "lora_model"))
    tokenizer.save_pretrained(os.path.join(args.output_path, "lora_model"))