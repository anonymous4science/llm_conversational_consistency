import os
import json
import argparse

from datasets import load_dataset

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

    # output
    parser.add_argument('--output_path', default='./output', type=str, help='output path')

    args = parser.parse_args()

    print(args)

    return args


def format_sent(string, end_sym='.'):
    # Format a string into a sentence.
    string = string.strip()
    if len(string) > 0:
        string = string[0].upper() + string[1:]  # Capitalize the first letter
    if len(string) > 0 and string[-1] not in '.!?':
        string += end_sym  # Add a full stop if the string does not end with punctuation (., !, ?)

    return string


def load_medqa_data(args):
    filename = os.path.join(args.data_path, args.split + '.jsonl')

    items = []
    with open(filename, 'r') as file:
        for line in file:
            # Parse the line as a JSON object
            json_object = json.loads(line)
            items.append(json_object)

    return items


def medqa_alpaca_reformat(data, args):
    assert data['options'][data['answer_idx']] == data['answer']

    question = data['question']
    answer = data['answer']
    options = ', '.join([f'{key}: {value}' for key, value in data['options'].items()])
    answer_idx = data['answer_idx']

    instruction_content = format_sent(question, '?')
    input_content = format_sent(options, '.')
    output_content = format_sent(answer_idx + ': ' + answer, '.')

    message = {}
    message['instruction'] = instruction_content
    message['input'] = input_content
    message['output'] = output_content

    return message


def medqa_sharegpt_reformat(data, args):
    assert data['options'][data['answer_idx']] == data['answer']

    question = data['question']
    answer = data['answer']
    options = ', '.join([f'{key}: {value}' for key, value in data['options'].items()])
    answer_idx = data['answer_idx']

    human_content = format_sent(question, '?') + ' ' + format_sent(options, '.')
    gpt_content = format_sent(answer_idx + ': ' + answer, '.')

    message = []
    message.append({'from': 'human', 'value': human_content})
    message.append({'from': 'gpt', 'value': gpt_content})

    conversation = {'conversations': message}

    return conversation

##############################################################

def load_openbookqa_data(args):

    dataset = load_dataset('allenai/openbookqa', name='additional', split=args.split)

    return dataset


def openbookqa_alpaca_reformat(data, args):
    facts = data["fact1"]
    question_stems = data["question_stem"]
    choices = data["choices"]
    answerKeys = data["answerKey"]

    messages = []
    for fact, question_stem, choice, answerKey in zip(facts, question_stems, choices, answerKeys):
        fact = format_sent(fact, '.')
        question_stem = format_sent(question_stem, '?')

        instruction_content = fact + ' ' + question_stem
        input_content = ' '.join([l + ': ' + format_sent(t, '.') for l, t in zip(choice['label'], choice['text'])])
        output_content = answerKey + ': ' + format_sent(choice['text'][choice['label'].index(answerKey)], '.')

        message = {}
        message['instruction'] = instruction_content
        message['input'] = input_content
        message['output'] = output_content

        messages.append(message)

    return messages


def openbookqa_sharegpt_reformat(data, args):
    facts = data["fact1"]
    question_stems = data["question_stem"]
    choices = data["choices"]
    answerKeys = data["answerKey"]

    conversations = []
    for fact, question_stem, choice, answerKey in zip(facts, question_stems, choices, answerKeys):
        fact = format_sent(fact, '.')
        question_stem = format_sent(question_stem, '?')

        choice_str = ' '.join([l + ': ' + format_sent(t, '.') for l, t in zip(choice['label'], choice['text'])])

        human_content = fact + ' ' + question_stem + ' ' + choice_str
        gpt_content = answerKey + ': ' + format_sent(choice['text'][choice['label'].index(answerKey)], '.')

        message = []
        message.append({'from': 'human', 'value': human_content})
        message.append({'from': 'gpt', 'value': gpt_content})

        conversations.append({'conversations': message})

    return conversations

##############################################################

def load_gsm8k_data(args):

    dataset = load_dataset('openai/gsm8k', name='main', split=args.split)

    return dataset


def gsm8k_alpaca_reformat(data, args):
    questions = data["question"]
    answers = data["answer"]

    messages = []
    for question, answer in zip(questions, answers):
        instruction_content = format_sent(question, '?')
        input_content = ''
        output_content = answer

        message = {}
        message['instruction'] = instruction_content
        message['input'] = input_content
        message['output'] = output_content

        messages.append(message)

    return messages


def gsm8k_sharegpt_reformat(data, args):
    questions = data["question"]
    answers = data["answer"]

    conversations = []
    for question, answer in zip(questions, answers):
        human_content = format_sent(question, '?')
        gpt_content = answer

        message = []
        message.append({'from': 'human', 'value': human_content})
        message.append({'from': 'gpt', 'value': gpt_content})

        conversations.append({'conversations': message})

    return conversations



if __name__ == '__main__':

    args = get_argument()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.dataset == 'medqa':
        datum = load_medqa_data(args)

        reformatted_datum = []
        for data in datum:
            if args.data_format == 'alpaca':
                reformatted_data = medqa_alpaca_reformat(data, args)
            elif args.data_format == 'sharegpt':
                reformatted_data = medqa_sharegpt_reformat(data, args)
            else:
                raise NotImplementedError(f'Data format {args.data_format} not supported.')
            reformatted_datum.append(reformatted_data)
    elif args.dataset == 'openbookqa':
        datum = load_openbookqa_data(args)

        if args.data_format == 'alpaca':
            reformatted_datum = openbookqa_alpaca_reformat(datum, args)
        elif args.data_format == 'sharegpt':
            reformatted_datum = openbookqa_sharegpt_reformat(datum, args)
        else:
            raise NotImplementedError(f'Data format {args.data_format} not supported.')
    elif args.dataset == 'gsm8k':
        datum = load_gsm8k_data(args)

        if args.data_format == 'alpaca':
            reformatted_datum = gsm8k_alpaca_reformat(datum, args)
        elif args.data_format == 'sharegpt':
            reformatted_datum = gsm8k_sharegpt_reformat(datum, args)
        else:
            raise NotImplementedError(f'Data format {args.data_format} not supported.')
    else:
        raise ValueError(f'Dataset {args.dataset} not supported.')

    filename = os.path.join(args.output_path, args.split + '_' + args.data_format + '_sft.jsonl')
    with open(filename, 'w') as f:
        for item in reformatted_datum:
            f.write(json.dumps(item) + '\n')

    print('Done.')
