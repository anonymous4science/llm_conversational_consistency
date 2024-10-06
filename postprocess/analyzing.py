import os
import glob
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import pdb


def get_argument():
    parser = argparse.ArgumentParser(description='Description of your program')

    # dataset
    parser.add_argument('--dataset', default='medqa', required=False, type=str,
                        help='dataset name')
    parser.add_argument('--data_path', default='./MedQA-USMLE/questions/US', required=False, type=str,
                        help='dataset path')
    parser.add_argument('--split', default='train', required=False, type=str,
                        help='train | dev or validation | test')
    parser.add_argument('--split_idx', default=-1, type=int,
                        help='only valid for train set: 1 to num_splits')

    # output
    parser.add_argument('--output_path', default='./outputs', type=str, help='output path')

    args = parser.parse_args()

    print(args)

    return args


def load_medqa_data(filename):
    answer_ids, pred_ids, attack_pred_ids = [], [], []

    with open(filename, 'r') as file:
        for line in file:
            # Parse the line as a JSON object
            json_object = json.loads(line)
            answer_ids.append(json_object['answer_idx'])
            pred_ids.append(json_object['pred_idx'])
            attack_pred_ids.append(json_object['attack_pred_idx'])

    return answer_ids, pred_ids, attack_pred_ids


def load_openbookqa_data(filename):
    answer_ids, pred_ids, attack_pred_ids = [], [], []

    with open(filename, 'r') as file:
        for line in file:
            # Parse the line as a JSON object
            json_object = json.loads(line)
            answer_ids.append(json_object['answer_idx'])
            pred_ids.append(json_object['pred_idx'])
            attack_pred_ids.append(json_object['attack_pred_idx'])

    return answer_ids, pred_ids, attack_pred_ids


def load_gsm8k_data(filename):
    answer_vals, pred_vals, attack_pred_vals = [], [], []

    with open(filename, 'r') as file:
        for line in file:
            # Parse the line as a JSON object
            json_object = json.loads(line)

            answer_val = json_object['answer_val'].replace(',', '')
            pred_val = json_object['pred_val'].replace(',', '')
            attack_pred_val = json_object['attack_pred_val'].replace(',', '')

            if answer_val != '' and pred_val != '' and attack_pred_val != '':
                try:
                    answer_val_float = float(answer_val)
                    pred_val_float = float(pred_val)
                    attack_pred_val_float = float(attack_pred_val)
                except:
                    answer_vals.append(answer_val)
                    pred_vals.append(pred_val)
                    attack_pred_vals.append(attack_pred_val)

                    continue

                # Determine the number of decimal places
                answer_decimal_places = len(answer_val.split('.')[1]) if '.' in answer_val else 0
                pred_decimal_places = len(pred_val.split('.')[1]) if '.' in pred_val else 0
                attack_pred_decimal_places = len(attack_pred_val.split('.')[1]) if '.' in attack_pred_val else 0

                # Normalize both A and B to have the same number of decimal places
                max_decimal_places = max(answer_decimal_places, pred_decimal_places, attack_pred_decimal_places)

                # Format both floats with the max decimal places
                answer_val = f"{answer_val_float:.{max_decimal_places}f}"
                pred_val = f"{pred_val_float:.{max_decimal_places}f}"
                attack_pred_val = f"{attack_pred_val_float:.{max_decimal_places}f}"

            answer_vals.append(answer_val)
            pred_vals.append(pred_val)
            attack_pred_vals.append(attack_pred_val)

    return answer_vals, pred_vals, attack_pred_vals


def compute_metrics(ground_truth, predictions):
    # Ensure that both lists are of the same length
    if len(ground_truth) != len(predictions):
        raise ValueError("The length of ground_truth and predictions must be the same.")

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth, predictions)

    # Calculate precision, recall, and F1 score
    # Note: `average='macro'` computes the metric independently for each class and then takes the average (hence treating all classes equally)
    precision = precision_score(ground_truth, predictions, average='macro', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='macro', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average='macro', zero_division=0)

    # Compute and display the confusion matrix
    labels = list(np.sort(np.unique(predictions)))
    cm = confusion_matrix(ground_truth, predictions, labels=labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def compute_flipping_metrics(ground_truth, predictions, attack_predictions):
    # Ensure that both lists are of the same length
    if len(ground_truth) != len(predictions) and len(ground_truth) != len(attack_predictions):
        raise ValueError("The length of ground_truth and predictions must be the same.")

    sum_cnt, flipping_cnt, correct_flipping_cnt, incorrect_flipping_cnt = 0, 0, 0, 0
    for gt, pred, attack_pred in zip(ground_truth, predictions, attack_predictions):
        sum_cnt += 1
        if pred != attack_pred:
            flipping_cnt += 1
        if gt == attack_pred and pred != attack_pred:
            correct_flipping_cnt += 1
        if gt == pred and pred != attack_pred:
            incorrect_flipping_cnt += 1

    flipping_rate = float(flipping_cnt) / float(sum_cnt)
    correct_flipping_rate = float(correct_flipping_cnt) / float(sum_cnt)
    incorrect_flipping_rate = float(incorrect_flipping_cnt) / float(sum_cnt)

    return {
        'overall_flipping_rate': flipping_rate,
        'correct_flipping_rate': correct_flipping_rate,
        'incorrect_flipping_rate': incorrect_flipping_rate
    }


def save_dict_to_txt(metrics, filename, mode='w', note=None):
    with open(filename, mode) as file:
        file.write(f'{note}:\n\n')
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                # Convert numpy array to a string in a readable format
                array_str = np.array2string(value, separator=', ')
            else:
                array_str = str(value)
            # Write the key-value pair to the file
            file.write(f"{key}: {array_str}\n")

        file.write('\n\n')


if __name__ == '__main__':

    args = get_argument()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    jsonl_files = []
    for dirpath, dirnames, filenames in os.walk(args.output_path):
        for filename in filenames:
            if filename.endswith('.jsonl'):
                file_path = os.path.join(dirpath, filename)
                jsonl_files.append(file_path)

    for jsonl_file in jsonl_files:
        if 'medqa' in jsonl_file or 'medaq' in jsonl_file:
            answer_ids, pred_ids, attack_pred_ids = load_medqa_data(jsonl_file)
            metrics = compute_metrics(answer_ids, pred_ids)
            attack_metrics = compute_metrics(answer_ids, attack_pred_ids)
            flipping_metrics = compute_flipping_metrics(answer_ids, pred_ids, attack_pred_ids)
        elif 'openbookqa' in jsonl_file:
            answer_ids, pred_ids, attack_pred_ids = load_openbookqa_data(jsonl_file)
            metrics = compute_metrics(answer_ids, pred_ids)
            attack_metrics = compute_metrics(answer_ids, attack_pred_ids)
            flipping_metrics = compute_flipping_metrics(answer_ids, pred_ids, attack_pred_ids)
        elif 'gsm8k' in jsonl_file:
            answer_vals, pred_vals, attack_pred_vals = load_gsm8k_data(jsonl_file)
            metrics = compute_metrics(answer_vals, pred_vals)
            attack_metrics = compute_metrics(answer_vals, attack_pred_vals)
            flipping_metrics = compute_flipping_metrics(answer_vals, pred_vals, attack_pred_vals)
        else:
            raise ValueError(f'Dataset {args.dataset} not supported.')

        latex_str = f" {round(metrics['accuracy'], 3):.3f} & {round(attack_metrics['accuracy'], 3):.3f} & {round(attack_metrics['accuracy'], 3)-round(metrics['accuracy'], 3):.3f} & {round(metrics['f1'], 3):.3f} & {round(attack_metrics['f1'], 3):.3f} & {round(attack_metrics['f1'], 3)-round(metrics['f1'], 3):.3f} & {round(flipping_metrics['overall_flipping_rate'], 3):.3f} & {round(flipping_metrics['correct_flipping_rate'], 3):.3f} & {round(flipping_metrics['incorrect_flipping_rate'], 3):.3f} "
        
        save_dict_to_txt(metrics, jsonl_file.replace('.jsonl', '.txt'), 'w', 'Before questioning')
        save_dict_to_txt(attack_metrics, jsonl_file.replace('.jsonl', '.txt'), 'a', 'After questioning')
        save_dict_to_txt(flipping_metrics, jsonl_file.replace('.jsonl', '.txt'), 'a', 'Flipping')
        save_dict_to_txt({'latex': latex_str}, jsonl_file.replace('.jsonl', '.txt'), 'a', 'Latex table')

    print('Done.')
