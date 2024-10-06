import re

import pdb


def parse_medqa_alpaca(text, answer_pattern):
    # Split the input into input and response parts
    parts = text.split('### Response:')

    # Extract response
    response_part = parts[1].strip()
    try:
        # Searching the text using the pattern
        match = re.search(r'%s:?\s+([A-Z]):?' % answer_pattern, response_part[response_part.rfind(answer_pattern):])
        if match:
            pred_idx = match.group(1)  # Return the first single letter found
        else:
            match = re.search(r'%s:?\s+([A-Z]):?' % answer_pattern.lower(), response_part[response_part.rfind(answer_pattern.lower()):])
            if match:
                pred_idx = match.group(1)  # Return the first single letter found
            else:
                match = re.search(r'^([A-Z]):', response_part)
                if match:
                    pred_idx = match.group(1)  # Return the first single letter found
                else:
                    pred_idx = ''
    except:
        pred_idx = ''

    ret = {'pred_idx' : f'{pred_idx.strip()}'}

    return ret


def parse_medqa_sharegpt(text, answer_pattern):
    # Split the input into user and assistant parts
    parts = text.split('<|start_header_id|>')

    # Extract assistant answer
    assistant_part = parts[2].strip()

    try:
        # answer = re.search(r'\n\n(.*?)<\|eot_id\|>', assistant_part, re.DOTALL).group(1)
        answer = re.search(r'(?<=\n\n)(.*)', assistant_part, re.DOTALL).group(1)

        # Searching the text using the pattern
        match = re.search(r'%s:?\s+([A-Z]):?' % answer_pattern, answer[answer.rfind(answer_pattern):])
        if match:
            pred_idx = match.group(1)  # Return the first single letter found
        else:
            match = re.search(r'%s:?\s+([A-Z]):?' % answer_pattern.lower(), answer[answer.rfind(answer_pattern.lower()):])
            if match:
                pred_idx = match.group(1)  # Return the first single letter found
            else:
                match = re.search(r'^([A-Z]):', answer)
                if match:
                    pred_idx = match.group(1)  # Return the first single letter found
                else:
                    pred_idx = ''
    except:
        pred_idx = ''

    ret = {'pred_idx' : f'{pred_idx.strip()}'}

    return ret

#################################################################

def parse_squad_v2_alpaca(text, answer_pattern):
    # Split the input into input and response parts
    parts = text.split('### Response:')

    # Extract response
    response_part = parts[1].strip()
    try:
        # Searching the text using the pattern
        match = re.search(r'%s(?:[^A-Za-z0-9]+)?\s*(.+?)(?:\.|$)' % answer_pattern, response_part[response_part.rfind(answer_pattern):])
        if match:
            pred_ans = match.group(1)  # Return the first single letter found
        else:
            match = re.search(r'%s(?:[^A-Za-z0-9]+)?\s*(.+?)(?:\.|$)' % answer_pattern.lower(), response_part[response_part.rfind(answer_pattern.lower()):])
            if match:
                pred_ans = match.group(1)  # Return the first single letter found
            else:
                pred_ans = ''
    except:
        pred_ans = ''

    ret = {'pred_ans' : f'{pred_ans.strip()}'}

    return ret

def parse_squad_v2_sharegpt(text, answer_pattern):
    # Split the input into user and assistant parts
    parts = text.split('<|start_header_id|>')

    # Extract assistant answer
    assistant_part = parts[2].strip()
    try:
        # answer = re.search(r'\n\n(.*?)<\|eot_id\|>', assistant_part, re.DOTALL).group(1)
        answer = re.search(r'(?<=\n\n)(.*)', assistant_part, re.DOTALL).group(1)

        # Searching the text using the pattern
        match = re.search(r'%s(?:[^A-Za-z0-9]+)?\s*(.+?)(?:\.|$)' % answer_pattern, answer[answer.rfind(answer_pattern):])
        if match:
            pred_ans = match.group(1)  # Return the first single letter found
        else:
            match = re.search(r'%s(?:[^A-Za-z0-9]+)?\s*(.+?)(?:\.|$)' % answer_pattern.lower(), answer[answer.rfind(answer_pattern.lower()):])
            if match:
                pred_ans = match.group(1)  # Return the first single letter found
            else:
                pred_ans = ''
    except:
        pred_ans = ''

    ret = {'pred_ans' : f'{pred_ans.strip()}'}

    return ret

################################################################

def parse_openbookqa_alpaca(text, answer_pattern):
    # Split the input into input and response parts
    parts = text.split('### Response:')

    # Extract response
    response_part = parts[1].strip()
    try:
        # Searching the text using the pattern
        match = re.search(r'%s:?\s+([A-Z]):?' % answer_pattern, response_part[response_part.rfind(answer_pattern):])
        if match:
            pred_idx = match.group(1)  # Return the first single letter found
        else:
            match = re.search(r'%s:?\s+([A-Z]):?' % answer_pattern.lower(), response_part[response_part.rfind(answer_pattern.lower()):])
            if match:
                pred_idx = match.group(1)  # Return the first single letter found
            else:
                match = re.search(r'^([A-Z]):', response_part)
                if match:
                    pred_idx = match.group(1)  # Return the first single letter found
                else:
                    pred_idx = ''
    except:
        pred_idx = ''

    ret = {'pred_idx' : f'{pred_idx.strip()}'}

    return ret


def parse_openbookqa_sharegpt(text, answer_pattern):
    # Split the input into user and assistant parts
    parts = text.split('<|start_header_id|>')

    # Extract assistant answer
    assistant_part = parts[2].strip()
    try:
        # answer = re.search(r'\n\n(.*?)<\|eot_id\|>', assistant_part, re.DOTALL).group(1)
        answer = re.search(r'(?<=\n\n)(.*)', assistant_part, re.DOTALL).group(1)

        # Searching the text using the pattern
        match = re.search(r'%s:?\s+([A-Z]):?' % answer_pattern, answer[answer.rfind(answer_pattern):])
        if match:
            pred_idx = match.group(1)  # Return the first single letter found
        else:
            match = re.search(r'%s:?\s+([A-Z]):?' % answer_pattern.lower(), answer[answer.rfind(answer_pattern.lower()):])
            if match:
                pred_idx = match.group(1)  # Return the first single letter found
            else:
                match = re.search(r'^([A-Z]):', answer)
                if match:
                    pred_idx = match.group(1)  # Return the first single letter found
                else:
                    pred_idx = ''
    except:
        pred_idx = ''

    ret = {'pred_idx' : f'{pred_idx.strip()}'}

    return ret

################################################################

def parse_gsm8k_alpaca(text, answer_pattern):
    # Split the input into input and response parts
    parts = text.split('### Response:')

    # Extract response
    response_part = parts[1].strip()
    try:
        # Searching the text using the pattern
        match = re.search(r'%s.*?([+-]?\d+(?:[,\.]\d+)?|\.\d+)' % answer_pattern, response_part[response_part.rfind(answer_pattern):])
        if match:
            pred_val = match.group(1)  # Return the first single letter found
        else:
            match = re.search(r'%s.*?([+-]?\d+(?:[,\.]\d+)?|\.\d+)' % answer_pattern, response_part)
            if match:
                pred_val = match.group(1)  # Return the first single letter found
            else:
                pred_val = ''
    except:
        pred_val = ''

    ret = {'pred_val' : f'{pred_val.strip()}'}

    return ret


def parse_gsm8k_sharegpt(text, answer_pattern):
    # Split the input into user and assistant parts
    parts = text.split('<|start_header_id|>')

    # Extract assistant answer
    assistant_part = parts[2].strip()
    try:
        # answer = re.search(r'\n\n(.*?)<\|eot_id\|>', assistant_part, re.DOTALL).group(1)
        answer = re.search(r'(?<=\n\n)(.*)', assistant_part, re.DOTALL).group(1)

        # Searching the text using the pattern
        match = re.search(r'%s.*?([+-]?\d+(?:[,\.]\d+)?|\.\d+)' % answer_pattern, answer[answer.rfind(answer_pattern):])
        if match:
            pred_val = match.group(1)  # Return the first single letter found
        else:
            match = re.search(r'%s.*?([+-]?\d+(?:[.\.]\d+)?|\.\d+)' % answer_pattern, answer)
            if match:
                pred_val = match.group(1)  # Return the first single letter found
            else:
                pred_val = ''
    except:
        pred_val = ''

    ret = {'pred_val' : f'{pred_val.strip()}'}

    return ret


if __name__ == '__main__':

    text = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nA junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take? A: Disclose the error to the patient but leave it out of the operative report, B: Disclose the error to the patient and put it in the operative report, C: Tell the attending that he cannot fail to disclose this mistake, D: Report the physician to the ethics committee, E: Refuse to dictate the operative report.\n\nSovle this problem step-by-step and choose the correct option. Use "The correct answer is" to indicate your choice.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nC: Tell the attending that he cannot fail to disclose this mistake.<|eot_id|>'
    pattern = 'The correct answer is'

    ret = parse_medqa_sharegpt(text, pattern)

    print(ret)