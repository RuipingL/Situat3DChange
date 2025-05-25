import argparse
import hashlib
import json
import pickle
import random
import re
import sys
import time
from openai import OpenAI
import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
from ours_eval_depth import parse_distance_to_meters
import os

def clean_answer(data):
    key = 'answer'
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)
    
    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    return data

#     return ret_data

def chat_llm(chatgpt_messages, temperature=0, max_tokens=100, model='gpt-4o-mini-2024-07-18'):
    # model = 'gpt-4o-mini'
    # model = 'gpt-4o'
    client = OpenAI(api_key="your-openai-key")
    response = client.chat.completions.create(
        model=model,
        messages=chatgpt_messages,
        max_tokens=max_tokens,
        temperature=temperature
    )


    return response.choices[0].message.content

def gpt_llm_eval(type, q, pred, a):
    if "direction" in type.lower():
        system_prompt = (
                    "Score the response about direction from 1 (worst) to 5 (best)."
                    "Score 2-4: Reflect partial correctness or minor errors."
                    "Mapping of proximity direction and clock face: front (from 11 to 1 o'clock), left (from 8 to 10 o'clock), right (from 2 to 4 o'clock), back (from 5 to 7 o'clock)."
                    "Criteria:"
                    "Score 1: If the response is in the opposite proximity direction to the ground truth, e.g., GT: '9 o’clock'(left), Response: '4 o’clock'(right)."
                    "Score 2: If the response has a significant directional error but is not completely opposite, e.g., GT: '3 o’clock'(right), Response: 'Back'."
                    "Score 3: If the response is adjacent to the correct direction, e.g., GT: '11 o’clock'(front), Response: 'Left'."
                    "Score 4: If the response is in the correct proximity direction, e.g., GT: '6 o'clock'(back), Response: 'Back'."
                    "Score 5: If the difference is less than or equal to 1 o'clock on the clock face, e.g., GT: '11 o'clock', Response: '10 o'clock'."
                    "Output only the score.")
    else:
        system_prompt = ("Score open-ended answers from 1 to 5 based on accuracy to the ground truth."
                         "Score 2-4: Reflect partial correctness or minor errors."
                        "Criteria:"
                        "Affordance: Question: Is there any furniture to rest feet on nearby? Ground Truth: Yes Example Response: Yes, there is a ottoman nearby. Score: 5 (Correct match) "
                        "Attribute: Question: What is the color of the ottoman? Ground Truth: Blue, red, brown. Example Response: The ottoman is brown. Score: 3 (Partial match)."
                        "Existence: Question: Is there a chair on my left? Ground Truth: Yes. Example Response: Yes, there is a chair on the left. Score: 5 (Correct match) "
                        "Counting: Question: How many tables are in the room? Ground Truth: Three Example. Response: Two. Score: 1 (Significant discrepancy) "
                        "Warning: Question: Are there any changed objects on my familiar route to the door? Ground Truth: Yes, a chair. Example Response: Yes, there is a table on the way to the door. Score: 2 (Major incorrect) " 
                        "Allocentric Relationship: Question: Where is the kettle? Ground Truth: On the kitchen cabinet. Example Response: The kettle is on the kitchen counter. Score: 4 (Approximate match) "
                        "Output only the score.")
    eval_prompt = f"{type}: Question: {q} Ground Truth: {a} Response: {pred} Score: "
    prompt = [ {"role": "system", "content": system_prompt}, {"role": "user", "content": eval_prompt}]
    output = chat_llm(prompt)
    return output

def resposne_to_scores(pred_file, log_path):
    pred_js = json.load(open(pred_file, 'r'))
    for response in pred_js:
        # for i in range(len(responses)):
        question = response["instruction"].replace('USER: ', "").replace(" ASSISTANT:", "")
        if 'distance' in response["type"].lower():
            pred_distance = parse_distance_to_meters(response["response_pred"])
            gt_distance = parse_distance_to_meters(response["response_gt"])
            if pred_distance is None or gt_distance is None:
                score = None
            elif pred_distance == 0 and gt_distance == 0:
                score = 1
            elif pred_distance == 0 or gt_distance == 0:
                score = 0
            else:
                score = 1- min((abs(pred_distance - gt_distance)/gt_distance), 1)
            response["score"] = score
        else:
            score = gpt_llm_eval(response["type"], question, response["response_pred"], response["response_gt"])
            response["score"] = score
        with open(log_path, 'a') as log_file:
            log_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(response["index"]) + ' ' + question + ' ' + response["response_pred"] + ' ' + response["response_gt"] + ' ' + str(score) + '\n')
    return pred_js

import re

def extract_single_number(s):
    """
    Extracts the only number from a given string.
    If multiple or no numbers are found, it raises a ValueError.

    :param s: Input string
    :return: Extracted number (int or float)
    :raises ValueError: If no number or multiple numbers are found
    """
    numbers = re.findall(r'\d+\.\d+|\d+', s)  # Match floating-point and integer numbers
    
    if len(numbers) == 0:
        raise ValueError("No number found in the string.")
    elif len(numbers) > 1:
        raise ValueError("Multiple numbers found in the string.")
    
    return float(numbers[0]) if '.' in numbers[0] else int(numbers[0])

def cls_score(qa_types, pred_file):
    responses = json.load(open(pred_file, 'r'))
    scores = {qa_type: 0 for qa_type in qa_types}
    num = {qa_type: 0 for qa_type in qa_types}
    # for scene in tqdm.tqdm(pred_file.keys()):
    #     responses = pred_file[scene]["response"]
    for i in range(len(responses)):
        for qa_type in qa_types:
            if qa_type in responses[i]["type"]:
                if 'distance' in responses[i]["type"].lower():
                    if type(responses[i]["score"]) == str:
                        score = responses[i]["score"].replace('\n', '')
                        if 'None' in score:
                            continue
                    else:
                        score = responses[i]["score"]
                    # score = float(score)
                    
                    if score is not None:
                        scores[qa_type] += score
                        num[qa_type] += 1
                        print(responses[i]["index"], score, qa_type)
                    break    
                else:
                    print(responses[i].keys())
                    score = extract_single_number(responses[i]["score"])
                    scores[qa_type] += (score-1)/4
                    num[qa_type] += 1
                    break
    for qa_type in qa_types:
        print(f'{qa_type}: {scores[qa_type]} {num[qa_type]}')
        scores[qa_type] = scores[qa_type]/num[qa_type]
    overall = sum([v for v in scores.values()])/len(scores)
    return scores, overall

def parse_line(line):
    index = line.split(' ')[2]
    score = line.split(' ')[-1]
    return index, score

def log_to_js(log_file, json_file):
    data_js = json.load(open(json_file, 'r'))
    data = {}
    with open(log_file, 'r') as file:
        for line in file:
            index, score = parse_line(line)
            data[index] = score
    for i in range(len(data_js)):
        data_js[i]["score"] = data[data_js[i]["index"]]
    json.dump(data_js, open(json_file, 'w'), indent=4)
            

if __name__ == '__main__':
    qa_types = ['Affordance', 'Attribute', 'Existence', 'Counting', 'Warning', 'Allocentric Relationship', 'Allocentric Distance', 'Egocentric Direction', 'Egocentric Distance']
    pred_files = {
        'SCReasoner': "results/SCReasoner/rscan_changeqa/results.json"
    }
    for key, value in pred_files.items():
        print(cls_score(qa_types, pred_file.replace('.json', '_wscore.json')))
        with open(f'leo_gpt.txt', 'a') as f:
            scores, overall = cls_score(qa_types, pred_file.replace('.json', '_wscore.json'))
            f.write(f'{key}: {scores} Overall: {overall}\n')
