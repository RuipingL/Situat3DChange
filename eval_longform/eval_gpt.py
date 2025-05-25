from openai import OpenAI
import json
import time
import os
def gpt_llm_eval(q, pred, a):
    system_prompt = ("You are an intelligent evaluator tasked with assessing the correctness and semantic similarity of model-generated answers to question-answering pairs."
                     "Your goal is to compare the predicted answer with the reference (correct) answer and assign a score based on how well they align in meaning. Use the following scoring rubric:\n"
                     "5 = Completely correct or semantically equivalent.\n"
                     "4 = Key information is correct, with minor inaccuracies or omissions.\n"
                     "3 = Some relevant information, but lacks sufficient correctness or completeness.\n"
                     "2 = Mostly incorrect, but shows some relevance to the question.\n"
                     "1 = Completely incorrect or nonsensical.\n"
                     "Your response must be a single integer from 1 to 5, with no additional text or explanation.")
    prompt = f"Question: {q}\nPredicted Answer: {pred}\nGround Truth: {a}\n Score:"
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    ouptut = chat_llm(messages)
    return ouptut
def chat_llm(chatgpt_messages, temperature=0, max_tokens=100, model='gpt-4o-mini-2024-07-18'):
    client = OpenAI(api_key="your-openai-key")
    response = client.chat.completions.create(
        model=model,
        messages=chatgpt_messages,
        max_tokens=max_tokens,
        temperature=temperature
    )


    return response.choices[0].message.content
def resposne_to_scores(pred_file, log_path):
    pred_js = json.load(open(pred_file, 'r'))
    for response in pred_js:
        # for i in range(len(responses)):
        question = response["instruction"].replace('USER: ', "").replace(" ASSISTANT:", "")
        score = gpt_llm_eval(question, response["response_pred"], response["response_gt"])
        response["score"] = score
        print(response["index"], question, response["response_pred"], response["response_gt"], score)
        with open(log_path, 'a') as log_file:
            log_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(response["index"]) + ' ' + question + ' ' + response["response_pred"] + ' ' + response["response_gt"] + ' ' + str(score) + '\n')
    return pred_js

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
def js_to_score(json_file):
    data_js = json.load(open(json_file, 'r'))
    score = 0
    num = 0
    for i in range(len(data_js)):
        score += int(data_js[i]["score"])
        num += 1
    return (score / num - 1)/4
if __name__ == "__main__":
    pred_file_cap = {
        'SCReasoner': "results/SCReasoner/rscan_changecap/results.json"}
    for key, value in pred_file_cap.items():
        if os.path.exists(value.replace('.json', '_wscore.json')):
            print(f"{value.replace('.json', '_wscore.json')} already exists")
            continue
        if not os.path.exists(value):
            print(f"{value} does not exist")
            continue
        log_path = value.replace('.json', '_output.log')
        pred = resposne_to_scores(value, log_path)
        json.dump(pred, open(value.replace('.json', '_wscore.json'), 'w'), indent=4)
        # log_to_js(log_path, value.replace('.json', '_wscore.json'))
        # print(js_to_score(value.replace('.json', '_wscore.json')))
        with open('caption_gpt.txt', 'a') as file:
            file.write(f"{key}: {js_to_score(value.replace('.json', '_wscore.json'))}\n")
