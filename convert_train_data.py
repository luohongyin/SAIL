import sys
import json

import random
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

def cls_evaluate(tok, model, txt_batch):
    input_enc = tok(
        txt_batch,
        max_length = 512,
        padding = 'longest',
        return_tensors = 'pt',
        truncation = True,
        return_attention_mask = True,
        verbose = True
    )

    input_ids = input_enc.input_ids.cuda()
    attn_mask = input_enc.attention_mask.cuda()

    with torch.no_grad():
        results = model(
            input_ids = input_ids,
            attention_mask = attn_mask
        )

    return results.logits


def format_case(case):
    title = case['title']
    body = case['body']
    return f'{title}\n{body}'


def format_search_res(search_res_list, bm25_list, rand_num, tok, model, gpt_str, n = 5):
    
    if search_res_list is not None:
        search_res_list = search_res_list[:n]
        search_res_list.reverse()
        search_res_list = [
            f'{format_case(x)}' for i, x in enumerate(search_res_list)
        ]
    else:
        if len(bm25_list) == 0 or bm25_list is None:
            return None, None
        search_res_list = []

    search_res_list += bm25_list[:1]

    if rand_num < 0.2:
        return None, None
    
    elif rand_num < 0.3:
        num_res = 1
    
    elif rand_num < 0.6:
        num_res = 2
    
    elif rand_num < 0.6:
        num_res = 3
    
    num_res = min(num_res, len(search_res_list))
    search_res_list = random.sample(search_res_list, num_res)
    ent_list = [
        f'{x} is entailed by {gpt_str}' for x in search_res_list
    ]

    logits = cls_evaluate(tok, model, ent_list)
    ent_pred = (logits[:, 0] > logits[:, 2]).float().tolist()

    if sum(ent_pred) == 0:
        if len(search_res_list) == 1:
            verify_str = 'The search result is distracting. I should ignore the search result and only utilize my own knowledge.'
        else:
            verify_str = 'All search results are distracting, I should ignore all search results and only utilize my own knowledge.'
    
    elif sum(ent_pred) == len(search_res_list):
        if len(search_res_list) == 1:
            verify_str = 'The search result is information, and I should ignore the search result and only utilize my own knowledge.'
        else:
            verify_str = 'All search results are informative, and I can utilize all search results and my knowledge.'
    
    else:
        item_list = []
        label_list = []
        for i, ent in enumerate(ent_pred):
            if ent == 1:
                item_list.append(f'search result ({i + 1}) is informative')
                label_list.append(f'({i + 1})')
            else:
                item_list.append(f'search result ({i + 1}) is distracting')
        
        item_list[-1] = f'and {item_list[-1]}'
        itemized_ent = ', '.join(item_list).capitalize()
        label_str = ', '.join(label_list)

        if len(label_list) == 1:
            res_term = 'result'
        else:
            res_term = 'results'

        verify_str = f'{itemized_ent}. I will utilize the informative search {res_term} {label_str} and my knowledge.'

    search_res_str = '\n\n'.join([
        f'({i + 1}) {x}' for i, x in enumerate(search_res_list)
    ])

    return search_res_str, verify_str


def format_search_res_rand(search_res_all, n = 3):
    search_res_list = random.sample(search_res_all, n)

    search_res_str = '\n\n'.join(search_res_list)
    return search_res_str


def convert(data_idx, data_case, rand_num, tok, model, search_res_all=None):
    gpt_str = data_case['output']

    if search_res_all is None:
        search_res_str, verify_str = format_search_res(
            data_case['search_res'], data_case['bm25_res'], rand_num, tok, model, gpt_str, n = 6
        )
    else:
        search_res_str = format_search_res_rand(search_res_all)
    
    if data_case['input'] is None:
        data_case['input'] = ''
    data_case['input'] = data_case['input'].strip()

    if len(data_case['input']) == 0:
        if search_res_str is not None:
            human_str = f'\n{search_res_str}\n\n### Instruction: {data_case["instruction"]}'
        else:
            human_str = f'\nNone\n\n### Instruction: {data_case["instruction"]}'

    else:
        if search_res_str is not None:
            human_str = f'\n{search_res_str}\n\n### Instruction: {data_case["instruction"]}\n### Input: {data_case["input"]}'
        else:
            human_str = f'\nNone\n\n### Instruction: {data_case["instruction"]}\n### Input: {data_case["input"]}'
    
    if verify_str is not None:
        gpt_str = f'{verify_str} {gpt_str}'
    
    conversation = {
        'id': f'conv_{data_idx}',
        'conversations': [
            {
                'from': 'human',
                'value': human_str,
            },
            {
                'from': 'gpt',
                'value': gpt_str
            }
        ]
    }
    return conversation


def merge_data():
    gpt4_data = json.load(open('data/alpaca_gpt4_data.json'))
    search_data = json.load(open('data/search_res_only.json'))

    for i, search_res in enumerate(search_data):
        gpt4_data['search_res'] = search_res[0]
        gpt4_data['bm25_res'] = search_res[1]
    
    return gpt4_data


if __name__ == '__main__':
    num_split = 16
    dataset = []

    model_type_str = 'roberta'

    if model_type_str == 'deberta':
        tokenizer_str = f'microsoft/{model_type_str}-large'
    
    elif model_type_str == 'roberta':
        tokenizer_str = f'{model_type_str}-large'
    
    model_file_str = f'luohy/ESP-{model_type_str}-large'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
    model = AutoModelForSequenceClassification.from_pretrained(model_file_str).cuda()

    model.eval()

    dataset = merge_data()
    rand_tensor = torch.rand(len(dataset))

    rand_table = rand_tensor.tolist()
    
    dataset = [
        convert(i, x, rand_table[i], tokenizer, model) for i, x in enumerate(dataset)
    ]

    json.dump(dataset, open('data/SAIL_train.json', 'w'))