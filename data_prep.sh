#!/bin/bash

mkdir data
cd data/
wget https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data.json
wget https://people.csail.mit.edu/hyluo/data/search_res_only.json

cd ..
python convert_train_data.py roberta