
import re
import numpy as np 

import torch

import torch.nn.functional as F
from onnxruntime import InferenceSession

from transformers import GPT2Tokenizer

from src.quote_generator.top_k_top_p_filtering import top_k_top_p_filtering
from src.quote_generator.train_config import Config

import torch.nn.functional as F


from flask import Flask, request, render_template, send_from_directory, make_response,jsonify


def preprocess_text(text):
    text = text[7:-15]

    punc_filter = re.compile('([.!?]\s*)')
    split_with_punctuation = punc_filter.split(text)

    text = ''.join([i.capitalize() for i in split_with_punctuation])
    
    return text

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(f"{Config.huggingFace_tokenizer}")
    SPECIAL_TOKENS_DICT = {
        'pad_token': '<pad>',
    }
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    return tokenizer

def predict(session, tokenizer, length_for_quote, no_of_quote = 1, temperature = 0.7, top_k = 0, top_p = 0.0):

    with torch.no_grad():
        for _ in range(no_of_quote):
        
            flag = 0
            cur_ids = tokenizer("Quote:", return_tensors="np")
           
            for _ in range(length_for_quote):
                outputs=session.run(output_names=["logits"], input_feed=dict(cur_ids))

                next_token_logits = outputs[0][0, -1] / (temperature if temperature > 0 else 1.)
                filtered_logits = top_k_top_p_filtering(torch.tensor(next_token_logits), top_k=top_k, top_p=top_p)

                if temperature == 0: # greedy sampling:
                  next_token = torch.argmax(F.softmax(filtered_logits, dim=-1)).unsqueeze(-1)
                  
                else:
                  next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                
                input_ids = torch.cat([torch.tensor(cur_ids['input_ids']), torch.ones((1,1)).long().to("cpu") * next_token], dim = 1) # Add the last word to the running sequence
                cur_ids = {'input_ids': np.array(input_ids), 'attention_mask': np.array([[1]* len(input_ids[0])])}
                
                if next_token in tokenizer.encode('<|endoftext|>'):
                    flag = 1
                    break

            if flag:
                
                output_list = list(cur_ids['input_ids'].squeeze())
                output_text = tokenizer.decode(output_list)

                output_text = preprocess_text(output_text)
                return output_text


app = Flask(__name__)
tokenizer = get_tokenizer()
session = InferenceSession(f"{Config.optimized_int8_model_path}/gpt2_int8.onnx")

@app.route("/")
def generate():
    text = predict(session = session, 
            tokenizer = tokenizer,
            length_for_quote = 50,
            temperature=0.7,
            top_k=50,
            top_p=0.95)
    return text
if __name__ == "__main__":
    app.run(debug=True)
