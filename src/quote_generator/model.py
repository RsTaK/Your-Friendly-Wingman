from train_config import Config
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def get_model():
    tokenizer = GPT2Tokenizer.from_pretrained(Config.MODEL_NAME)
    SPECIAL_TOKENS_DICT = {
        'pad_token': '<pad>',
    }
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    model = GPT2LMHeadModel.from_pretrained(Config.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer