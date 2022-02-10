import torch
from torch.utils.data import Dataset

class Quotesdataset(Dataset):
  
  def __init__(self,data,tokenizer):
    self.data = data
    self.tokenizer = tokenizer
    self.eos_tok = "<|endoftext|>"

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    quote = self.data[idx]
    quote = f"Quote: {str(quote)} {self.eos_tok}"
   
    inputs = self.tokenizer.encode_plus(
            quote,
            None,
            padding='max_length', 
            add_special_tokens = True,
            truncation=True,
            max_length = 100,
        )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    

    return {'ids':torch.tensor(ids,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'target':torch.tensor(ids,dtype=torch.long)}
