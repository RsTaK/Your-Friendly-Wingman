import os
import math
import json
import random

import numpy as np
import pandas as pd

import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from train_config import Config
from engine import Engine
from model import get_model
from dataloader import Quotesdataset

def perform_run(csv_path, config, model, tokenizer, weight_path='./', load_weights_path=None):
    df=pd.read_csv(csv_path)
    df_1 = df[df.cleaned.apply(lambda x: len(x.split()))<100]

    print(f"Original Dataset rows: {df.shape[0]}")
    print(f"Dataset rows after selection: {df_1.shape[0]}")
    print(f"Rows dropped: {df.shape[0]-df_1.shape[0]}")

    datasett = Quotesdataset(df_1.cleaned.values, tokenizer)

    indices = list(range(len(datasett)))
    random.shuffle(indices)

    split = math.floor(0.3 * len(datasett))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(datasett, batch_size=config.batch_size, 
                                sampler=train_sampler, drop_last=True, num_workers=config.num_workers)
    val_loader = DataLoader(datasett, batch_size=config.batch_size, 
                            sampler=val_sampler, num_workers=config.num_workers)
    if load_weights_path is not None:
        model.load_state_dict(torch.load(load_weights_path + f"{config.save_file_name}.pt")["model_state_dict"]) 
        print("Weight Loaded")

    engine = Engine(model=model.to(config.device), device=config.device, 
                    config=config, save_file_name = config.save_file_name, 
                    weight_path=weight_path)
    
    engine.fit(train_loader, val_loader)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__=="__main__":
    no_deprecation_warning=True
    warnings.filterwarnings("ignore", category=UserWarning)
    seed_everything(42)

    with open("conf/data_config.json") as f:
        data_config = json.load(f)
        csv_path = data_config["save_cleaned_csv"]
        
    model, tokenizer = get_model()
    perform_run(csv_path, Config, model, tokenizer, Config.load_weights_path)
