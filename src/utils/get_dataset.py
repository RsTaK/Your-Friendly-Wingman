from operator import index
import re
import json
import pandas as pd

import contractions
from textblob import TextBlob

def check(string, labels) :
    flag = 0
    for each_label in labels:
        regex = re.compile(f"\w*{each_label}\w*")
        match_object = regex.findall(string)
        if len(match_object) != 0:
            flag = 1
    return flag

def clean_dataset(string):
    expanded_words = list()
    string = string.lower()
    string = re.sub(' +', ' ', string)
    for each_word in string.split():
        expanded_words.append(contractions.fix(each_word))
    string = ' '.join(expanded_words)
    string = str(TextBlob(string).correct())
    string = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", string)
    return string

if __name__=="__main__":
    with open("../conf/data_config.json") as f:
        data_config = json.load(f)
        
    df = pd.read_csv(data_config["data_path"])
    labels = data_config["labels"]

    for i, rows in df.iterrows():
        try:
            if check(rows.Type):
                df.loc[i, 'Allow'] = 1
            else:
                df.loc[i, 'Allow'] = 0
        except:
            print(f"Faulty rows: {rows}")
    
    df.cleaned = df.Quotes.apply(lambda x: clean_dataset(x))
    df.to_csv(data_config["save_cleaned_csv"], index=False)
