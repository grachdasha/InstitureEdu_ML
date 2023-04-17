#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install -r requirements.txt --quiet')


# In[48]:


import pandas as pd
import matplotlib.pyplot as plt 
import random
import warnings
import nltk
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import word_tokenize
import re
nltk.download('punkt')


# In[8]:


def generate_start_text(df: pd.DataFrame, n_words: int) -> str:
    idx = random.choice(df.index)
    line = df.iloc[idx]["preprocessed_line"]
    tokens = word_tokenize(line)[:n_words]
    start = " ".join(tokens)
    start = re.sub(r'\s([?.!",](?:\s|$))', r'\1', start)
    return start


def load_tokenizer_and_model(model_name_or_path):
    return (
        GPT2Tokenizer.from_pretrained(model_name_or_path),
        GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda(),
    )


def generate(
    model,
    tok,
    text,
    do_sample=True,
    max_length=50,
    repetition_penalty=5.0,
    top_k=5,
    top_p=0.95,
    temperature=1,
    num_beams=None,
    no_repeat_ngram_size=3,
):
    input_ids = tok.encode(text, return_tensors="pt").cuda()
    out = model.generate(
        input_ids.cuda(),
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    return list(map(tok.decode, out))

def make_texts_examples_file(
    model_name,
    df,
    n_words,
    count=10,
    table_name="result.csv",
    max_length=200,
):
    df_list = []
    tok, model = load_tokenizer_and_model(model_name)
    if model_name.find("gpt3") or model_name.find("gpt2"):
        model.config.pad_token_id = model.config.eos_token_id
    for i in range(1, count + 1):
        start_text = generate_start_text(df, n_words=n_words)
        result = generate(
            model, tok, start_text, top_k=7, num_beams=12, max_length=max_length
        )[0]
        gen_part = result[len(start_text) :]
        df_list.append((start_text, gen_part, result))
    result_df = pd.DataFrame(df_list, columns=["Затравка", "Генерация", "Весь текст"])
    result_df.to_csv(table_name)


# ## Средняя школа

# In[62]:


with open('Neuro_texts.txt') as f:
    lines = f.readlines()

df_list = []
for line in lines:
    df_list.append((len(line.split(" ")), line))
    
df = pd.DataFrame(df_list, columns=["n_words", "line"])

# распределение по словам
plt.figure(figsize=(10,8))
plt.xlabel("Кол-во слов")
plt.ylabel("Частота")
plt.hist(x=df["n_words"], bins=50, range=(0,200))
plt.show()


# In[68]:


df[["preprocessed_line"]].to_csv("starter_middle.csv")


# In[5]:


# отфильтровываем короткие строки (диалоговые)
df = df[df["n_words"] >= 10].reset_index(drop=True)

# удаление \n
df["preprocessed_line"] = df["line"].apply(lambda x: x.replace("\n",""))


# In[9]:


#!g1.1
%%time
warnings.filterwarnings('ignore')
make_texts_examples_file(model_name="sberbank-ai/rugpt3large_based_on_gpt2",
                         df=df,
                         n_words=15)


# In[57]:


#!g1.1
result = pd.read_csv("result.csv")

print("***********Затравка*************")
print(result.iloc[9]["Затравка"], end="\n\n\n\n")
print("***********Генерация*************")
print(result.iloc[9]["Генерация"], end="\n\n\n\n")
print("***********Весь текст*************")
print(result.iloc[9]["Весь тексты"], end="\n\n\n\n")


# ## Младшая школа

# In[69]:


#!g1.1
with open('Neural_texts.txt') as f:
    lines = f.readlines()

df_list = []
for line in lines:
    df_list.append((len(line.split(" ")), line))
    
df = pd.DataFrame(df_list, columns=["n_words", "line"])

# распределение по словам
plt.figure(figsize=(10,8))
plt.xlabel("Кол-во слов")
plt.ylabel("Частота")
plt.hist(x=df["n_words"], bins=50, range=(0,200))
plt.show()


# In[70]:


#!g1.1
df["preprocessed_line"] = df["line"].apply(lambda x: x.replace("\n",""))


# In[71]:


#!g1.1
df[["preprocessed_line"]].to_csv("starter_beginner.csv")


# In[37]:


# отфильтровываем короткие строки (диалоговые)
df = df[df["n_words"] >= 5].reset_index(drop=True)

# удаление \n
df["preprocessed_line"] = df["line"].apply(lambda x: x.replace("\n",""))


# In[39]:


#!g1.1
%%time
warnings.filterwarnings('ignore')
make_texts_examples_file(model_name="sberbank-ai/rugpt3large_based_on_gpt2",
                         df=df,
                         n_words=15,
                         table_name="result_young.csv")


# In[60]:


#!g1.1
result = pd.read_csv("result_young.csv")

print("***********Затравка*************")
print(result.iloc[2]["Затравка"], end="\n\n\n\n")
print("***********Генерация*************")
print(result.iloc[2]["Генерация"], end="\n\n\n\n")
print("***********Весь текст*************")
print(result.iloc[2]["Весь тексты"], end="\n\n\n\n")


# In[ ]:


#!g1.1
get_ipython().run_line_magic('pinfo', 'pd.DataFrame')


# In[ ]:


#!g1.1

