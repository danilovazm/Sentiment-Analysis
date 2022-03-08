import spacy as sp
import pandas as pd
import numpy as np
import re

def preProcessing(df):
    df = re.sub(r"[^\w\s]", '', df)
    df = re.sub(r"\d", '', df)
    tokenize = sp.load("en_core_web_md")
    token_list = [token for token in tokenize(df)]
    tokensNSW = [token for token in token_list if not token.is_stop]
    tokensLemma = [token.lemma_ for token in tokensNSW]
    vectorizedTokens = [tokenize.vocab[token].vector for token in tokensLemma]
    if len(vectorizedTokens)>=100:
        return np.array(vectorizedTokens)[0:100]
    else:
        for i in range(len(vectorizedTokens), 100):
            vectorizedTokens.append(vectorizedTokens[i-len(vectorizedTokens)])
        return np.array(vectorizedTokens)

def dataLoading(x, y):
    Xvector = []
    Yvector = []
    count = 0
    for i in range(len(y)):
        Xvector.append(preProcessing(x.iloc[i]))
        if y.iloc[i] == 'positive':
            Yvector.append(1)
        else:
            Yvector.append(0)
    return Xvector, Yvector