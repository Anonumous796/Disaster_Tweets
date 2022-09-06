import json
from tqdm.auto import tqdm
import re
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\d+[ ]+\d+[ ]+\d+|\d+[ ]+\d+|[a-zA-Z]+[.]+[a-zA-Z]+|[A-Z]+[a-z]+|\d+[.,:+-]+\d+|\w+')


def word_processor(word):
    word = word.replace(' ', '')
    word = word.replace('.', ',')
    word = word.replace('-', ':')
    return word


def clear_text(text):
    text = re.sub('[(]+[\dЗ\W]+[)]', '', text)
    text = [word_processor(word) for word in tokenizer.tokenize(text)]
    return ' '.join(text)


