from fastai.data.external import URLs, untar_data
from fastai.data.transforms import get_files
from fastai.text.core import WordTokenizer, Tokenizer, SentencePieceTokenizer
import pandas as pd
from fastcore.foundation import first, coll_repr, L

print(URLs.LOCAL_PATH)
print(URLs.WIKITEXT_TINY)

path = untar_data(URLs.WIKITEXT_TINY)

files = get_files(path, extensions=['.csv'])
train = pd.read_csv(files[0])
print(train.columns)

txts = L([i for i in train[0]])
print(txts)

# Spacy tokenization for lang=en (limited use in e.g. biomedical domain)
# spacy = WordTokenizer()
# # toks = first(spacy([txt]))
# # print(coll_repr(toks, 30))

# tok = Tokenizer.from_df(txt, text_cols=0, tok=SentencePieceTokenizer())
# tok.setup(txt)

toks = txts.map(tok)
toks[0]