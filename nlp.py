from fastai.data.external import URLs, untar_data
from fastai.data.transforms import get_files, ColReader, RandomSplitter
from fastai.data.block import DataBlock
from fastai.text.core import WordTokenizer, Tokenizer, SentencePieceTokenizer
from fastai.text.data import Numericalize, TextBlock
import pandas as pd
from fastcore.foundation import first, coll_repr, L

if __name__ == '__main__':
    print(URLs.LOCAL_PATH)
    print(URLs.WIKITEXT_TINY)

    # Download data in .fastai if not already exists
    path = untar_data(URLs.WIKITEXT_TINY)
    print(path)

    # get directory of filenames
    files = get_files(path, extensions=['.csv'])
    df_train = pd.read_csv(files[0], header=None)
    df_valid = pd.read_csv(files[1], header=None)
    df_all = pd.concat([df_train, df_valid])
    df_all.columns = ["text"]


    # txt = df_all.iloc[-1].values[0][:200] # Get subset of the data
    # txt = L([i for i in df_all[0]]) # List
    # print(txt)

    # spacy = WordTokenizer()
    # toks = first(spacy([txt]))
    # print(coll_repr(toks, 30))

    # Setup tokenizer
    # tok=Tokenizer.from_df(df_all, tok=SentencePieceTokenizer()) 
    # is by default using WordTokenizer. 
    # WordTokenizer is at the moment defaulting to Spacy.
    # Spacy tokenization for lang=en (limited use in e.g. biomedical domain)
    # tok.setup(df_all)  # class with encodes and decodes (its a transformer that can reverse)

    # toks=txt.map(tok)
    # print(toks)


    # Numericalization
    # num = Numericalize() # Class to reversible (encodes, decodes) numericalize tokens
    # num.setup(toks)
    # nums = toks.map(num)
    # print(nums[0][:10])

    # print(num.encodes(toks[0]))
    # print(num.decodes(nums[0][:10]))

    # Setup a processing pipeline (tokenization and numericalization) with DataBlock and TextBlock
    dls_lm = DataBlock(
        blocks=TextBlock.from_df("text", is_lm=True),
        get_x=ColReader("text"),
        splitter=RandomSplitter(0.1)
        )
    dls_lm = dls_lm.dataloaders(df_all, bs=64, seq_len=72)
    print(dls_lm.show_batch(max_n=3))

    learn = language_model_learner(
        dls_lm, AWD_LSTM,
        metrics=[accuracy, Perplexity()]).to_fp16()
    print(learn.model)


