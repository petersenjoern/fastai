#%%
from fastai.data.external import URLs, untar_data
from fastai.data.transforms import get_files, get_text_files, ColReader, RandomSplitter, parent_label,GrandparentSplitter
from fastai.data.block import DataBlock
from fastai.text.core import WordTokenizer, Tokenizer, SentencePieceTokenizer
from fastai.text.data import Numericalize, TextBlock, CategoryBlock
from fastai.text.learner import language_model_learner, text_classifier_learner
from fastai.text.models.awdlstm import AWD_LSTM
from fastai.metrics import Perplexity, accuracy
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.callback.schedule import fine_tune, lr_find, fit_one_cycle
from fastcore.foundation import first, coll_repr, L
from fastcore.utils import partialler
from functools import partial

import pandas as pd
import pathlib

#%%


#%%
if __name__ == '__main__':
    PATH_TENSORBOARD=URLs.LOCAL_PATH.joinpath('tmp','runs')
    print(PATH_TENSORBOARD)
    cbs=TensorBoardCallback(PATH_TENSORBOARD, trace_model=False) # Trace has to be false, because of mixed precision (FP16)

    # Prepare data for wikitext
    # path = untar_data(URLs.WIKITEXT_TINY)
    # files = get_files(path, extensions=['.csv'])
    # df_train = pd.read_csv(files[0], header=None)
    # df_valid = pd.read_csv(files[1], header=None)
    # df_all = pd.concat([df_train, df_valid])
    # df_all.columns = ["text"]


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
    # dls_lm = DataBlock(
    #     blocks=TextBlock.from_df("text", is_lm=True),
    #     get_x=ColReader("text"),
    #     splitter=RandomSplitter(0.1)
    #     )
    # dls_lm = dls_lm.dataloaders(df_all, bs=64, seq_len=72)
    # print(dls_lm.show_batch(max_n=3))


    # learn = language_model_learner(
    #     dls_lm, AWD_LSTM,
    #     metrics=[accuracy, Perplexity()]).to_fp16()
    # print(learn.model)

    # print(learn.lr_find())
    # learn.fine_tune(5, 1e-2, cbd=TensorBoardCallback(PATH_TENSORBOARD, trace_model=True))

    #%%
    # Prepare IMDB data
    path = untar_data(URLs.IMDB)
    bs=32

    # Fine-tune pretrained language model (based on wikitext) to the IMDB corpus
    get_imdb = partial(get_text_files, folders=["train", "test", "unsup"])
    dls_lm = DataBlock(
        blocks=TextBlock.from_folder(path, is_lm=True, n_workers=4),
        get_items=get_imdb,
        splitter=RandomSplitter(0.1)
    )
    dls_lm = dls_lm.dataloaders(path, path=path, bs=bs, seq_len=80)
    print(dls_lm.show_batch(max_n=3))

# #%%
    # learn = language_model_learner(
    #     dls_lm, AWD_LSTM, drop_mult=0.3,
    #     metrics=[accuracy, Perplexity()]).to_fp16()
    # learn.lr_find()
    # print(learn.model)
    # learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7,0.8), cbs=cbs)
    # learn.save("1epoch")

# #%%
    learn = language_model_learner(
        dls_lm, AWD_LSTM, drop_mult=0.3,
        metrics=[accuracy, Perplexity()]).to_fp16()
    learn = learn.load("1epoch")
    print(learn.model)
# #%%

    # learn.unfreeze()
    # learn.fit_one_cycle(1, 2e-3, moms=(0.8,0.7,0.8), cbs=cbs)
    # learn.save("5epochs")
    # learn.save_encoder("finetuned")



# #%%
#     # Classification
#     # learn = learn.load("5epochs")
#     # print(dls_lm.vocab)
    def read_tokenized_file(f): return L(f.read_text().split(' '))
    imdb_clas = DataBlock(
                        blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab, n_workers=0),CategoryBlock()),
                        get_x=read_tokenized_file,
                        get_y = parent_label,
                        get_items=partial(get_text_files, folders=['train', 'test']),
                        splitter=GrandparentSplitter(valid_name='test'))

    dbunch_clas = imdb_clas.dataloaders(path, path=path, bs=bs, seq_len=80)

    print(dbunch_clas.show_batch(max_n=7))

# #%%

    learn = text_classifier_learner(
        dbunch_clas,
        AWD_LSTM,
        drop_mult=0.5,
        n_workers=0,
        metrics=accuracy).to_fp16()

    learn = learn.load_encoder("finetuned")
    learn.lr_find()

#     #%%
    learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7, 0.8))
    learn.save('first')
    learn.load('first')

# # %%
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7, 0.8))
    learn.save('second')
    learn.load('second')

# #%%
    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3),moms=(0.8,0.7, 0.8))
    learn.save('third')
    learn.load('third')

# #%%
    learn.unfreeze()
    learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7, 0.8))
    learn.predict("I really loved that movie , it was awesome !")
# #%%
