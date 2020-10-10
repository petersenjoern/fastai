from pathlib import Path
import os
import yaml
import requests
from fastprogress.fastprogress import progress_bar,master_bar
import json
import tarfile
import shutil
import spacy
from spacy.symbols import ORTH
import pandas as pd
from types import SimpleNamespace
from fastcore.foundation import first, coll_repr, L
from fastcore.imports import is_iter


class Config:
    "Setup config at `~/.fastai` unless it exists already."
    config_path = Path(os.getenv('FASTAI_HOME', '~/.fastai')).expanduser()
    config_file = config_path/'config.yml'

    def __init__(self):
        self.config_path.mkdir(parents=True, exist_ok=True)
        if not self.config_file.exists(): self.create_config()
        self.d = self.load_config()

    def __getitem__(self,k):
        k = k.lower()
        if k not in self.d: k = k+'_path'
        return Path(self.d[k])

    def __getattr__(self,k):
        if k=='d': raise AttributeError
        return self[k]

    def __setitem__(self,k,v): self.d[k] = str(v)
    def __contains__(self,k): return k in self.d

    def load_config(self):
        "load and return config if version equals 2 in existing, else create new config."
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            if 'version' in config and config['version'] == 2: return config
            elif 'version' in config: self.create_config(config)
            else: self.create_config()
        return self.load_config()

    def create_config(self, cfg=None):
        "create new config with default paths and set `version` to 2."
        config = {'data_path':    str(self.config_path/'data'),
                  'archive_path': str(self.config_path/'archive'),
                  'storage_path': '/tmp',
                  'model_path':   str(self.config_path/'models'),
                  'version':      2}
        if cfg is not None:
            cfg['version'] = 2
            config = merge(config, cfg)
        self.save_file(config)

    def save(self): self.save_file(self.d)
    def save_file(self, config):
        "save config file at default config location `~/.fastai/config.yml`."
        with self.config_file.open('w') as f: yaml.dump(config, f, default_flow_style=False)


class URLs():
    "Global constants for dataset and model URLs."
    LOCAL_PATH = Path.cwd()
    MDL = 'http://files.fast.ai/models/'
    S3  = 'https://s3.amazonaws.com/fast-ai-'
    URL = f'{S3}sample/'
    S3_NLP      = f'{S3}nlp/'

    # NLP datasets
    WIKITEXT_TINY           = f'{S3_NLP}wikitext-2.tgz'
    
    def path(url='.', c_key='archive'):
        "Return local path where to download based on `c_key`"
        fname = url.split('/')[-1]
        local_path = URLs.LOCAL_PATH/('models' if c_key=='models' else 'data')/fname
        if local_path.exists(): return local_path
        return Config()[c_key]/fname

def download_url(url, dest, overwrite=False, pbar=None, show_progress=True, chunk_size=1024*1024,
                 timeout=4, retries=5):
    "Download `url` to `dest` unless it exists and not `overwrite`"
    if os.path.exists(dest) and not overwrite: return

    s = requests.Session()
    s.mount('http://',requests.adapters.HTTPAdapter(max_retries=retries))
    # additional line to identify as a firefox browser, see fastai/#2438
    s.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:71.0) Gecko/20100101 Firefox/71.0'})
    u = s.get(url, stream=True, timeout=timeout)
    try: file_size = int(u.headers["Content-Length"])
    except: show_progress = False

    with open(dest, 'wb') as f:
        nbytes = 0
        if show_progress: pbar = progress_bar(range(file_size), leave=False, parent=pbar)
        try:
            if show_progress: pbar.update(0)
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                if show_progress: pbar.update(nbytes)
                f.write(chunk)
        except requests.exceptions.ConnectionError as e:
            fname = url.split('/')[-1]
            data_dir = dest.parent
            print(f'\n Download of {url} has failed after {retries} retries\n'
                  f' Fix the download manually:\n'
                  f'$ mkdir -p {data_dir}\n'
                  f'$ cd {data_dir}\n'
                  f'$ wget -c {url}\n'
                  f'$ tar xf {fname}\n'
                  f' And re-run your code once the download is successful\n')

def newest_folder(path):
    "Return newest folder on path"
    list_of_paths = path.glob('*')
    return max(list_of_paths, key=lambda p: p.stat().st_ctime)


def rename_extracted(dest):
    "Rename file if different from dest"
    extracted = newest_folder(dest.parent)
    if not (extracted.name == dest.name): extracted.rename(dest)


def _add_check(url, fname):
    "Internal function to update the internal check file with `url` and check on `fname`."
    checks = json.load(open(Path(__file__).parent/'checks.txt', 'r'))
    checks[url] = _check_file(fname)
    json.dump(checks, open(Path(__file__).parent/'checks.txt', 'w'), indent=2)

def _check_file(fname):
    "internal function to get the hash of the local file at `fname`."
    size = os.path.getsize(fname)
    with open(fname, "rb") as f: hash_nb = hashlib.md5(f.read(2**20)).hexdigest()
    return [size,hash_nb]

def _get_check(url):
    "internal function to get the hash of the file at `url`."
    checks = json.load(open(Path(__file__).parent/'checks.txt', 'r'))
    return checks.get(url, '')


def _try_from_storage(dest, storage):
    "an internal function to create symbolic links for files from `storage` to `dest` if `storage` exists"
    if not storage.exists(): return
    os.makedirs(dest, exist_ok=True)
    for f in storage.glob('*'): os.symlink(f, dest/f.name, target_is_directory=f.is_dir())


def download_data(url, fname=None, c_key='archive', force_download=False, timeout=4):
    "Download `url` to `fname`."
    fname = Path(fname or URLs.path(url, c_key=c_key))
    fname.parent.mkdir(parents=True, exist_ok=True)
    if not fname.exists() or force_download: download_url(url, fname, overwrite=force_download, timeout=timeout)
    return fname

# Cell
def file_extract(fname, dest=None):
    "Extract `fname` to `dest` using `tarfile` or `zipfile`."
    if dest is None: dest = Path(fname).parent
    fname = str(fname)
    if   fname.endswith('gz'):  tarfile.open(fname, 'r:gz').extractall(dest)
    elif fname.endswith('zip'): zipfile.ZipFile(fname     ).extractall(dest)
    else: raise Exception(f'Unrecognized archive: {fname}')

def untar_data(url, fname=None, dest=None, c_key='data', force_download=False, extract_func=file_extract, timeout=4):
    "Download `url` to `fname` if `dest` doesn't exist, and un-tgz or unzip to folder `dest`."
    default_dest = URLs.path(url, c_key=c_key).with_suffix('')
    dest = default_dest if dest is None else Path(dest)/default_dest.name
    fname = Path(fname or URLs.path(url))
    force_download = True
    if force_download:
        if fname.exists(): os.remove(fname)
        if dest.exists(): shutil.rmtree(dest)
    if not dest.exists(): _try_from_storage(dest, URLs.path(url, c_key='storage').with_suffix(''))
    if not dest.exists():
        fname = download_data(url, fname=fname, c_key=c_key, timeout=timeout)
        extract_func(fname, dest.parent)
        rename_extracted(dest)
    return dest

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

def noops(self, x=None, *args, **kwargs):
    "Do nothing (method)"
    return x

def setify(o):
    "Turn any list like-object into a set."
    return o if isinstance(o,set) else set(L(o))

def is_indexer(idx):
    "Test whether `idx` will index a single item in a list"
    return isinstance(idx,int) or not getattr(idx,'ndim',1)


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def _listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str) or _is_array(o): return [o]
    if is_iter(o): return list(o)
    return [o]

class CollBase:
    "Base class for composing a list of `items`"
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, k): return self.items[list(k) if isinstance(k,CollBase) else k]
    def __setitem__(self, k, v): self.items[list(k) if isinstance(k,CollBase) else k] = v
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self): return self.items.__repr__()
    def __iter__(self): return self.items.__iter__()

class _L_Meta(type):
    def __call__(cls, x=None, *args, **kwargs):
        if not args and not kwargs and x is not None and isinstance(x,cls): return x
        return super().__call__(x, *args, **kwargs)

class GetAttr:
    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default='default'
    def _component_attr_filter(self,k):
        if k.startswith('__') or k in ('_xtra',self._default): return False
        xtra = getattr(self,'_xtra',None)
        return xtra is None or k in xtra
    def _dir(self): return [k for k in dir(getattr(self,self._default)) if self._component_attr_filter(k)]
    def __getattr__(self,k):
        if self._component_attr_filter(k):
            attr = getattr(self,self._default,None)
            if attr is not None: return getattr(attr,k)
        raise AttributeError(k)
    def __dir__(self): return custom_dir(self,self._dir())
#     def __getstate__(self): return self.__dict__
    def __setstate__(self,data): self.__dict__.update(data)

class L(GetAttr, CollBase, metaclass=_L_Meta):
    "Behaves like a list of `items` but can also index with list of indices or masks"
    _default='items'
    def __init__(self, items=None, *rest, use_list=False, match=None):
        if rest: items = (items,)+rest
        if items is None: items = []
        if (use_list is not None) or not _is_array(items):
            items = list(items) if use_list else _listify(items)
        if match is not None:
            if is_coll(match): match = len(match)
            if len(items)==1: items = items*match
            else: assert len(items)==match, 'Match length mismatch'
        super().__init__(items)

    @property
    def _xtra(self): return None
    def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    def __getitem__(self, idx): return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    def copy(self): return self._new(self.items.copy())

    def _get(self, i):
        if is_indexer(i) or isinstance(i,slice): return getattr(self.items,'iloc',self.items)[i]
        i = mask2idxs(i)
        return (self.items.iloc[list(i)] if hasattr(self.items,'iloc')
                else self.items.__array__()[(i,)] if hasattr(self.items,'__array__')
                else [self.items[i_] for i_ in i])

    def __setitem__(self, idx, o):
        "Set `idx` (can be list of indices, or mask, or int) items to `o` (which is broadcast if not iterable)"
        if isinstance(idx, int): self.items[idx] = o
        else:
            idx = idx if isinstance(idx,L) else _listify(idx)
            if not is_iter(o): o = [o]*len(idx)
            for i,o_ in zip(idx,o): self.items[i] = o_

    def __eq__(self,b):
        if isinstance_str(b, 'ndarray'): return array_equal(b, self)
        if isinstance(b, (str,dict)): return False
        return all_equal(b,self)

    def sorted(self, key=None, reverse=False): return self._new(sorted_ex(self, key=key, reverse=reverse))
    def __iter__(self): return iter(self.items.itertuples() if hasattr(self.items,'iloc') else self.items)
    def __contains__(self,b): return b in self.items
    def __reversed__(self): return self._new(reversed(self.items))
    def __invert__(self): return self._new(not i for i in self)
    def __repr__(self): return repr(self.items) if _is_array(self.items) else coll_repr(self)
    def __mul__ (a,b): return a._new(a.items*b)
    def __add__ (a,b): return a._new(a.items+_listify(b))
    def __radd__(a,b): return a._new(b)+a
    def __addi__(a,b):
        a.items += list(b)
        return a

    @classmethod
    def split(cls, s, sep=None, maxsplit=-1): return cls(s.split(sep,maxsplit))
    @classmethod
    def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))

    def map(self, f, *args, gen=False, **kwargs): return self._new(map_ex(self, f, *args, gen=gen, **kwargs))
    def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    def filter(self, f=noop, negate=False, gen=False, **kwargs):
        return self._new(filter_ex(self, f=f, negate=negate, gen=gen, **kwargs))

    def unique(self): return L(dict.fromkeys(self).keys())
    def enumerate(self): return L(enumerate(self))
    def renumerate(self): return L(renumerate(self))
    def val2idx(self): return {v:k for k,v in self.enumerate()}
    def itemgot(self, *idxs):
        x = self
        for idx in idxs: x = x.map(itemgetter(idx))
        return x

    def attrgot(self, k, default=None):
        return self.map(lambda o: o.get(k,default) if isinstance(o, dict) else nested_attr(o,k,default))
    def cycle(self): return cycle(self)
    def map_dict(self, f=noop, *args, gen=False, **kwargs): return {k:f(k, *args,**kwargs) for k in self}
    def map_filter(self, f=noop, g=noop, *args, gen=False, **kwargs):
        res = filter(g, self.map(f, *args, gen=gen, **kwargs))
        if gen: return res
        return self._new(res)
    def map_first(self, f=noop, g=noop, *args, **kwargs):
        return first(self.map_filter(f, g, *args, gen=False, **kwargs))

    def starmap(self, f, *args, **kwargs): return self._new(itertools.starmap(partial(f,*args,**kwargs), self))
    def zip(self, cycled=False): return self._new((zip_cycle if cycled else zip)(*self))
    def zipwith(self, *rest, cycled=False): return self._new([self, *rest]).zip(cycled=cycled)
    def map_zip(self, f, *args, cycled=False, **kwargs): return self.zip(cycled=cycled).starmap(f, *args, **kwargs)
    def map_zipwith(self, f, *rest, cycled=False, **kwargs):
        return self.zipwith(*rest, cycled=cycled).starmap(f, **kwargs)
    def shuffle(self):
        it = copy(self.items)
        random.shuffle(it)
        return self._new(it)

    def concat(self): return self._new(itertools.chain.from_iterable(self.map(L)))
    def reduce(self, f, initial=None): return reduce(f, self) if initial is None else reduce(f, self, initial)
    def sum(self): return self.reduce(operator.add)
    def product(self): return self.reduce(operator.mul)
    def setattrs(self, attr, val): [setattr(o,attr,val) for o in self]

def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders=L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(folders) !=0 and i==0: d[:] = [o for o in d if o in folders]
            else:                         d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) !=0 and i==0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return L(res)

def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a

class SpacyTokenizer():
    "Spacy tokenizer for `lang`"
    def __init__(self, lang='en', special_toks=None, buf_sz=5000):
        self.special_toks = ifnone(special_toks, defaults.text_spec_tok)
        nlp = spacy.blank(lang, disable=["parser", "tagger", "ner"])
        for w in self.special_toks: nlp.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pipe,self.buf_sz = nlp.pipe,buf_sz

    def __call__(self, items):
        return (L(doc).attrgot('text') for doc in self.pipe(map(str,items), batch_size=self.buf_sz))

# Cell
WordTokenizer = SpacyTokenizer


defaults = SimpleNamespace()
UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split()
defaults.text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ]

def _is_array(x): return hasattr(x,'__array__') or hasattr(x,'iloc')

def nested_attr(o, attr, default=None):
    "Same as `getattr`, but if `attr` includes a `.`, then looks inside nested objects"
    try:
        for a in attr.split("."): o = getattr(o, a)
    except AttributeError: return default
    return o

class bind:
    "Same as `partial`, except you can use `arg0` `arg1` etc param placeholders"
    def __init__(self, fn, *pargs, **pkwargs):
        self.fn,self.pargs,self.pkwargs = fn,pargs,pkwargs
        self.maxi = max((x.i for x in pargs if isinstance(x, _Arg)), default=-1)

    def __call__(self, *args, **kwargs):
        args = list(args)
        kwargs = {**self.pkwargs,**kwargs}
        for k,v in kwargs.items():
            if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
        fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
        return self.fn(*fargs, **kwargs)

def map_ex(iterable, f, *args, gen=False, **kwargs):
    "Like `map`, but use `bind`, and supports `str` and indexing"
    g = (bind(f,*args,**kwargs) if callable(f)
         else f.format if isinstance(f,str)
         else f.__getitem__)
    res = map(g, iterable)
    if gen: return res
    return list(res)


_tfm_methods = 'encodes','decodes','setups'

class _TfmDict(dict):
    def __setitem__(self,k,v):
        if k not in _tfm_methods or not callable(v): return super().__setitem__(k,v)
        if k not in self: super().__setitem__(k,TypeDispatch())
        self[k].add(v)


class _TfmMeta(type):
    def __new__(cls, name, bases, dict):
        res = super().__new__(cls, name, bases, dict)
        for nm in _tfm_methods:
            base_td = [getattr(b,nm,None) for b in bases]
            if nm in res.__dict__: getattr(res,nm).bases = base_td
            else: setattr(res, nm, TypeDispatch(bases=base_td))
        res.__signature__ = inspect.signature(res.__init__)
        return res

    def __call__(cls, *args, **kwargs):
        f = args[0] if args else None
        n = getattr(f,'__name__',None)
        if callable(f) and n in _tfm_methods:
            getattr(cls,n).add(f)
            return f
        return super().__call__(*args, **kwargs)

    @classmethod
    def __prepare__(cls, name, bases): return _TfmDict()


class Transform(metaclass=_TfmMeta):
    "Delegates (`__call__`,`decode`,`setup`) to (<code>encodes</code>,<code>decodes</code>,<code>setups</code>) if `split_idx` matches"
    split_idx,init_enc,order,train_setup = None,None,0,None
    def __init__(self, enc=None, dec=None, split_idx=None, order=None):
        self.split_idx = ifnone(split_idx, self.split_idx)
        if order is not None: self.order=order
        self.init_enc = enc or dec
        if not self.init_enc: return

        self.encodes,self.decodes,self.setups = TypeDispatch(),TypeDispatch(),TypeDispatch()
        if enc:
            self.encodes.add(enc)
            self.order = getattr(enc,'order',self.order)
            if len(type_hints(enc)) > 0: self.input_types = first(type_hints(enc).values())
            self._name = _get_name(enc)
        if dec: self.decodes.add(dec)

    @property
    def name(self): return getattr(self, '_name', _get_name(self))
    def __call__(self, x, **kwargs): return self._call('encodes', x, **kwargs)
    def decode  (self, x, **kwargs): return self._call('decodes', x, **kwargs)
    def __repr__(self): return f'{self.name}:\nencodes: {self.encodes}decodes: {self.decodes}'

    def setup(self, items=None, train_setup=False):
        train_setup = train_setup if self.train_setup is None else self.train_setup
        return self.setups(getattr(items, 'train', items) if train_setup else items)

    def _call(self, fn, x, split_idx=None, **kwargs):
        if split_idx!=self.split_idx and self.split_idx is not None: return x
        return self._do_call(getattr(self, fn), x, **kwargs)

    def _do_call(self, f, x, **kwargs):
        if not _is_tuple(x):
            if f is None: return x
            ret = f.returns_none(x) if hasattr(f,'returns_none') else None
            return retain_type(f(x, **kwargs), x, ret)
        res = tuple(self._do_call(f, x_, **kwargs) for x_ in x)
        return retain_type(res, x)



class Tokenizer(Transform):
    "Provides a consistent `Transform` interface to tokenizers operating on `DataFrame`s and folders"
    input_types = (str, list, L, tuple, Path)
    def __init__(self, tok, rules=None, counter=None, lengths=None, mode=None, sep=' '):
        if isinstance(tok,type): tok=tok()
        store_attr('tok,counter,lengths,mode,sep')
        self.rules = defaults.text_proc_rules if rules is None else rules

    @classmethod
    @delegates(tokenize_df, keep=True)
    def from_df(cls, text_cols, tok=None, rules=None, sep=' ', **kwargs):
        if tok is None: tok = WordTokenizer()
        res = cls(tok, rules=rules, mode='df')
        res.kwargs,res.train_setup = merge({'tok': tok}, kwargs),False
        res.text_cols,res.sep = text_cols,sep
        return res

    @classmethod
    @delegates(tokenize_folder, keep=True)
    def from_folder(cls, path, tok=None, rules=None, **kwargs):
        path = Path(path)
        if tok is None: tok = WordTokenizer()
        output_dir = tokenize_folder(path, tok=tok, rules=rules, **kwargs)
        res = cls(tok, counter=(output_dir/fn_counter_pkl).load(),
                  lengths=(output_dir/fn_lengths_pkl).load(), rules=rules, mode='folder')
        res.path,res.output_dir = path,output_dir
        return res

    def setups(self, dsets):
        if not self.mode == 'df' or not isinstance(dsets.items, pd.DataFrame): return
        dsets.items,count = tokenize_df(dsets.items, self.text_cols, rules=self.rules, **self.kwargs)
        if self.counter is None: self.counter = count
        return dsets

    def encodes(self, o:Path):
        if self.mode=='folder' and str(o).startswith(str(self.path)):
            tok = self.output_dir/o.relative_to(self.path)
            return L(tok.read().split(' '))
        else: return self._tokenize1(o.read())

    def encodes(self, o:str): return self._tokenize1(o)
    def _tokenize1(self, o): return first(self.tok([compose(*self.rules)(o)]))

    def get_lengths(self, items):
        if self.lengths is None: return None
        if self.mode == 'df':
            if isinstance(items, pd.DataFrame) and 'text_lengths' in items.columns: return items['text_length'].values
        if self.mode == 'folder':
            try:
                res = [self.lengths[str(Path(i).relative_to(self.path))] for i in items]
                if len(res) == len(items): return res
            except: return None

    def decodes(self, o): return TitledStr(self.sep.join(o))


if __name__ == "__main__":
    path = untar_data(URLs.WIKITEXT_TINY)
    print(path)

    files = get_files(path, extensions=['.csv'])
    df_train = pd.read_csv(files[0], header=None)
    df_valid = pd.read_csv(files[1], header=None)
    df_all = pd.concat([df_train, df_valid])

    print(df_all.head())
    # print(df_train.iloc[-1].values)

    txt = df_all.iloc[-1].values[0][:200]
    print(txt)

    spacy = WordTokenizer()
    toks = first(spacy([txt]))
    print(toks)

    toks=Tokenizer.from_df(0)
    print(toks)



    # https://fastpages.fast.ai/fastcore/#:~:text=Type%20dispatch%20can%20be%20especially%20useful%20in%20data,Unfortunately%2C%20Python%20does%20not%20support%20this%20out-of-the%20box.