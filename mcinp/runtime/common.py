"""Portable identity, atomic-file, and completed-artifact utilities."""
import hashlib,json,os,random,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from functools import lru_cache
from pathlib import Path
import numpy as np,torch
from threadpoolctl import threadpool_limits
# Limit BLAS threads for small dense cross-checks and time integration to avoid startup overhead; tolerances are unchanged.
threadpool_limits(1)
ROOT=Path(__file__).resolve().parent
__all__=['sha','canonical','identity','array_sha','load','write','save_npz','save_checkpoint','tensor_state','strict_load','source_hash','receipt','verify','seed_all','rng_state','restore_rng','synchronize','timer','elapsed']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def canonical(o):return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def identity(o):return hashlib.sha256(canonical(o).encode()).hexdigest()
def array_sha(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)
def save_npz(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    with open(tmp,'wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,p)
def save_checkpoint(p,obj):
    tmp=Path(p).with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,p)
def tensor_state(model):
    """Return the model state as a plain name-to-tensor dictionary, rejecting anything else."""
    state=model.state_dict()
    for k,v in state.items():
        if not isinstance(v,torch.Tensor) and k!='_metadata':raise ValueError('unregistered non-tensor state: '+k)
    return {k:v.detach().cpu().clone() for k,v in state.items() if isinstance(v,torch.Tensor)}
def strict_load(model,state):
    """Load ``state`` into ``model`` after checking keys, shapes, dtypes and finiteness."""
    expected=tensor_state(model)
    if set(state)!=set(expected):raise ValueError('tensor keys')
    for k,v in state.items():
        if not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all():raise ValueError('tensor schema '+k)
    model.load_state_dict(state,strict=True)
# Scientific source identity: the equation, the data contract, the training/evaluation logic and every
# method implementation. They are hashed as one set so that a run cannot silently mix code revisions.
# Documentation and the command-line layer are excluded, because they cannot change a result. Paths are
# relative to `mcinp/`.
SCIENTIFIC_SOURCES=('runtime/common.py','runtime/datasets.py','runtime/equation.py','runtime/pipeline.py')
@lru_cache(maxsize=1)
def source_hash():
    files=[ROOT.parent/n for n in SCIENTIFIC_SOURCES]
    files+=sorted(p for p in (ROOT.parent/'methods').rglob('*.py') if '__pycache__' not in p.parts)
    missing=[p for p in files if not p.is_file()]
    if missing:raise FileNotFoundError('scientific source file missing: '+', '.join(str(p) for p in missing))
    return identity({p.relative_to(ROOT.parent).as_posix():sha(p) for p in sorted(files)})
def receipt(p,binding,files,**details):write(p,dict(binding=binding,files={Path(f).name:sha(f) for f in files},**details))
def verify(p,binding=None):
    r=load(p)
    if binding is not None and r['binding']!=binding:raise ValueError('binding mismatch '+str(p))
    for name,h in r['files'].items():
        if sha(Path(p).parent/name)!=h:raise ValueError('artifact SHA mismatch '+name)
    return r
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def rng_state():
    s=np.random.get_state()
    return dict(python=random.getstate(),numpy=torch.tensor(s[1].astype(np.int64)),numpy_position=s[2],numpy_has_gauss=s[3],numpy_cached=s[4],torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(('MT19937',s['numpy'].numpy().astype(np.uint32),s['numpy_position'],s['numpy_has_gauss'],s['numpy_cached']));torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])
def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
def timer(device='cpu'):synchronize(device);return time.perf_counter()
def elapsed(start,device='cpu'):synchronize(device);return time.perf_counter()-start
