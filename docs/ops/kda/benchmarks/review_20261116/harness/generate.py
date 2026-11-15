import sys,pathlib,json,hashlib
sys.path.insert(0,'/workspace/ready/main')
import jax,jax.numpy as jnp,numpy as np,ml_dtypes
from tokamax._src import numerics
from tokamax._src.ops.kda.arg_specs import ARG_SPECS
root=pathlib.Path('/workspace/ready/inputs');root.mkdir(exist_ok=True);manifest=[]
with jax.default_device(jax.devices('tpu')[0]):
 for spec in ARG_SPECS:
  kw=numerics.random_initialize(dict(spec.args),seed=619)
  kw['beta']=jnp.full_like(kw['beta'],.5)
  data={};meta={'name':spec.name,'tags':list(spec.tags),'static':{},'arrays':{}}
  digest=hashlib.sha256()
  for key,value in sorted(kw.items()):
   if isinstance(value,(jax.Array,np.ndarray)):
    x=np.asarray(value);dtype=str(x.dtype);meta['arrays'][key]={'shape':list(x.shape),'dtype':dtype};data[key]=x.view(np.uint16) if dtype=='bfloat16' else x;digest.update(key.encode());digest.update(str((x.shape,dtype)).encode());digest.update(x.tobytes())
   else:meta['static'][key]=value
  meta['input_sha256']=digest.hexdigest();np.savez(root/(spec.name+'.npz'),**data);manifest.append(meta);print(spec.name,meta['input_sha256'],flush=True)
assert len(manifest)==15
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
