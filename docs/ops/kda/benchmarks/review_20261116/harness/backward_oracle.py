import pathlib,json,sys,hashlib,time,os
os.environ.update(JAX_PLATFORMS='tpu,cpu',LIBTPU_INIT_ARGS='--xla_tpu_impure_enable_large_2nd_minor_layout=true')
sys.path.insert(0,'/workspace/ready/backward')
import jax,jax.numpy as jnp,numpy as np,ml_dtypes
from matrix import CASES
r=pathlib.Path('/workspace/ready');folder=r/'backward-oracle';folder.mkdir(exist_ok=True)
def host_inputs(c):
    b,t,h,n=(c[k] for k in ['b','t','h','n'])
    rng=np.random.default_rng(619)
    def normal(shape,std):return (rng.standard_normal(shape,dtype=np.float32)*std).astype(ml_dtypes.bfloat16)
    shape=(b,t,h,128)
    q,k,v,g=(normal(shape,s).transpose(2,0,1,3) for s in [.05,.05,.05,.1])
    if c['preactivated']:g=np.full(q.shape,-.01,ml_dtypes.bfloat16)
    logits=rng.standard_normal(shape[:-1],dtype=np.float32)*.1
    beta=(1/(1+np.exp(-logits))).astype(ml_dtypes.bfloat16).transpose(2,0,1)
    a=rng.uniform(-1,0,h).astype(np.float32);bias=rng.uniform(-4,-3,h*128).astype(np.float32)
    cot=normal(shape,.05).transpose(2,0,1,3)
    if c['cp']>1:
        ids=np.zeros((b,t),np.int32);local=t//c['cp']
        for rank in range(c['cp']):
            start=rank*local
            ids[:,start:start+local//2+1]=2*rank+1
            ids[:,start+local//2+1:start+local]=2*rank+3
        if c['preactivated']:ids[:]=1;ids[:,64:]=2
        if c['over']:ids[-1,-local+local//4:-local+local//2+1]=2*c['cp']
    else:
        lengths=np.full(n,t//n,np.int32);lengths[:t%n]+=1
        ids=np.stack([np.repeat(np.arange(1,n+1,dtype=np.int32),np.roll(lengths,row)) for row in range(b)])
    state=(rng.standard_normal((b,n,h,128,128),dtype=np.float32)*.01) if c['state'] else np.zeros((0,),np.float32)
    if c.get('batch_first'):
        q,k,v,g=(x.transpose(1,2,0,3) for x in (q,k,v,g))
        beta=beta.transpose(1,2,0);cot=cot.transpose(1,2,0,3)
    values=(q,k,v,g,beta,a,bias,ids,cot,state)
    sha=hashlib.sha256()
    for x in values:
        sha.update(str((x.shape,str(x.dtype))).encode());sha.update(np.ascontiguousarray(x).view(np.uint8))
    return values,sha.hexdigest()
def _grouped_recurrence(q, k, v, g, beta, a, bias, ids, *, context=None):
  """Independent FP32 recurrence for grouped labels; padding preserves state.

  Keep one active state instead of an unused-label state table, so a long CP
  gradient check does not retain that entire table at every scan iteration.
  """
  local_tokens = q.shape[2]
  if context is not None:
    axis = context.axis_name
    q, k, v, g, beta = [
        jax.lax.all_gather(x, axis, axis=2, tiled=True)
        for x in (q, k, v, g, beta)
    ]
    ids = jax.lax.all_gather(ids, axis, axis=1, tiled=True)
  heads, _, _, dim = q.shape
  gates = -5.0 * jax.nn.sigmoid(
      jnp.exp(a.astype(jnp.float32))[:, None, None, None]
      * (g.astype(jnp.float32) + bias.reshape(heads, 1, 1, dim))
  )
  xs = (
      q[:, 0].transpose(1, 0, 2),
      k[:, 0].transpose(1, 0, 2),
      v[:, 0].transpose(1, 0, 2),
      gates[:, 0].transpose(1, 0, 2),
      beta[:, 0].T,
      ids[0],
  )

  def step(carry, x):
    state, last = carry
    qi, ki, vi, gi, bi, label = x
    qi, ki, vi, gi, bi = [z.astype(jnp.float32) for z in (qi, ki, vi, gi, bi)]
    state = jnp.where((label > 0) & (label != last), 0.0, state)
    decayed = state * jnp.exp(gi)[..., None]
    prediction = jnp.einsum(
        "hk,hkv->hv", ki, decayed, precision=jax.lax.Precision.HIGHEST
    )
    updated = (
        decayed + (ki * bi[:, None])[..., None] * (vi - prediction)[:, None, :]
    )
    output = jnp.einsum(
        "hk,hkv->hv",
        qi * dim**-0.5,
        updated,
        precision=jax.lax.Precision.HIGHEST,
    )
    return (
        jnp.where(label > 0, updated, state),
        jnp.where(label > 0, label, last),
    ), jnp.where(label > 0, output, 0).astype(q.dtype)

  _, ys = jax.lax.scan(
      jax.checkpoint(step),
      (jnp.zeros((heads, dim, v.shape[-1]), jnp.float32), jnp.int32(0)),
      xs,
  )
  output = ys.transpose(1, 0, 2)[:, None]
  if context is not None:
    output = jax.lax.dynamic_slice_in_dim(
        output,
        jax.lax.axis_index(context.axis_name) * local_tokens,
        local_tokens,
        axis=2,
    )
  return output
c=dict(CASES['backward']['compact_packed'],batch_first=False)
values,digest=host_inputs(c)
report=json.loads((r/'features/backward-compact-0/report.json').read_text())
assert digest==report['cases']['compact_packed']['input_sha256']
inputs=tuple(jax.device_put(x,jax.devices()[0]) for x in values)
q,k,v,g,beta,a,bias,ids,dy,state=inputs
print('MATCHED_INPUTS',digest,flush=True)
def run(q,k,v,g,beta,a,bias,ids,dy):
 out,back=jax.vjp(lambda *xs:_grouped_recurrence(*xs,ids),q,k,v,g,beta,a,bias)
 return out,back(dy)
start=time.time();exe=jax.jit(run).lower(q,k,v,g,beta,a,bias,ids,dy).compile();print('COMPILED',time.time()-start,exe.memory_analysis(),flush=True)
result=jax.block_until_ready(exe(q,k,v,g,beta,a,bias,ids,dy));leaves=jax.tree.leaves(result)
raw={str(i):np.asarray(x).view(np.uint16) if x.dtype==jnp.bfloat16 else np.asarray(x) for i,x in enumerate(leaves)}
np.savez_compressed(folder/'independent.npz',**raw)
rows=[]
for variant in ['main','optimized','compact']:
 path=r/('features/backward-'+variant+'-0/compact_packed-full.npz');arrays=np.load(path)
 rec=report if variant=='compact' else json.loads((r/('features/backward-'+variant+'-0/report.json')).read_text())
 dtypes=rec['cases']['compact_packed']['modes']['full']['array_dtypes']
 errors=[]
 for i,key in enumerate(arrays.files):
  x=arrays[key].view(ml_dtypes.bfloat16) if dtypes[i]=='bfloat16' else arrays[key]
  y=np.asarray(leaves[i]);x,y=x.astype(np.float64),y.astype(np.float64)
  errors.append(dict(leaf=i,relative_rms=float(np.linalg.norm((x-y).ravel())/max(np.linalg.norm(y.ravel()),1e-30)),max_abs=float(np.max(np.abs(x-y)))))
 rows.append(dict(variant=variant,errors=errors));print('ERRORS',variant,errors,flush=True)
(folder/'comparison.json').write_text(json.dumps(dict(input_sha256=digest,rows=rows),indent=2));print('COMPLETE',flush=True)
