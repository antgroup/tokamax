import pathlib,json,sys,numpy as np,ml_dtypes
sys.path.insert(0,'/workspace/ready/backward')
from tokamax._src.ops.kda.pallas_mosaic_tpu_compact_forward_test import assert_reference
r=pathlib.Path('/workspace/ready');a=np.load(r/'features/backward-compact-0/compact_packed-full.npz');b=np.load(r/'backward-oracle/independent.npz');report=json.loads((r/'features/backward-compact-0/report.json').read_text());types=report['cases']['compact_packed']['modes']['full']['array_dtypes']
for i,key in enumerate(a.files):
 x,y=a[key],b[str(i)]
 if types[i]=='bfloat16':x,y=x.view(ml_dtypes.bfloat16),y.view(ml_dtypes.bfloat16)
 assert_reference(x,y,relative_rms_limit=.05 if i>=6 else .02)
 print('PASS',i,flush=True)
p=r/'backward-oracle/comparison.json';j=json.loads(p.read_text());j['candidate_acceptance']=dict(passed=True,threshold_source='unchanged assert_reference from pallas_mosaic_tpu_compact_forward_test.py and original compact-backward 5% parameter budget',relative_rms_nonparameter=.02,relative_rms_gate_parameters=.05,elementwise_rtol=.05,elementwise_atol=1e-4,zero_reference_atol=1e-7);p.write_text(json.dumps(j,indent=2));print('ALL_ORIGINAL_BOUNDS_PASS',flush=True)
