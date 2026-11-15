import os,time,pathlib,sys
r=pathlib.Path('/workspace')
while not (r/'READY').exists():time.sleep(5)
os.environ['JAX_PLATFORMS']='tpu,cpu'
os.environ['LIBTPU_INIT_ARGS']='--xla_tpu_impure_enable_large_2nd_minor_layout=true'
sys.path.insert(0,str(r/'compact-cp'))
import jax
rank=int(os.environ['TPU_WORKER_ID'])
jax.distributed.initialize(coordinator_address='kda-review-cp16-1116-worker-0-0.kda-review-cp16-1116:12361',num_processes=2,process_id=rank,initialization_timeout=600)
print('DISTRIBUTED',rank,jax.device_count(),jax.local_device_count(),flush=True)
import pytest
code=pytest.main(['-q','-x','-s','tokamax/_src/ops/kda/compact_gate_parameter_test.py'])
print('PYTEST_RESULT',rank,code,flush=True)
sys.exit(code)
