import pathlib,os,sys,time,json
r=pathlib.Path('/workspace')
while 'PYTEST_RESULT' not in (r/'final-driver.log').read_text():time.sleep(5)
assert 'PYTEST_RESULT '+os.environ['TPU_WORKER_ID']+' 0' in (r/'final-driver.log').read_text()
os.environ['JAX_PLATFORMS']='tpu,cpu'
os.environ['LIBTPU_INIT_ARGS']='--xla_tpu_impure_enable_large_2nd_minor_layout=true'
sys.path.insert(0,str(r/'compact-cp'))
import jax,numpy as np
rank=int(os.environ['TPU_WORKER_ID'])
jax.distributed.initialize(coordinator_address='kda-review-cp16-1116-worker-0-0.kda-review-cp16-1116:12363',num_processes=2,process_id=rank,initialization_timeout=600)
source=(r/'compact-cp/tokamax/_src/ops/kda/cp_compact_test.py').read_text()
needle='      results.append(jax.block_until_ready(fn(*inputs)))'
replacement="      exe = fn.lower(*inputs).compile()\n      resident = tuple(jax.device_put(x, f) for x, f in zip(inputs, exe.input_formats[0], strict=True))\n      jax.block_until_ready(resident)\n      for _ in range(10): jax.block_until_ready(exe(*resident))\n      samples = []\n      for _ in range(30):\n        start = time.perf_counter_ns()\n        result = jax.block_until_ready(exe(*resident))\n        samples.append((time.perf_counter_ns() - start) / 1e6)\n      folder = pathlib.Path('/workspace/benchmark-production-split') / route\n      folder.mkdir(parents=True, exist_ok=True)\n      for capture in range(2):\n        with jax.profiler.trace(str(folder / str(capture))):\n          for _ in range(5): jax.block_until_ready(exe(*resident))\n      memory = exe.memory_analysis()\n      (folder/'report.json').write_text(json.dumps(dict(cp=cp, heads=heads, local_tokens=local_tokens, route=route, process=jax.process_index(), median_ms=float(np.median(samples)), samples_ms=samples, compiler_temporary_bytes=int(memory.temp_size_in_bytes), input_formats=list(map(str,exe.input_formats[0]))),indent=2))\n      (folder/'module.hlo.txt').write_text(exe.as_text())\n      print('RESULT', jax.process_index(), route, float(np.median(samples)), memory.temp_size_in_bytes, flush=True)\n      results.append(result)"
assert needle in source
replacement += "\n      snapshot(\"production-split-route-\" + route)"
source=source.replace('lower_bound=-1.0,','lower_bound=-5.0,')
source=source.replace(needle,replacement)
from snapshot import snapshot
namespace=dict(time=time,pathlib=pathlib,json=json,rank=rank,snapshot=snapshot,__name__='cp_benchmark')
exec(compile(source,'instrumented_cp_test','exec'),namespace)
namespace['_run_compact_cp'](16,True,'rank',heads=64,local_tokens=16384)
print('COMPLETE',rank,flush=True)
