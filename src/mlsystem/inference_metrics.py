"""CPU inference protocol v2: identical boundaries for both decoding paths."""
import hashlib
import json
import os
import platform
from pathlib import Path


def generation_metrics(ids, token_times, seconds, decoder_setup_seconds):
    n = len(token_times)
    span = token_times[-1] - token_times[0] if n > 1 else None
    intervals = [b-a for a,b in zip(token_times, token_times[1:])]
    return {
        'timing_protocol': 'inference-v2',
        'prompt_token_sha256': hashlib.sha256(json.dumps(ids,separators=(',', ':')).encode('utf-8')).hexdigest(),
        'decoder_setup_seconds': decoder_setup_seconds,
        'first_token_seconds': token_times[0] if n else None,
        'prefill_to_first_token_seconds': token_times[0] if n else None,
        'last_output_token_seconds': token_times[-1] if n else None,
        'decode_span_seconds': span,
        'decode_tokens_per_second': (n-1)/span if span is not None and span > 0 else None,
        'time_per_output_token_seconds': span/(n-1) if span is not None and span > 0 else None,
        'inter_token_seconds': intervals,
        'termination_tail_seconds': seconds-token_times[-1] if n else None,
        'token_times_seconds': token_times,
        'metric_scope': 'server generation phase includes decoder setup, prefill, decoding, sampling and in-loop callbacks; excludes tokenization, loading, retrieval and network',
    }


def runtime_environment(torch):
    cpu=platform.processor()
    if Path('/proc/cpuinfo').is_file():
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),cpu)
    return {'device':'cpu','dtype':'float32','cpu':cpu,'logical_cpus':os.cpu_count(),
            'affinity_cpus':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
            'python':platform.python_version(),'torch':str(torch.__version__),'platform':platform.platform(),
            'pid':os.getpid(),'omp_threads':os.environ.get('OMP_NUM_THREADS'),
            'mkl_threads':os.environ.get('MKL_NUM_THREADS'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}


def system_load():
    return list(os.getloadavg()) if hasattr(os,'getloadavg') else None
