"""Runtime and memory bookkeeping for a training run, written to results/<experiment_id>/resources.json."""
import json
import os
import platform
import resource
import time

import torch


def peak_host_memory_mb():
    # ru_maxrss is in kilobytes on Linux
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def peak_gpu_memory_mb(device):
    """Peak memory held by tensors and by PyTorch's caching allocator since the start of the run."""
    if device.type != 'cuda':
        return None, None
    return (round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1),
            round(torch.cuda.max_memory_reserved(device) / 2 ** 20, 1))


class ResourceLog:
    def __init__(self, results_dir):
        self.path = os.path.join(results_dir, 'resources.json')
        self.start = time.time()
        self.phase_start = self.start
        self.info = {
            'hostname': platform.node(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'seconds': {},
        }

    def end_phase(self, name):
        now = time.time()
        self.info['seconds'][name] = round(now - self.phase_start, 1)
        self.phase_start = now
        print(f'[resources] {name}: {self.info["seconds"][name]:.0f} s')

    def save(self, device, **extra):
        self.info['seconds']['total'] = round(time.time() - self.start, 1)
        self.info['peak_host_memory_mb'] = peak_host_memory_mb()
        self.info['peak_gpu_allocated_mb'], self.info['peak_gpu_reserved_mb'] = peak_gpu_memory_mb(device)
        self.info.update(extra)
        with open(self.path, 'w') as f:
            json.dump(self.info, f, indent=4)
        print('[resources]', json.dumps(self.info))
