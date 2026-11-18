"""Run independent ALFWorld task shards on separate GPUs, then merge reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

from alfworld_env import ROOT, project_path
from agent import validate_agent_config
from evaluate import evaluate, write_json
from report import EVAL_SPLITS, build_report, generate_report
from tqdm.auto import tqdm

SPLIT_CONFIG = {'seen': 'eval_in_distribution', 'unseen': 'eval_out_of_distribution'}


def partition_sizes(total, count):
    return [len(range(rank, total, count)) for rank in range(count)]


def worker(config_path):
    from model import ModelCaller
    job = json.loads(config_path.read_text())
    root = Path(job['output'])
    status_path = root / 'status.json'
    status = dict(completed=0, successes=0, stage='loading model', gpu=job['gpu'],
                  total=job['total'], turn=0, actions=0)
    write_json(status_path, status)
    try:
        model = ModelCaller(job['config'])
        completed = successes = 0
        for selection in job['selections']:
            if job['sizes'][selection] == 0:
                continue
            config = dict(job['config'], split=SPLIT_CONFIG[selection])
            status.update(split=selection, stage='initializing environment')
            write_json(status_path, status)

            def update(**fields):
                if 'completed' in fields:
                    fields['completed'] += completed
                if 'successes' in fields:
                    fields['successes'] += successes
                status.update(fields)
                write_json(status_path, status)

            results = evaluate(config, model, root / EVAL_SPLITS[selection], progress_callback=update)
            completed += len(results)
            successes += sum(r['success'] for r in results)
        if completed != job['total']:
            raise RuntimeError(f"Expected {job['total']} tasks, completed {completed}")
        status.update(completed=completed, successes=successes, stage='done')
        write_json(status_path, status)
    except BaseException as error:
        status.update(stage='error', error=f'{type(error).__name__}: {error}')
        write_json(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/alfworld.yaml')
    parser.add_argument('--gpus', default='0,1,2', help='Distinct indices among GPUs visible to this process')
    parser.add_argument('--split', choices=['seen', 'unseen', 'both'], default='both')
    parser.add_argument('--batch-size', type=int, default=4, help='Concurrent tasks per GPU (default: 4)')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--worker-config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config)
        return
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    import yaml
    import torch
    import matplotlib  # Validate plotting dependency before starting workers.

    try:
        gpus = [int(value) for value in args.gpus.split(',')]
    except ValueError:
        parser.error('--gpus must be comma-separated GPU indices, e.g. 0,1,2')
    if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0:
        parser.error('--gpus must contain distinct nonnegative indices')
    visible_count = torch.cuda.device_count()
    if max(gpus) >= visible_count:
        parser.error(f'Requested GPUs {gpus}, but only {visible_count} CUDA GPU(s) are visible. '
                     'Allocate/expose the requested GPUs first; no evaluation was started.')
    config = yaml.safe_load(project_path(args.config).read_text())
    validate_agent_config(config)
    if (config.get('inference_backend') == 'vllm'
            and config.get('vllm', {}).get('tensor_parallel_size', 1) != 1):
        parser.error('This launcher assigns one GPU per worker; use tensor_parallel_size=1, '
                     'or run evaluate.py directly with tensor parallelism')
    config.update(start_case=0, end_case=None, task_types=[1, 2, 3, 4, 5, 6],
                  no_progress=True, device='cuda:0', shard_count=len(gpus), batch_size=args.batch_size)
    selections = list(EVAL_SPLITS) if args.split == 'both' else [args.split]
    output = project_path(config.get('output_dir', 'outputs')) / datetime.now().strftime('%Y%m%d-%H%M%S-%f-parallel')
    output.mkdir(parents=True, exist_ok=False)
    inventory = build_report(output, project_path(config.get('data_dir', 'data/alfworld')))
    sizes = {s: partition_sizes(inventory['splits'][EVAL_SPLITS[s]]['available_cases'], len(gpus))
             for s in selections}
    write_json(output / 'config.json', dict(config, requested_split=args.split, gpus=gpus))
    inherited_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    device_tokens = inherited_devices.split(',') if inherited_devices else None
    processes, logs, bars, jobs = [], [], [], []
    interrupted = False
    print(f'Results: {output}', flush=True)
    try:
        for rank, gpu in enumerate(gpus):
            root = output / f'worker_{rank}'
            root.mkdir()
            job = dict(config=dict(config, shard_rank=rank), output=str(root), gpu=gpu,
                       selections=selections, sizes={s: sizes[s][rank] for s in selections},
                       total=sum(sizes[s][rank] for s in selections))
            jobs.append(job)
            path = root / 'worker_config.json'
            write_json(path, job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device_tokens[gpu].strip() if device_tokens else str(gpu),
                       PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
            log = (root / 'worker.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                                '--worker-config', str(path)],
                                               env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
            bars.append(tqdm(total=job['total'], desc=f'GPU {gpu}', unit='task', position=rank,
                             dynamic_ncols=True, disable=args.no_progress))
        while True:
            for job, bar in zip(jobs, bars):
                path = Path(job['output']) / 'status.json'
                if path.exists():
                    status = json.loads(path.read_text())
                    bar.update(status['completed'] - bar.n)
                    bar.set_postfix(batch=status.get('batch', 0), split=status.get('split', '-'), turn=status.get('turn', 0),
                                    actions=status.get('actions', 0), success=status['successes'],
                                    stage=status['stage'])
            failed = [p for p in processes if p.poll() not in (None, 0)]
            if failed or all(p.poll() is not None for p in processes):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Each child has its own process group, including TextWorld subprocesses.
        import signal
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for log in logs:
            log.close()
        for bar in bars:
            bar.close()
        print('Generating combined charts...', flush=True)
        report = generate_report(output, project_path(config.get('data_dir', 'data/alfworld')))
        write_json(output / 'summary.json', report)
    failed = [rank for rank, p in enumerate(processes) if p.returncode != 0]
    if interrupted or failed:
        raise SystemExit(f'Evaluation incomplete. Worker errors/interruption: {failed}. '
                         f'Inspect {output}/worker_*/worker.log; partial charts are saved.')
    completed = sum(v['completed_cases'] for v in report['splits'].values())
    expected = sum(job['total'] for job in jobs)
    if completed != expected:
        raise SystemExit(f'Incomplete result coverage: {completed}/{expected}')
    print(f'Completed {completed} tasks. Charts: {output / "charts"}')


if __name__ == '__main__':
    main()
