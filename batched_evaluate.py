"""Cooperatively run independent ReAct episodes with one batched GPU call."""
from pathlib import Path
from time import perf_counter

from agent import create_agent
from evaluate import Logger, write_json
from report import task_statistics
from tqdm.auto import tqdm


def evaluate_batched(config, model, output, env_factory, progress_callback=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    lanes, results = [], []
    progress = None
    keys = ('input_tokens', 'output_tokens', 'total_tokens')

    def notify(**fields):
        if progress_callback:
            progress_callback(completed=len(results), successes=sum(r['success'] for r in results), **fields)

    def start(lane):
        if lane['next'] >= len(lane['env']):
            lane['episode'] = None
            return
        index = lane['env'].case_indices[lane['next']]
        lane.update(index=index, path=output / 'cases' / f'case_{index:04d}',
                    started=perf_counter(), usage=dict.fromkeys(keys, 0), status={})
        lane['logger'] = Logger(lane['path'] / 'events.jsonl')
        def update(**fields):
            lane['status'].update(fields)
        lane['episode'] = create_agent(config, model, lane['logger'], progress=update).run_iter(lane['env'])
        advance(lane)

    def advance(lane, response=None):
        try:
            lane['messages'] = lane['episode'].send(response)
        except StopIteration as finished:
            result = finished.value
            result.update(case_seconds=round(perf_counter() - lane['started'], 3),
                          token_usage=lane['usage'].copy())
            write_json(lane['path'] / 'result.json', result)
            lane['logger'].close()
            lane['logger'] = None
            results.append(result)
            write_json(output / 'summary.json', dict(
                completed_cases=len(results), selected_cases=sum(len(l['env']) for l in lanes),
                successes=sum(r['success'] for r in results),
                success_rate=sum(r['success'] for r in results) / len(results),
                by_task_type=task_statistics(results), total_steps=sum(r['steps'] for r in results),
                token_usage={k: sum(r['token_usage'][k] for r in results) for k in keys},
                case_seconds=sum(r['case_seconds'] for r in results)))
            progress.update(1)
            lane['next'] += 1
            start(lane)

    try:
        # Construct and use TextWorld environments on the main thread.
        for rank in range(int(config['batch_size'])):
            env = env_factory(dict(config, lane_rank=rank, lane_count=int(config['batch_size'])))
            lanes.append(dict(env=env, next=0, episode=None, logger=None))
        total = sum(len(l['env']) for l in lanes)
        if not total:
            raise ValueError('No tasks selected; check data and case range')
        progress = tqdm(total=total, desc=config['split'], unit='task',
                        disable=config.get('no_progress', False), dynamic_ncols=True)
        for lane in lanes:
            start(lane)
        while active := [l for l in lanes if l['episode'] is not None]:
            notify(batch=len(active), stage='generating batch',
                   turn=max(l['status'].get('turn', 0) for l in active),
                   actions=max(l['status'].get('actions', 0) for l in active))
            responses = model.generate_batch([l['messages'] for l in active])
            if len(responses) != len(active):
                raise RuntimeError('Model returned a different number of batch responses')
            for lane, (response, usage) in zip(active, responses):
                for key in keys:
                    lane['usage'][key] += usage[key]
                advance(lane, response)
            notify(batch=sum(l['episode'] is not None for l in lanes), stage='batch finished')
            progress.set_postfix(success=sum(r['success'] for r in results), batch=len(active))
        return results
    except BaseException as error:
        for lane in lanes:
            if lane['episode'] is not None and 'path' in lane:
                write_json(lane['path'] / 'error.json', dict(case_index=lane['index'],
                           error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        for lane in lanes:
            try:
                if lane['episode'] is not None:
                    lane['episode'].close()
                if lane['logger'] is not None:
                    lane['logger'].close()
            finally:
                lane['env'].close()
        if progress is not None:
            progress.close()
