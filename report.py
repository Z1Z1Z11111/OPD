"""Summarize one evaluation run and plot per-task success rates."""
import argparse
import json
from pathlib import Path

from alfworld_env import ROOT, project_path

TASK_TYPES = {
    'pick_and_place_simple': 'Pick & place',
    'look_at_obj_in_light': 'Examine in light',
    'pick_clean_then_place_in_recep': 'Clean & place',
    'pick_heat_then_place_in_recep': 'Heat & place',
    'pick_cool_then_place_in_recep': 'Cool & place',
    'pick_two_obj_and_place': 'Pick two & place',
}
EVAL_SPLITS = {'seen': 'valid_seen', 'unseen': 'valid_unseen'}


def case_metadata(case_id):
    parts = Path(case_id).parts
    if len(parts) != 4 or parts[0] != 'json_2.1.1':
        raise ValueError(f'Unexpected case_id: {case_id}')
    task_type = parts[2].split('-', 1)[0]
    if task_type not in TASK_TYPES:
        raise ValueError(f'Unknown task type: {task_type}')
    return parts[1], task_type


def task_statistics(results):
    rows = {}
    for task_type in TASK_TYPES:
        cases = [r for r in results if case_metadata(r['case_id'])[1] == task_type]
        successes = sum(bool(r['success']) for r in cases)
        rows[task_type] = dict(completed_cases=len(cases), successes=successes,
                               success_rate=successes / len(cases) if cases else None)
    return rows


def build_report(run_dir, data_dir):
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Run directory not found: {run_dir}')
    inventory = {}
    for split in EVAL_SPLITS.values():
        if not (data_dir / 'json_2.1.1' / split).is_dir():
            raise FileNotFoundError(f'Dataset split not found: {split} in {data_dir}')
        inventory[split] = {}
        for game in (data_dir / 'json_2.1.1' / split).glob('*/*/game.tw-pddl'):
            if 'movable' in str(game) or 'Sliced' in str(game):
                continue
            path = game.parent / 'traj_data.json'
            if not path.is_file() or not json.loads(game.read_text()).get('solvable', False):
                continue
            metadata = json.loads(path.read_text())
            if metadata['task_type'] not in TASK_TYPES:
                continue
            case_id = str(path.parent.relative_to(data_dir))
            _, task_type = case_metadata(case_id)
            if task_type != metadata['task_type']:
                raise ValueError(f'Task type metadata mismatch: {path}')
            if not (path.parent / 'initial_state.pddl').is_file():
                raise ValueError(f'Missing initial_state.pddl: {path.parent}')
            inventory[split][case_id] = task_type
    results = []
    seen_ids = set()
    for path in sorted(run_dir.glob('**/result.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        split, _ = case_metadata(result['case_id'])
        if split not in inventory:
            continue
        if result['case_id'] not in inventory[split]:
            raise ValueError(f'Result does not match local dataset: {path}')
        if result['case_id'] in seen_ids:
            raise ValueError('Duplicate task results; select a single evaluation run, not all outputs')
        if not isinstance(result['success'], bool):
            raise ValueError(f'success must be boolean: {path}')
        seen_ids.add(result['case_id'])
        results.append(result)
    report = {'run_dir': str(run_dir.resolve()),
              'rate_denominator': 'completed cases only; untested categories have null rate',
              'error_files': len(list(run_dir.glob('**/error.json'))), 'splits': {}}
    for split in EVAL_SPLITS.values():
        cases = [r for r in results if case_metadata(r['case_id'])[0] == split]
        rows = task_statistics(cases)
        for task_type, row in rows.items():
            row['available_cases'] = sum(t == task_type for t in inventory[split].values())
        successes = sum(r['success'] for r in cases)
        report['splits'][split] = dict(
            available_cases=len(inventory[split]), completed_cases=len(cases),
            successes=successes, success_rate=successes / len(cases) if cases else None,
            by_task_type=rows)
    return report


def plot_report(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    colors = ['#2878b5', '#ef8a32']
    keys = list(TASK_TYPES)
    labels = [TASK_TYPES[k].replace(' & ', ' &\n') for k in keys]

    def bars(ax, split, offset, width, color):
        rows = report['splits'][split]['by_task_type']
        for i, key in enumerate(keys):
            row = rows[key]
            rate = row['success_rate']
            x = i + offset
            if rate is None:
                ax.text(x, 2, 'N/A', ha='center', fontsize=9, color='#666666')
            else:
                ax.bar(x, rate * 100, width, color=color,
                       label=split if i == 0 else None)
                ax.text(x, rate * 100 + 2,
                        f"{rate:.0%}\n{row['successes']}/{row['completed_cases']}",
                        ha='center', fontsize=9)

    def finish(fig, ax, name, title):
        ax.set(title=title, ylabel='Success rate (%)', ylim=(0, 118))
        ax.set_xlim(-.65, len(keys) - .35)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(range(len(keys)), labels)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=.2)
        fig.text(.5, .02, 'Labels: successes/completed. N/A = not evaluated. Rates exclude untested cases.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, 1))
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'{name}.{suffix}', dpi=170)
        plt.close(fig)

    for (short, split), color in zip(EVAL_SPLITS.items(), colors):
        info = report['splits'][split]
        fig, ax = plt.subplots(figsize=(11, 5))
        bars(ax, split, 0, .6, color)
        finish(fig, ax, f'{short}_success_rates',
               f"{split}: {info['completed_cases']}/{info['available_cases']} tasks evaluated")
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for split, offset, color in zip(EVAL_SPLITS.values(), (-.2, .2), colors):
        bars(ax, split, offset, .36, color)
    ax.legend(handles=[Patch(color=c, label=s) for s, c in zip(EVAL_SPLITS.values(), colors)],
              loc='upper right')
    coverage = ' | '.join(f"{s}: {v['completed_cases']}/{v['available_cases']} evaluated"
                          for s, v in report['splits'].items())
    finish(fig, ax, 'seen_unseen_comparison', 'Success rate by task type\n' + coverage)


def generate_report(run_dir, data_dir, output=None):
    output = output or run_dir / 'charts'
    report = build_report(run_dir, data_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'task_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot_report(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='One run directory; do not combine runs')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/alfworld')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = generate_report(project_path(args.run_dir), project_path(args.data_dir),
                             project_path(args.output) if args.output else None)
    for split, row in report['splits'].items():
        print(f"{split}: {row['completed_cases']}/{row['available_cases']} evaluated, {row['successes']} successes")


if __name__ == '__main__':
    main()
