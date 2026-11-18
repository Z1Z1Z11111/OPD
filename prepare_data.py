"""Prepare ALFWorld data without overwriting differing files (adapted from PoS)."""

import argparse
import filecmp
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.request
import zipfile

from alfworld_env import ROOT, SPLITS, project_path

ARCHIVES = (
    "0.2.2/json_2.1.1_json.zip",
    "0.2.2/json_2.1.1_pddl.zip",
    "0.4.0/json_2.1.2_tw-pddl.zip",
)


def verify_data(root, split, expected_count=None):
    folder = root / "json_2.1.1" / SPLITS[split]
    games = sorted(folder.rglob("game.tw-pddl"))
    if not games:
        raise ValueError(f"No game.tw-pddl files under {folder}")
    if expected_count is not None and len(games) != expected_count:
        raise ValueError(f"Expected {expected_count} games, found {len(games)}")
    for name in ("alfred.pddl", "alfred.twl2"):
        path = root / "logic" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty logic file: {path}")
    for game in games:
        for name in ("game.tw-pddl", "traj_data.json", "initial_state.pddl"):
            path = game.parent / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty task file: {path}")
    return len(games)


def copy_data(source, target):
    """Check all conflicts before copying; refuse symlinks or changed files."""
    source, target = source.resolve(), target.resolve()
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No files in {source}")
    for path in files:
        destination = target / path.relative_to(source)
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f"Unsafe source: {path}")
        if destination.is_symlink() or not destination.resolve().is_relative_to(target):
            raise ValueError(f"Unsafe destination: {destination}")
        if destination.exists() and (
            not destination.is_file() or not filecmp.cmp(path, destination, shallow=False)
        ):
            raise ValueError(f"Refusing to overwrite different data: {destination}")
    for path in files:
        destination = target / path.relative_to(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


def extract_split(archive, destination, split):
    """Only extract the requested split, rejecting traversal and symlink entries."""
    prefix = f"json_2.1.1/{SPLITS[split]}/"
    with zipfile.ZipFile(archive) as data:
        for entry in data.infolist():
            relative = Path(entry.filename)
            if (relative.is_absolute() or relative.drive or ".." in relative.parts
                    or "\\" in entry.filename or ":" in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f"Unsafe archive entry: {entry.filename}")
            if entry.is_dir() or not entry.filename.startswith(prefix):
                continue
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive target: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with data.open(entry) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=ROOT / "data/alfworld")
    parser.add_argument("--source", type=Path, help="Existing data root containing json_2.1.1/ and logic/")
    parser.add_argument("--split", choices=SPLITS, default="eval_out_of_distribution")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-count", type=int, help="Defaults to 134 for valid-unseen")
    args = parser.parse_args()
    target = project_path(args.target)
    count = args.expected_count
    if count is None and args.split == "eval_out_of_distribution":
        count = 134
    if args.verify_only:
        print(f"Verified {verify_data(target, args.split, count)} games in {target}")
        return
    with tempfile.TemporaryDirectory(prefix="alfworld-") as temporary:
        work = Path(temporary)
        source = args.source.resolve() if args.source else work / "source"
        if args.source is None:
            # Fail before downloading if the environment dependency is missing.
            from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH

            for suffix in ARCHIVES:
                archive = work / Path(suffix).name
                url = "https://github.com/alfworld/alfworld/releases/download/" + suffix
                print(f"Downloading {url}", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "ALFWorld-data-preparation"})
                with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
                    shutil.copyfileobj(response, out)
                extract_split(archive, source, args.split)
            (source / "logic").mkdir(parents=True)
            shutil.copyfile(ALFRED_PDDL_PATH, source / "logic/alfred.pddl")
            shutil.copyfile(ALFRED_TWL2_PATH, source / "logic/alfred.twl2")
        verify_data(source, args.split, count)
        # Stage only this split and the logic, and preflight all copy conflicts.
        stage = work / "stage"
        shutil.copytree(source / "json_2.1.1" / SPLITS[args.split],
                        stage / "json_2.1.1" / SPLITS[args.split], symlinks=True)
        shutil.copytree(source / "logic", stage / "logic", symlinks=True)
        copy_data(stage, target)
    print(f"Prepared {verify_data(target, args.split, count)} games in {target}")


if __name__ == "__main__":
    main()
