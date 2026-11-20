"""Follow an OPD GRPO training log in the terminal (Ctrl+C to leave)."""
import argparse
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
COUNTS = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


def bar(done, total):
    filled = min(24, 24 * done // total)
    return f"[{'█' * filled}{'░' * (24 - filled)}] {done}/{total} ({100 * done / total:.1f}%)"


def follow(path, once):
    stage = "等待训练器"
    done = 0
    total = 0
    scanning = False
    scan = ""
    last = None
    finished = False
    with path.open(encoding="utf-8", errors="replace") as log:
        while True:
            for raw in log:
                line = ANSI.sub("", raw).strip()
                if line.startswith("Traceback (most recent call last):"):
                    stage = "训练报错，请查看日志末尾"
                    finished = True
                elif line.startswith("Evaluation checkpoint:"):
                    stage = "权重已导出"
                    finished = True
                elif finished:
                    continue
                elif "Initializing AlfredTWEnv" in line:
                    stage, scanning, scan = "扫描 ALFWorld 数据", True, ""
                elif scanning and "Overall we have" in line:
                    stage, scanning, scan = "初始化环境", False, ""
                elif scanning and (counts := COUNTS.search(line)):
                    scan = bar(*map(int, counts.groups()))
                elif match := re.search(r"Total training steps:\s*(\d+)", line):
                    total = int(match[1])
                    stage, scanning, scan = "初始化训练 worker", False, ""
                elif "test_gen_batch meta info:" in line:
                    stage = "验证中"
                elif "Training Progress:" in line and (counts := COUNTS.search(line)):
                    current, total = map(int, counts.groups())
                    done = max(done, current)
                    stage = "GRPO 训练"
                elif match := re.search(r"(?:^|\)\s*)step:(\d+)\s*[-\s]", line):
                    done = max(done, int(match[1]))
                    stage = "GRPO 训练"
            progress = bar(done, total) if total else f"{done}/? 步"
            message = f"训练 {progress}  |  {stage} {scan}".rstrip()
            if last != message:
                print(("\r\033[2K" if sys.stdout.isatty() else "") + message,
                      end="" if sys.stdout.isatty() else "\n", flush=True)
                last = message
            if once or finished:
                break
            time.sleep(2)
    if sys.stdout.isatty():
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, nargs="?", help="默认读取最近的启动日志")
    parser.add_argument("--once", action="store_true", help="只显示当前进度")
    args = parser.parse_args()
    path = args.log or max((ROOT / "outputs/grpo").glob("launch-*/train.log"),
                           key=lambda file: file.stat().st_mtime, default=None)
    if not path or not path.is_file():
        parser.error("找不到训练日志，请指定 train.log 路径")
    try:
        follow(path, args.once)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
