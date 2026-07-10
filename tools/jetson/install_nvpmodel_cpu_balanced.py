#!/usr/bin/env python3

from __future__ import annotations

import datetime
import re
import shutil
import sys
from pathlib import Path


CONFIG_LINK = Path("/etc/nvpmodel.conf")
CUSTOM_NAME = "25W_BALANCE"
CPU_MAX_KHZ = 1_497_600
GPU_MAX_HZ = 816_000_000


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


if not CONFIG_LINK.exists():
    fail(f"{CONFIG_LINK} does not exist")

config_path = CONFIG_LINK.resolve()

if not config_path.is_file():
    fail(f"Resolved config is not a regular file: {config_path}")

text = config_path.read_text(encoding="utf-8")

block_pattern = re.compile(
    r"(?ms)^< POWER_MODEL ID=(\d+) NAME=([^>]+) >\n"
    r".*?"
    r"(?=^< POWER_MODEL ID=|^< PM_CONFIG )"
)

blocks = list(block_pattern.finditer(text))

if not blocks:
    fail("No POWER_MODEL blocks found")

base_block = None
existing_custom = None
used_ids: set[int] = set()

for match in blocks:
    mode_id = int(match.group(1))
    mode_name = match.group(2).strip()

    used_ids.add(mode_id)

    if mode_name == "25W":
        base_block = match

    if mode_name == CUSTOM_NAME:
        existing_custom = match

if base_block is None:
    available = ", ".join(
        f"{match.group(1)}:{match.group(2).strip()}"
        for match in blocks
    )
    fail(f"25W mode not found. Available modes: {available}")

if existing_custom is not None:
    custom_id = int(existing_custom.group(1))
else:
    custom_id = max(used_ids) + 1

custom_block = base_block.group(0)

custom_block, header_count = re.subn(
    r"^< POWER_MODEL ID=\d+ NAME=[^>]+ >",
    f"< POWER_MODEL ID={custom_id} NAME={CUSTOM_NAME} >",
    custom_block,
    count=1,
)

custom_block, cpu_count = re.subn(
    r"(?m)^(CPU_A78_\d+\s+MAX_FREQ\s+)\d+\s*$",
    rf"\g<1>{CPU_MAX_KHZ}",
    custom_block,
)

custom_block, gpu_count = re.subn(
    r"(?m)^(GPU\s+MAX_FREQ\s+)\d+\s*$",
    rf"\g<1>{GPU_MAX_HZ}",
    custom_block,
)

if header_count != 1:
    fail("Could not replace POWER_MODEL header")

if cpu_count < 1:
    fail("No CPU_A78_* MAX_FREQ lines found in 25W block")

if gpu_count != 1:
    fail(
        "Expected exactly one GPU MAX_FREQ line, "
        f"found {gpu_count}"
    )

if existing_custom is not None:
    new_text = (
        text[: existing_custom.start()]
        + custom_block
        + text[existing_custom.end() :]
    )
    action = "updated"
else:
    pm_config = re.search(r"(?m)^< PM_CONFIG ", text)

    if pm_config is None:
        fail("PM_CONFIG section not found")

    prefix = text[: pm_config.start()].rstrip() + "\n\n"
    suffix = text[pm_config.start() :]

    new_text = prefix + custom_block.rstrip() + "\n\n" + suffix
    action = "created"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = config_path.with_name(
    f"{config_path.name}.backup_{timestamp}"
)

shutil.copy2(config_path, backup_path)
config_path.write_text(new_text, encoding="utf-8")

print(f"Config:  {config_path}")
print(f"Backup:  {backup_path}")
print(f"Profile: {CUSTOM_NAME}")
print(f"ID:      {custom_id}")
print(f"Action:  {action}")
print(f"CPU max: {CPU_MAX_KHZ} kHz")
print(f"GPU max: {GPU_MAX_HZ} Hz")
