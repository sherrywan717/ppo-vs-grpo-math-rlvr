#!/usr/bin/env python3
"""Validate the frozen GRPO-v2 contract without model or CUDA imports."""

import json
from pathlib import Path

from math_rlvr.grpo_v2_contract import validate_contract_tree

if __name__ == "__main__":
    print(json.dumps(validate_contract_tree(Path.cwd()), indent=2, sort_keys=True))
