#!/usr/bin/env python
"""兼容入口；固定管线使用 ``python -m syncopate.pipeline.behavior_think_probe``。"""
from syncopate.pipeline.behavior_think_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
