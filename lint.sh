#!/bin/bash

flake8 . --count --max-complexity=10 --max-line-length=90 \
	--per-file-ignores="__init__.py:F401,benchmark_aamutils.py:E402,benchmark_gm.py:E402" \
	--exclude venv \
	--statistics