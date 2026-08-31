#!/usr/bin/env bash

python scripts/run_testing/mr_ct/tv/test_sv_ct_tv.py --device cuda:0
python scripts/run_testing/mr_ct/tv/test_la_ct_tv.py --device cuda:0