#!/usr/bin/env bash

python scripts/run_testing/mr_ct/fbp/test_sv_ct_fbp.py --device cuda:0
python scripts/run_testing/mr_ct/fbp/test_la_ct_fbp.py --device cuda:0