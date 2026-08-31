#!/usr/bin/env bash

python scripts/run_testing/mr_ct/dps/test_sv_ct_dps.py --device cuda:1
python scripts/run_testing/mr_ct/dps/test_la_ct_dps.py --device cuda:1