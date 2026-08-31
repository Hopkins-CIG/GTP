#!/usr/bin/env bash

python scripts/run_testing/mr_ct/daps_gtp/test_sv_ct_daps_gtp.py --device cuda:0
python scripts/run_testing/mr_ct/daps_gtp/test_la_ct_daps_gtp.py --device cuda:0