#!/usr/bin/env bash

python scripts/run_testing/mr_ct/daps/test_sv_ct_daps.py --device cuda:3
python scripts/run_testing/mr_ct/daps/test_la_ct_daps.py --device cuda:3