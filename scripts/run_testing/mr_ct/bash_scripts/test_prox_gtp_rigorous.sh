#!/usr/bin/env bash

python scripts/run_testing/mr_ct/prox_gtp_rigorous/test_sv_ct_prox_gtp_rigorous.py --device cuda:1
python scripts/run_testing/mr_ct/prox_gtp_rigorous/test_la_ct_prox_gtp_rigorous.py --device cuda:1