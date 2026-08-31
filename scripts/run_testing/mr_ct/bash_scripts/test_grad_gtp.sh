#!/usr/bin/env bash

python scripts/run_testing/mr_ct/grad_gtp/test_sv_ct_grad_gtp.py --device cuda:3
python scripts/run_testing/mr_ct/grad_gtp/test_la_ct_grad_gtp.py --device cuda:3