#!/usr/bin/env bash

python scripts/run_testing/mr_ct/dds_gtp/test_sv_ct_dds_gtp.py --device cuda:2
python scripts/run_testing/mr_ct/dds_gtp/test_la_ct_dds_gtp.py --device cuda:2