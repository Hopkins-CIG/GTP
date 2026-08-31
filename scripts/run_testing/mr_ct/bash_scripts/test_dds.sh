#!/usr/bin/env bash

python scripts/run_testing/mr_ct/dds/test_sv_ct_dds.py --device cuda:2
python scripts/run_testing/mr_ct/dds/test_la_ct_dds.py --device cuda:2