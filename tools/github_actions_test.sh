#!/bin/bash -ef

USE_DIRS="mne_incubator/"
echo 'pytest --tb=short --cov=mne_incubator --cov-report xml -vv ${USE_DIRS}'
pytest --tb=short --cov=mne_incubator --cov-report xml -vv ${USE_DIRS}
