#!/usr/bin/env bash

# GDAL performance optimization
export GDAL_CACHEMAX=1024

# Conda-forge channel priority
conda config --add channels conda-forge
conda config --set channel_priority strict

# Enable CPL logging for GDAL reproducibility
export CPL_LOG=/tmp/gdal_log.txt