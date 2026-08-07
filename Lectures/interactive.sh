#!/bin/bash
srun \
  --job-name=example \
  --ntasks=1 \
  --time=1:00:00 \
  --mem-per-cpu=10000 \
  --partition=gpu \
  --gres=gpu:1 \
  --pty \
  julia --project
