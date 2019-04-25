#!/bin/bash

source venv/bin/activate

# List of dimensions to run
dim_list=(1 2 3 10 100 1000)

# Data Generation Section
n_samples=1000
n_batches=10

python gen-samples.py -n ${n_samples} -b ${n_batches} -d ${dim_list[@]}


# Training Section
n_epochs=1000

for dim in ${dim_list[@]}; do
    echo "Running Dimension ", $dim
    python train.py -d ${dim} -n ${n_epochs}
done
