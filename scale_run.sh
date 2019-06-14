#!/bin/bash

source venv/bin/activate

# For timing purposes
start=`date +%s`

# List of dimensions to run
dim=50

# List of latent dimensions to run
#ldim_list=(1 10 100 1000)
ldim_list=(1 5 10 20 30 40 50 60 70 80 90 100 250 500 1000)

# Training Section
n_epochs=100

# Latent Parameters
sig=1.0

# Generator scaling parameter
dscale_list=(1 2 4 8 10 20 50 100)

echo "Data dimension:" ${dim}
#for ((idx=0; idx<3; ++idx)); do
for ((idx=0; idx<15; idx=$((idx+3)))); do
    
    echo "Running datasets with latent dimensions:" "${ldim_list[idx]}" "${ldim_list[idx+1]}" "${ldim_list[idx+2]}"

    for dscale in ${dscale_list[@]}; do
        echo "  Scale value:" ${dscale}
        python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -c ${dscale} -d ${ldim_list[idx]} -s ${sig} -g 1 -w &
        P1=$!
        python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -c ${dscale} -d ${ldim_list[idx+1]} -s ${sig} -g 2 -w &
        P2=$!
        python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -c ${dscale} -d ${ldim_list[idx+2]} -s ${sig} -g 3 -w &
        P3=$!
        wait $P1 $P2 $P3
    done
done


end=`date +%s`

runtime=$((end-start))
echo "Runtime:" ${runtime}
