#!/bin/bash

source venv/bin/activate

# For timing purposes
start=`date +%s`

# List of dimensions to run
dim_list=(2 10 100 1000)

# List of latent dimensions to run
ldim_list=(1 10 100 1000)

# Training Section
n_epochs=100

# Latent Parameters
sig=1.0

python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim2.h5 -n ${n_epochs} -d 1 -s ${sig} -g 2 &
P1=$!
python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim10.h5 -n ${n_epochs} -d 1 -s ${sig} -g 3 &
P2=$!
wait $P1 $P2

python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim100.h5 -n ${n_epochs} -d 1 -s ${sig} -g 1 &
P1=$!
python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim1000.h5 -n ${n_epochs} -d 1 -s ${sig} -g 2 &
P2=$!
wait $P1 $P2


for dim in ${dim_list[@]}; do
    echo "Running latent dimensions 10, 100, 1000 over data dimension" ${dim}
    python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -d 10 -s ${sig} -g 1 &
    P1=$!
    python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -d 100 -s ${sig} -g 2 &
    P2=$!
    python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -d 1000 -s ${sig} -g 3 &
    P3=$!
    wait $P1 $P2 $P3
done

end=`date +%s`

runtime=$((end-start))
echo "Runtime:" ${runtime}
