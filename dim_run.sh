#!/bin/bash

source venv/bin/activate

# For timing purposes
start=`date +%s`

# List of dimensions to run
#dim_list=(2 10 100 1000)
dim_list=(10 25 50 75 100 250 500 750 1000)

# List of latent dimensions to run
#ldim_list=(1 10 100 1000)
ldim_list=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)

# Training Section
n_epochs=100

# Latent Parameters
sig=1.0

# Generator scaling parameter
dscale=10

#for ((idx=0; idx<3; ++idx)); do
for ((idx=0; idx<9; idx=$((idx+3)))); do
    
    echo "Running datasets with dimensions:" "${dim_list[idx]}" "${dim_list[idx+1]}" "${dim_list[idx+2]}"

    for ldim in ${ldim_list[@]}; do
        echo "  Latent dimension:" ${ldim}
        python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim_list[idx]}.h5 -n ${n_epochs} -c ${dscale} -d ${ldim} -s ${sig} -g 1 -w &
        P1=$!
        python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim_list[idx+1]}.h5 -n ${n_epochs} -c ${dscale} -d ${ldim} -s ${sig} -g 2 -w &
        P2=$!
        python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim_list[idx+2]}.h5 -n ${n_epochs} -c ${dscale} -d ${ldim} -s ${sig} -g 3 -w &
        P3=$!
        wait $P1 $P2 $P3
    done
done



#python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim2.h5 -n ${n_epochs} -d 1 -s ${sig} -g 2 &
#P1=$!
#python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim10.h5 -n ${n_epochs} -d 1 -s ${sig} -g 3 &
#P2=$!
#wait $P1 $P2
#
#python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim100.h5 -n ${n_epochs} -d 1 -s ${sig} -g 1 &
#P1=$!
#python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim1000.h5 -n ${n_epochs} -d 1 -s ${sig} -g 2 &
#P2=$!
#wait $P1 $P2
#
#
#for dim in ${dim_list[@]}; do
#    echo "Running latent dimensions 10, 100, 1000 over data dimension" ${dim}
#    python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -d 10 -s ${sig} -g 1 &
#    P1=$!
#    python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -d 100 -s ${sig} -g 2 &
#    P2=$!
#    python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -d 1000 -s ${sig} -g 3 &
#    P3=$!
#    wait $P1 $P2 $P3
#done

end=`date +%s`

runtime=$((end-start))
echo "Runtime:" ${runtime}
