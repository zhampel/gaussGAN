#!/bin/bash

source venv/bin/activate

# For timing purposes
start=`date +%s`

# List of dimensions to run
dim_list=(1 2 10 100 1000)

# List of distributions to run
dist_list=("gauss" "cauchy" "trunc_gauss")

# List of latent dimensions to run
ldim_list=(1 10 100 1000)

# Data Generation Section
n_samples=512
n_batches=100

#for dist in ${dist_list[@]}; do
#    echo ${dist}
#    python gen-samples.py -n ${n_samples} -s ${dist} -b ${n_batches} -d ${dim_list[@]}
#done


# Training Section
n_epochs=50

# Latent Parameters
sig=0.01

for ldim in ${ldim_list[@]}; do
    for dim in ${dim_list[@]}; do
        echo "Running latent dimension" $ldim "over data dimension" ${dim}
        python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -g 1 -d ${ldim} -s ${sig} &
        P1=$!
        python train.py -f /home/zhampel/gaussGAN/datasets/data_trunc_gauss_dim${dim}.h5 -n ${n_epochs} -d ${ldim} -s ${sig} -g 2 &
        P2=$!
        python train.py -f /home/zhampel/gaussGAN/datasets/data_cauchy_dim${dim}.h5 -n ${n_epochs} -g 1 -d ${ldim} -s ${sig} -g 3 &
        P3=$!
        wait $P1 $P2 $P3
    done
done

for ldim in ${ldim_list[@]}; do
    for dim in ${dim_list[@]}; do
        echo "Running latent dimension" $ldim "over data dimension" ${dim}
        python train.py -f /home/zhampel/gaussGAN/datasets/data_gauss_dim${dim}.h5 -n ${n_epochs} -d ${ldim} -s ${sig} -g 1 -w &
        P1=$!
        python train.py -f /home/zhampel/gaussGAN/datasets/data_trunc_gauss_dim${dim}.h5 -n ${n_epochs} -d ${ldim} -s ${sig} -g 2 -w &
        P2=$!
        python train.py -f /home/zhampel/gaussGAN/datasets/data_cauchy_dim${dim}.h5 -n ${n_epochs} -d ${ldim} -s ${sig} -g 3 -w &
        P3=$!
        wait $P1 $P2 $P3
    done
done

#for filename in /home/zhampel/gaussGAN/datasets/data_cauchy*.h5; do
#    for ldim in ${ldim_list[@]}; do
#        echo "Running Latent Dimension" $ldim "over" ${filename}
#        python train.py -f ${filename} -n ${n_epochs} -g 1 -d ${ldim} -s 0.01 -g 3
#    done
#done

end=`date +%s`

runtime=$((end-start))
echo "Runtime:" ${runtime}
