#!/bin/bash

partition=gpu2080
implementation="ppseminar"
dirname=$(date +"%Y-%m-%dT%H-%M-%S-${partition}-${implementation}")

module purge
ml palma/2022a
ml CUDA/11.7.0
ml foss/2022a
ml UCX-CUDA/1.12.1-CUDA-11.7.0
ml CMake/3.23.1

cd /home/t/t_zimm11/

path=/scratch/tmp/t_zimm11/ppseminar/${dirname}
mkdir -p "$path"

buildname=build-${partition}

(
cd ppseminar
rm -rf "$buildname"
mkdir "$buildname"
cd "$buildname"
cmake ..
make
)

for size in 10; do
    ./${buildname}/ppseminar $size >> "ppseminar.out"
done

exit 0