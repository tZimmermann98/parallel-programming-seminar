#!/bin/bash

#SBATCH --exclusive
#SBATCH --partition=gpu2080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=parallel_runtime_comparison
#SBATCH --output=/scratch/tmp/t_zimm11/gpu2080node1.out
#SBATCH --error=/scratch/tmp/t_zimm11/gpu2080node1.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t_zimm11@uni-muenster.de
#SBATCH --mem=0

partition=gpu2080
implementation="parallel"
dirname=$(date +"%Y-%m-%dT%H-%M-%S-${partition}-${implementation}")

module purge
ml palma/2022a
ml CUDA/11.7.0
ml foss/2022a
ml UCX-CUDA/1.12.1-CUDA-11.7.0
ml CMake/3.23.1

cd /home/t/t_zimm11/

path=/scratch/tmp/t_zimm11/${dirname}
mkdir -p "$path"

buildname=build-${partition}

(
cd $implementation
rm -rf "$buildname"
mkdir "$buildname"
cd "$buildname"
cmake ..
cmake --build .
)

echo "size;numThreads;map_copy_device;map_kernel;map_copy_host;map_total;reduce_copy_device;reduce_kernel;reduce_copy_host;reduce_total;zip_copy_device;zip_kernel;zip_copy_host;zip_total" > "${path}/ppseminar_parallel.out"

for size in 1000000 10000000 10000000; do
    ./${implementation}/${buildname}/main $size >> "${path}/ppseminar_parallel.out"
done

exit 0