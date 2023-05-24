#!/bin/bash

#SBATCH --exclusive
#SBATCH --partition=gpu2080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --time=2:00:00
#SBATCH --job-name=sequential_runtime_comparison
#SBATCH --output=/scratch/tmp/t_zimm11/gpu2080node1_sequential.out
#SBATCH --error=/scratch/tmp/t_zimm11/gpu2080node1_sequential.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t_zimm11@uni-muenster.de
#SBATCH --mem=0

partition=normal
implementation="sequential"
dirname=$(date +"%Y-%m-%dT%H-%M-%S-${partition}-${implementation}")

module purge
ml foss/2022a
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

for (( run=1; run<=50; run++ )) do
    echo "size,map_duration,reduce_duration,zip_duration" > "${path}/ppseminar_sequential_${run}.out"
    for (( size=10000000; size<=100000000; size+=10000000 )) do
        ./${implementation}/${buildname}/main $size >> "${path}/ppseminar_sequential_${run}.out"
    done
done

exit 0