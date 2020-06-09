#!/bin/bash
#SBATCH --job-name=Job012_goli
#SBATCH --output=/exports/lkeb-hpc/gkarami/Code/Jobs/5_output_jointly.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --nodelist=res-hpc-gpu01
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/lkeb-hpc/gkarami/Program/cuda/cuda/lib64/
source /exports/lkeb-hpc/gkarami/Program/TF112/bin/activate
echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo
python /exports/lkeb-hpc/gkarami/Code/age_gender_classification-master/joint/jointly.py --where_to_run Cluster


#comment:
#partition=gpu:
            #nodelist=res-hpc-gpu01 , res-hpc-gpu02
#partition=LKEBgpu:
            #nodelist=  res-hpc-lkeb02, res-hpc-lkeb03, res-hpc-lkeb04, res-hpc-lkeb05