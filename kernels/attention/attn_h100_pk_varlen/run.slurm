#!/bin/bash
#SBATCH --job-name=tk_mha_h100
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:10:00

source /etc/profile.d/modules.sh
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate megatron
module load cuda/12.8 gcc/13.3.0 cmake/4.2.0 nccl/2.27_cuda12.8 cudnn/9.6.0.74_cuda12

export CUDA_HOME=/data/apps/cuda/12.8
export CUDNN_HOME=/data/apps/cudnn/9.6.0.74_cuda12
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDNN_HOME/include:$CPLUS_INCLUDE_PATH
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128

cd /data/run01/scyb683/ThunderKittens/kernels/attention/attn_h100_pk_varlen

# make clean && make && 
python benchmark.py --qo_head 32 --kv_head 8 --d 128 --num_seqs 16 --causal
python benchmark.py --qo_head 32 --kv_head 8 --d 128 --num_seqs 16 --no_causal
