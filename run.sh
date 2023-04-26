#####################
# CORRECTNESS CHECK #
#                   #
#####################


# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping/build/lib/
# python3 allreduce.py --gpus 4 --name X --M 1024

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_XY/build/lib/
# python3 allreduce.py --gpus 4 --name XY --M 1024

#####################
# TIME PROFILING    #
#                   #
#                   #
#                   #
#####################

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/baseline/AR_G%p_100MB.csv python3  allreduce.py --gpus 4 --name baseline --M 26214400 

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping/build/lib/
nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/deviceDropping_X1/AR_G%p_100MB.csv python3  allreduce.py --gpus 4 --name deviceDropping_X1 --M 26214400 

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_XY/build/lib/
nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/deviceDropping_X1Y1/AR_G%p_100MB.csv python3  allreduce.py --gpus 4 --name deviceDropping_X1Y1 --M 26214400 

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/topk_99/AR_G%p_100MB.csv python3  topk.py --gpus 4 --name topk_99 --M 26214400 --top 0.01

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/topk_90/AR_G%p_100MB.csv python3  topk.py --gpus 4 --name topk_90 --M 26214400 --top 0.1

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/topk_75/AR_G%p_100MB.csv python3  topk.py --gpus 4 --name topk_75 --M 26214400 --top 0.25

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
nvprof --devices 0,1,2,3 --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/topk_50/AR_G%p_100MB.csv python3  topk.py --gpus 4 --name topk_50 --M 26214400 --top 0.5

# 256KB = 65536
# 1MB = 262144
# 4MB = 1048576
# 25MB = 6553600
# 32MB = 8388608
# 100MB = 26214400
# 500 MB = 131072000
# 1GB = 268435456
# 5GB = 1342177280
# 10GB = 2684354560
