export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
python3 allreduce.py --gpus 4 --name originalNCCL --M 1048576
 
export LD_LIBRARY_PATH=/src/main/KimSum/build/lib/
python3 allreduce.py --gpus 4 --name modifiedNCCL --M 1048576



# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# nvprof --devices 1 --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/baseline/AR_G%p.csv python3  allreduce.py --gpus 4 --name baseline --M 262144 

# export LD_LIBRARY_PATH=/src/main/KimSum/build/lib/
# nvprof --devices 1 --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/modified/AR_G%p.csv python3  allreduce.py --gpus 4 --name modifiedNCCL --M 262144 


# 256KB = 65536
# 1MB = 262144
# 4MB = 1048576
# 25MB = 6553600
# 100MB = 26214400
# 1GB = 268435456
# 10GB = 2684354560