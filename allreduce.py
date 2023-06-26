import os, torch, argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.cuda.profiler as profiler



def saveData(data, rank, state):
    path = "./dump/"
    filename = "GPU" + str(rank) + "_" + state
    np.savetxt(path+filename+".txt", data.numpy())

def generateRandomTensor(M, mean, std):
    return torch.randn(M) * std + mean

def generateNumTensor(M, num):
    return torch.ones(M)

def runProcess(rank, args):
    torch.cuda.set_device(rank)

    #data = generateRandomTensor(args.M, args.mean, args.std)

    data = generateNumTensor(args.M, rank)

    print("Device {}. Doing AllReduce...".format(rank))
    
    data = data.cuda()

    profiler.start()

    for i in range(1000):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    
    dist.barrier()
    profiler.stop()

    print("Device {}. AllReduce done".format(rank))

    data = data.cpu()
    saveData(data, rank, args.name)


def init_process(rank, function, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "2950" 

    os.environ['NCCL_ALGO'] = 'Ring'
    #os.environ['NCCL_PROTO'] = 'LL'
    # os.environ['NCCL_MAX_NCHANNELS'] = "1"
    # os.environ['NCCL_MIN_NCHANNELS'] = "1"
    #os.environ['NCCL_CHECKS_DISABLE'] = '1'

    dist.init_process_group("nccl", rank=rank, world_size=args.gpus)
    function(rank, args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpus", default=4, type=int)
    parser.add_argument("--M", default=1000, type=int)
    parser.add_argument("--name", default="originalNCCL", type=str)
    parser.add_argument("--mean", default = 0.00000028992896587, type=float)
    parser.add_argument("--std", default = 0.000511206 , type=float)

    args = parser.parse_args()

    print()
    print(args.name)

    mp.spawn(init_process, nprocs = args.gpus, args=(runProcess, args)) 
    
