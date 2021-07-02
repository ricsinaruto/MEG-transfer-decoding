import os
from training import main
from args import Args

os.environ["NVIDIA_VISIBLE_DEVICES"] = Args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = Args.gpu

main(Args)
