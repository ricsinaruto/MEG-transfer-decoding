import os
os.environ["NVIDIA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from training import main
from args import Args


main(Args)
