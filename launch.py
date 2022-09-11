import os
os.environ["NVIDIA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from training import main
from args import Args


main(Args)
