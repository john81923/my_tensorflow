import sys
import argparse

def str2bool(v):
     return v.lower() in ("yes","true","t","1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str2bool, default=False )
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    print args.train
