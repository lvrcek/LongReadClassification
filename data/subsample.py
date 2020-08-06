import random
import subprocess
import os
import argparse
from pathlib import Path


class Subsampler():
    
    def __init__(self, number, read_types):
        self.number = number
        self.types = read_types

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def run(self):
        if self.types == 'all':
            types = ['junk', 'repetitive', 'chimeric', 'regular']
        else:
            types = self.types.split()
        for t in types:
            src = os.path.abspath(t + '_all')
            dest = os.path.abspath(t)
            reads = os.listdir(src)
            random.shuffle(reads)
            for r in reads[:self.number]:
                file_path  = os.path.join(src, r)
                copy_files = subprocess.run(['cp', file_path, dest], check=True)
            print("Subsampled and copied:", t)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', required=True, type=int,
        help="number of reads to be copied")
    parser.add_argument('-t', '--type', default='all',
        help="type of reads to be subsampled")
    args = parser.parse_args()
    subsampler = Subsampler(args.number, args.type)
    
    with subsampler:
        random.seed(0)
        subsampler.run()