import os
import argparse
import subprocess
from datetime import datetime

QUAST = os.path.abspath('vendor/quast-5.0.2/quast.py')
DNADIFF = os.path.abspath('vendor/MUMmer3.23/dnadiff')
# REFERENCE = os.path.abspath('data/arabidopsis.fna')
# ASSEMBLY_1 = os.path.abspath('data/initial_assembly.fasta')
# ASSEMBLY_2 = os.path.abspath('data/re_assembly.fasta')


def main():
    # time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
    quast_dir = os.path.abspath(output + '/quast_results/')
    dnadiff_dir1 = os.path.abspath(output + '/dnadiff_results_1/')
    dnadiff_dir2 = os.path.abspath(output + '/dnadiff_results_2/')
    subprocess.run(['mkdir', '-p', quast_dir, dnadiff_dir1, dnadiff_dir2])

    print('\nRuning QUAST\n')
    subprocess.run([QUAST, assembly_1, assembly_2, '-o', quast_dir, '-r', reference])

    print('\nRunning dnadiff the first time\n')
    subprocess.run([DNADIFF, '-p', str(dnadiff_dir1)+'/out', reference, assembly_1])

    print('\nRunning dnadiff the second time\n')
    subprocess.run([DNADIFF, '-p', str(dnadiff_dir2)+'/out', reference, assembly_2])

    with open(os.path.join(quast_dir, 'report.txt')) as f:
        lines = f.readlines()

    metrics = {}

    for line in lines:
        if 'N50' in line:
            _, assembly1, assembly2 = line.split()
            metrics['N50'] = (assembly1, assembly2)
        if 'NG50' in line:
            _, assembly1, assembly2 = line.split()
            metrics['NG50'] = (assembly1, assembly2)
        if 'NGA50' in line:
            _, assembly1, assembly2 = line.split()
            metrics['NGA50'] = (assembly1, assembly2)

    print(metrics)

    with open(os.path.join(dnadiff_dir1, 'out.report')) as f:
        lines = f.readlines()

    for line in lines:
        if '1-to-1' in line:
            _, ref, assembly = line.split()
            print(assembly)
        # other metrics, discuss with Megan and Mile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assembly_1', help='First assembly to compare')
    parser.add_argument('assembly_2', help='Second assembly to compare')
    parser.add_argument('reference', help='Reference to compare against')
    parser.add_argument('-o', '--output', help='Path to output directory')
    args = parser.parse_args()

    assembly_1 = os.path.abspath(args.assembly_1)
    assembly_2 = os.path.abspath(args.assembly_2)
    reference = os.path.abspath(args.reference)

    if args.output:
        output = os.path.abspath(args.output)
    else:
        output = os.path.join('test', datetime.now().strftime('%Y-%b-%d-%H-%M'))
        output = os.path.abspath(output)

    main()
