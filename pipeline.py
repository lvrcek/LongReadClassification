import os
import json
import argparse
import subprocess
from datetime import datetime

import numpy as np


def create_classic_assembly(raven, reads, output):
    assembly = os.path.join(output, 'classic_aassembly.fasta')
    subprocess.run(f'{raven} -t 12 {reads} > {assembly}', shell=True)
    return assembly


def create_pileogram_json(raven, reads):
    subprocess.run(f'{raven} -t 12 --split {reads}', shell=True)


def create_pileogram_signals(pile_dir):
    pile_json = 'uncontained.json'
    with open(pile_json) as f:
        pile_dict = json.load(f)

    for key, value in pile_dict.items():
        pile = os.path.join(pile_dir, key + '.npy')
        data = value['data_']
        np.save(pile, data)


def classify_pileograms(classifier, pile_dir):
    subprocess.run(f'python {classifier} {pile_dir}', shell=True)
    chimerics = os.path.abspath('chimerics.txt')  # Name for the file is hardcoded in Megan's code
    return chimerics


def create_ml_assembly(raven, reads, chimerics, output):
    assembly = os.path.join(output, 'ml_aassembly.fasta')
    subprocess.run(f'{raven} -t 12 --notations {chimerics} --resume {reads} > assembly', shell=True)
    return assembly


def evaluate_assemblies(classic_assembly, ml_assembly):
    # Idk, all the evaluation metrics seemed kinda shit
    # Need to see why quast is slow or just use Bandage instead
    # dnadiff seemed to work fine
    subprocess.run(f'python evaluate.py {classic_assembly} {ml_assembly}', shell=True)


def main(raven, reads, classifier, classic_assembly, output):

    # Step 1: Create classic assembly if one is not already given
    print('\n1. Creating the classic assembly with Raven\n')
    if classic_assembly is None:
        classic_assembly = create_classic_assembly(raven, reads, output)
    
    # Step 2: Run Raven to create pile-o-gram JSON
    print('\n2. Creating pileograms\n')
    create_pileogram_json(raven, reads)

    # Step 3: Parse JSON file to obtain the signals
    print('\n3. Parsing JSON and saved the signals\n')
    pile_dir = 'pileograms'
    pile_dir = os.path.join(output, pile_dir)
    os.mkdir(pile_dir)
    create_pileogram_signals(pile_dir)

    # Step 4: Run the trained classifier on those pileograms, get labels
    print('\n4. Running the classifier\n')
    chimerics = classify_pileograms(pile_dir)

    # Step 5: Run Raven again - obtain the ML assembly
    print('\n5. Creating the ML assembly\n')
    create_ml_assembly(raven, reads, chimerics, output)

    # Step 6: Compare these two assemblies
    print('\n6. Evaluating the created assemblies\n')
    evaluate_assemblies(classic_assembly, ml_assembly)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('raven', help='Path to Raven')
    parser.add_argument('reads', help='Path to reads for creating the assembly')
    parser.add_argument('classifier', help='Path to pile-o-gram classifier')
    parser.add_argument('--assembly', help='Path to classic assembly previously obtained by Raven')
    parser.add_argument('-o', '--output', help='Path to output directory')
    args = parser.parse_args()

    raven = os.path.abspath(args.raven)
    reads = os.path.abspath(args.reads)
    classifier = os.path.abspath(args.classifier)

    if args.assembly:
        assembly = os.path.abspath(args.assembly)
    else:
        assembly = None
    if args.output:
        output = os.path.abspath(args.output)
    else:
        output = os.path.abspath(datetime.now().strftime('%Y-%b-%d-%H-%M'))
        
    subprocess.run(['mkdir', output])
    main(raven, reads, classifier, assembly, output)
