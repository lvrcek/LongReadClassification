import os
import json
import argparse
import subprocess
from datetime import datetime

import numpy as np


def create_classic_assembly(raven, reads, output):
    assembly = os.path.join(output, 'classic_assembly.fasta')
    subprocess.run(f'{raven} -t 12 {reads} > {assembly}', shell=True, cwd=output)
    return assembly


def create_pileogram_json(raven, reads, output):
    subprocess.run(f'{raven} -t 12 --split {reads}', shell=True, cwd=output)


def create_pileogram_signals(pile_dir, output):
    pile_json = os.path.join(output, 'uncontained.json')
    with open(pile_json) as f:
        pile_dict = json.load(f)

    for key, value in pile_dict.items():
        pile = os.path.join(pile_dir, key + '.npy')
        data = value['data_']
        np.save(pile, data)


def classify_pileograms(classifier, pile_json, output):
    subprocess.run(f'python {classifier} {pile_json}', shell=True, cwd='classifier')
    chimerics = os.path.abspath(os.path.join('classifier', '/chimerics.txt'))
    return chimerics


def create_ml_assembly(raven, reads, chimerics, output):
    assembly = os.path.join(output, 'ml_assembly.fasta')
    subprocess.run(f'{raven} -t 12 --notations {chimerics} --resume {reads} > {assembly}', shell=True, cwd=output)
    return assembly


def evaluate_assemblies(classic_assembly, ml_assembly, reference, output):
    # Idk, all the evaluation metrics seemed kinda shit
    # Need to see why quast is slow or just use Bandage instead
    # dnadiff seemed to work fine
    subprocess.run(f'python evaluate.py {classic_assembly} {ml_assembly} {reference} -o {output}', shell=True)


def main(raven, reads, classifier, classic_assembly, output):

    # Step 1: Create classic assembly if one is not already given
    print('\n1. Creating the classic assembly with Raven\n')
    if classic_assembly is None:
        classic_assembly = create_classic_assembly(raven, reads, output)
    
    # Step 2: Run Raven to create pile-o-gram JSON
    print('\n2. Creating pileograms\n')
    create_pileogram_json(raven, reads, output)

    # Step 3: Parse JSON file to obtain the signals
    print('\n3. Parsing JSON and saved the signals\n')
    pile_dir = 'pileograms'
    pile_dir = os.path.join(output, pile_dir)
    os.mkdir(pile_dir)
    create_pileogram_signals(pile_dir, output)

    # Step 4: Run the trained classifier on those pileograms, get labels
    print('\n4. Running the classifier\n')
    pile_json = os.path.join(output, 'uncontained.json')
    chimerics = classify_pileograms(classifier, pile_json, output)

    # Step 5: Run Raven again - obtain the ML assembly
    print('\n5. Creating the ML assembly\n')
    ml_assembly = create_ml_assembly(raven, reads, chimerics, output)

    # Step 6: Compare these two assemblies
    print('\n6. Evaluating the created assemblies\n')
    evaluate_assemblies(classic_assembly, ml_assembly, reference, output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('reads', help='Path to reads for creating the assembly')
    parser.add_argument('reference', help='Path to reference for evaluation')
    parser.add_argument('--raven', help='Path to Raven')
    parser.add_argument('--classifier', help='Path to pile-o-gram classifier')
    parser.add_argument('--assembly', help='Path to classic assembly previously obtained by Raven')
    parser.add_argument('-o', '--output', help='Path to output directory')
    args = parser.parse_args()

    reads = os.path.abspath(args.reads)
    reference = os.path.abspath(args.reference)

    if args.raven:
        raven = os.path.abspath(args.raven)
    else:
        raven = os.path.abspath('vendor/raven/build/bin/raven')

    if args.classifier:
        classifier = os.path.abspath(args.classifier)
    else:
        classifier = os.path.abspath('classifier/classifier.py')

    if args.assembly:
        assembly = os.path.abspath(args.assembly)
    else:
        assembly = None
    if args.output:
        output = os.path.abspath(args.output)
    else:
        output = os.path.join('results', datetime.now().strftime('%Y-%b-%d-%H-%M'))
        output = os.path.abspath(output)

    subprocess.run(['mkdir', output])
    main(raven, reads, classifier, assembly, output)
