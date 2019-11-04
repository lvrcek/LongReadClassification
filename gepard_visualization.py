import os
import re

GEPARD_CMD = 'java -cp /home/lovro/Software/gepard/Gepard-1.40.jar org.gepard.client.cmdline.CommandLine'
REFERENCES = '/home/lovro/Data/Zymo/subsampled/references/Genomes/'
READS = '/home/lovro/Data/Zymo/subsampled'
DOT_DIR = '/home/lovro/Data/Zymo/subsampled/dotplots'
PILE_DIR = '/home/lovro/Data/Zymo/subsampled/pileograms'

ref_dict = {
    'bs': 'Bacillus_subtilis_complete_genome.fasta',
    'cn': 'Cryptococcus_neoformans_draft_genome.fasta',
    'ec': 'Escherichia_coli_complete_genome.fasta',
    'ef': 'Enterococcus_faecalis_complete_genome.fasta',
    'lf': 'Lactobacillus_fermentum_complete_genome.fasta',
    'lm': 'Listeria_monocytogenes_complete_genome.fasta',
    'pa': 'Pseudomonas_aeruginosa_complete_genome.fasta',
    'sa': 'Staphylococcus_aureus_complete_genome.fasta',
    'sc': 'Saccharomyces_cerevisiae_draft_genome.fa',
    'se': 'Salmonella_enterica_complete_genome.fasta',
}

type_dict = {
    'RP': 'repeats',
    'CH': 'chimeric',
    'NM': 'normal',
}

wrong = '/home/lovro/Data/Zymo/subsampled/wrong.txt'


with open(wrong) as f:
    for line in f.readlines():
        line = line.strip()
        if 'flipped' in line:
            continue
        r = re.compile('.*/([a-z]{2})_([0-9]*)([A-Z]{2}_[A-Z]{2}).png')
        groups = re.match(r, line)
        species = groups.group(1)
        read_id = groups.group(2)
        mis_type = groups.group(3)
        # print(groups)
        # print(species, read_id, mis_type)
        reference = os.path.join(REFERENCES, ref_dict[species])
        pileogram = os.path.join(READS, type_dict[mis_type[:2]], species+'_'+read_id+'.png')
        read = os.path.join(READS, type_dict[mis_type[:2]]+'_reads', species+'_'+read_id+'.fasta')
        outfile = species + '_' + read_id + '_' + mis_type + '.png'
        command = GEPARD_CMD + ' -seq1 ' + reference + ' -seq2 ' + read + ' -matrix '
        command += '/home/lovro/Software/gepard/src/matrices/edna.mat -outfile ' + os.path.join(DOT_DIR, outfile)
        os.system(command)
        os.system('cp ' + pileogram + ' ' + os.path.join(PILE_DIR, outfile))