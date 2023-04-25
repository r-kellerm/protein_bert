import argparse
from Bio import Entrez
import matplotlib.pyplot as plt
import pandas as pd
import re
from string import digits
from tqdm import tqdm
from urllib.error import HTTPError

Entrez.email = 'na@example.com'

tla2aa = {'Ala':'A',
    'Arg':'R',
    'Asn':'N',
    'Asp':'D',
    'Cys':'C',
    'Glu':'E',
    'Gln':'Q',
    'Gly':'G',
    'His':'H',
    'Ile':'I',
    'Leu':'L',
    'Lys':'K',
    'Met':'M',
    'Phe':'F',
    'Pro':'P',
    'Ser':'S',
    'Thr':'T',
    'Trp':'W',
    'Tyr':'Y',
    'Val':'V',
    'Ter':'X'}

def extract_nm_name(db_name):
    prot_name = db_name.split('.')[0]
    pos = prot_name.find('=')
    if pos != -1:
        prot_name = prot_name[pos+1:]
    return prot_name


def fetch_seq_from_ncbi(prot_name):
    try: 
        handle = Entrez.efetch(db='Protein', id=prot_name, rettype='gb', retmode='text')    
        text = handle.read()
    except HTTPError as e:
        text = e.read()
    
    prefix = 'ORIGIN' if prot_name.startswith('NP') else 'translation='
    seq_start = text.find(prefix) + len(prefix) + 1
    seq_lines = text[seq_start:].split('\n')
    seq_text = ''
    remove_digits = str.maketrans('', '', digits)
    for i in range(len(seq_lines)):
        res = seq_lines[i].translate(remove_digits)
        res = res.replace(' ','').upper()
        seq_text += res
    seq = re.match('[A-Z]*', seq_text)
    return seq[0]


def extract_mut(mut):
    mut_pos = -1
    mut_ref = mut_alt = '-1'
    re_mut = re.match('p\.[A-Z][0-9]*[A-Z]', mut)
    re_mut_tla = re.match('p\.[A-z]{3}[0-9]*[A-z]{3}', mut)
    mut = mut.split('.')[1]
    if re_mut:
        mut_ref = mut[0]
        mut_alt = mut[-1]
        mut_pos = int(mut[1:-1])
    elif re_mut_tla:
        mut_ref = tla2aa[mut[0:3]]
        mut_alt = tla2aa[mut[-3:]]
        mut_pos = int(mut[3:-3]) - 1
    return mut_ref, mut_alt, mut_pos


def add_mut_to_ref_seq(seq, mut):
    mut_seq = '-1'
    mut_ref, mut_alt, mut_pos = extract_mut(mut)
    if mut_pos != -1:
        mut_seq = seq[:mut_pos] + mut_alt + seq[mut_pos+1:]    
    return mut_seq, mut_pos


def main(args, draw_hist=True):
    db_path = args.db_path
    out_db_path = args.out_db_path
    prot_name_col = args.prot_name_col
    ref_seq_col = args.ref_seq_col
    mut_seq_col = args.mut_seq_col
    change_col = args.change_col
    checkpoint = args.checkpoint
    failed = []
    all_pos = []
    db = pd.read_csv(db_path)
    for i,row in enumerate(tqdm(db.iterrows(), total=len(db))):
        try:
            prot_name = extract_nm_name(row[1][prot_name_col])
            seq = fetch_seq_from_ncbi(prot_name)
            db.at[i, ref_seq_col] = seq
            mut_seq, mut_pos = add_mut_to_ref_seq(seq, row[1][change_col])
            all_pos.append(mut_pos)
            db.at[i, mut_seq_col] = mut_seq
            if i % checkpoint == 0:
                db.to_csv(out_db_path)
        except Exception as e:
            print(f'Failed at index {i}, protein name {prot_name}. Error: {e}')
            failed.append([i, prot_name])
    
    db.to_csv(out_db_path)

    if len(failed) > 0:
        lines = ['index,orig_prot_name\n']
        lines += [str(failed[i][0]) + ',' + failed[i][1] + '\n' for i in range(len(failed))]
        out_failures_path = 'failures.csv'
        with open(out_failures_path, 'w') as f:
            f.writelines(lines)
        print(f'Indexs and protein names of failures saved at {out_failures_path}')
        
    if draw_hist:
        plt.hist(all_pos, bins='auto')
        plt.savefig('mut_pos_hist.png')

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get protein sequences and save references and mutations')
    parser.add_argument('--db-path', required=True, type=str, help='CSV file contains protein names to fetch and mutations information')
    parser.add_argument('--out-db-path', type=str, default='mut_ref_db.csv')
    parser.add_argument('--prot-name-col', type=str, help='Name of the column holding the protein name (nm name for NCBI)', default='name')
    parser.add_argument('--change-col', type=str, help='Name of the column holding the mutation information', default='aa_change_')
    parser.add_argument('--ref-seq-col', type=str, help='Name of the columns to hold the reference (wild type) sequence', default='refSequence')
    parser.add_argument('--mut-seq-col', type=str, help='Name of the columns to hold the mutation sequence', default='mutSequence')
    parser.add_argument('--checkpoint', type=int, help='Output saving checkpoint', default=1000)

    args = parser.parse_args()
    main(args)