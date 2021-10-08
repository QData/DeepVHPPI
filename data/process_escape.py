from Bio import SeqIO
from pdb import set_trace as stop

def load_baum2020():
    seq = SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq

    muts = [
        'K417E', 'K444Q', 'V445A', 'N450D', 'Y453F', 'L455F',
        'E484K', 'G485D', 'F486V', 'F490L', 'F490S', 'Q493K',
        'H655Y', 'R682Q', 'R685S', 'V687G', 'G769E', 'Q779K',
        'V1128A',
    ]

    seqs_escape = {}
    for mut in muts:
        aa_orig = mut[0]
        aa_mut = mut[-1]
        pos = int(mut[1:-1]) - 1
        assert(seq[pos] == aa_orig)
        escaped = seq[:pos] + aa_mut + seq[pos + 1:]
        assert(len(seq) == len(escaped))
        if escaped not in seqs_escape:
            seqs_escape[escaped] = []
        seqs_escape[escaped].append({
            'mutation': mut,
            'significant': True,
        })

    return seq, seqs_escape

def load_greaney2020(survival_cutoff=0.3,
                     binding_cutoff=-0.4, expr_cutoff=-0.4):
    seq = SeqIO.read('data/sarscov2/data/cov/cov2_spike_wt.fasta', 'fasta').seq

    sig_sites = set()
    with open('data/sarscov2/data/cov/greaney2020cov2/significant_escape_sites.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            sig_sites.add(int(fields[1]) - 1)

    binding = {}
    with open('data/sarscov2/data/cov/starr2020cov2/single_mut_effects.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            pos = float(fields[1]) - 1
            aa_orig = fields[2].strip('"')
            aa_mut = fields[3].strip('"')
            if aa_mut == '*':
                continue
            if fields[8] == 'NA':
                score = float('-inf')
            else:
                score = float(fields[8])
            if fields[11] == 'NA':
                expr = float('-inf')
            else:
                expr = float(fields[11])
            binding[(pos, aa_orig, aa_mut)] = score, expr

    seqs_escape = {}
    with open('data/sarscov2/data/cov/greaney2020cov2/escape_fracs.csv') as f:
        f.readline() # Consume header.
        for line in f:
            fields = line.rstrip().split(',')
            antibody = fields[2]
            escape_frac = float(fields[10])
            aa_orig = fields[5]
            aa_mut = fields[6]
            pos = int(fields[4]) - 1
            assert(seq[pos] == aa_orig)
            escaped = seq[:pos] + aa_mut + seq[pos + 1:]
            assert(len(seq) == len(escaped))
            if escaped not in seqs_escape:
                seqs_escape[escaped] = []
            significant = (
                escape_frac > survival_cutoff and
                pos in sig_sites and
                binding[(pos, aa_orig, aa_mut)][0] > binding_cutoff and
                binding[(pos, aa_orig, aa_mut)][1] > expr_cutoff
            )
            seqs_escape[escaped].append({
                'pos': pos,
                'frac_survived': escape_frac,
                'antibody': antibody,
                'significant': significant,
            })

    return seq, seqs_escape

if __name__ == '__main__':
    # load_doud2018()
    # load_lee2019()
    # load_dingens2019()
    # load_baum2020()
    seq, seqs_escape = load_greaney2020()
    k = 0
    for seq in seqs_escape: 
        sum_val = sum([ m['significant'] for m in seqs_escape[seq] ])
        if sum_val > 0:
            k+=1
    print(k)

    unique_abs = set()

    sample = next(iter(seqs_escape.values()))
    for seq,sample in seqs_escape.items():
        ab_set = set()
        for sample_dict in sample:
            ab = sample_dict['antibody']
            if ab not in unique_abs:
                unique_abs.add(ab)
            if sample_dict['antibody'] not in ab_set:
                ab_set.add(sample_dict['antibody'])
            else:
                print(sample_dict['antibody'])
            

    stop()