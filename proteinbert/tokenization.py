import os
ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
USE_SEP = os.environ.get('USE_SEP_TOKEN') == '1'
if USE_SEP:
    ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>', '<SEP>']
else:
    ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']

# Each sequence is added <START> and <END> tokens
ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)

def tokenize_seq(seq):
    other_token_index = additional_token_to_index['<OTHER>']
    return [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(seq)] + \
            [additional_token_to_index['<END>']]

def tokenize_pair(seq_ref, seq_mut):
    sep_token_index = additional_token_to_index['<SEP>'] if USE_SEP else additional_token_to_index['<PAD>']
    return tokenize_seq(seq_ref) + [sep_token_index] + tokenize_seq(seq_mut)

def parse_seq(seq):
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError('Unexpected sequence type: %s' % type(seq))
