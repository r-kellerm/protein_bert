import os
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import tensorflow as tf
from pdb import set_trace as bp

# A local (non-global) bianry output
OUTPUT_TYPE = OutputType(False, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
'''
#train_set_file_path = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_06_09/train_data_undersample_06.09.csv'
#test_set_file_path = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_06_09/test_data_undersample_06.09.csv'
#val_set_file_path = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_06_09/val_data_undersample_06.09.csv'

train_set_file_path = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_12_01/train_data_undersample_12_01.csv'
test_set_file_path = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_12_01/test_data_undersample_12_01.csv'
val_set_file_path = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_12_01/val_data_undersample_12_01.csv'
train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
valid_set = pd.read_csv(val_set_file_path).dropna().drop_duplicates()
test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()

# Remove spaces from sequences
train_set['mutSequence'] = train_set['mutSequence'].str.replace(' ', '')
valid_set['mutSequence'] = valid_set['mutSequence'].str.replace(' ', '')
test_set['mutSequence'] = test_set['mutSequence'].str.replace(' ', '')
'''
def get_label(all_labels):
    # Majority voting
    vals, cnt = np.unique(all_labels, return_counts=True)
    voted_label = np.argmax(cnt) if len(cnt) > 1 else vals[0]
    return voted_label
    
def better_split(df, train_percent=.6, validate_percent=.2, seed=42):
    # labels: 'benign', 'patho'
    pnames = df['pname'].tolist()
    unames = np.unique(pnames)
    labels = df['labels'].tolist()
    p_to_lbl = dict([(key, []) for key in unames])
    for pname, lbl in zip(pnames, labels):
        p_to_lbl[pname].append(lbl)
    uname_to_label = dict([(key, get_label(p_to_lbl[key])) for key in unames])
    x_all = np.array(list(uname_to_label.keys())).reshape(-1, 1)
    y_all = list(uname_to_label.values())
    sampler = RandomOverSampler(sampling_strategy='minority')
    x_sampled, y_sampled = sampler.fit_resample(x_all, y_all)
    x_train, x_test_val, y_train, y_test_val = train_test_split(x_sampled, y_sampled, test_size=0.4, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=seed)
    print(f'Train super samples: {len(x_train)}')
    print(f'Validation super samples: {len(x_val)}')
    print(f'Test super samples: {len(x_test)}')
    x_train = x_train.flatten()
    x_val = x_val.flatten()
    x_test = x_test.flatten()

    train_db = df.loc[df['pname'].isin(x_train)]
    val_db = df.loc[df['pname'].isin(x_val)]
    test_db = df.loc[df['pname'].isin(x_test)]
    print(f'Train samples: {len(train_db)}')
    print(f'Validation samples: {len(val_db)}')
    print(f'Test samples: {len(test_db)}')

    return train_db, val_db, test_db

def augment_db(df):
    pass

def untokenize(db):
    db['mutSequence'] = db['mutSequence'].apply(lambda x: x.replace(' ', ''))
    #db['refSequence'] = db['refSequence'].apply(lambda x: x.replace(' ', ''))
    return db

def find_max_len(db):
    seqlen = [len(seq) for seq in db['mutSequence']]
    print(f'Min len: {min(seqlen)} max len: {max(seqlen)}') 

print(tf.config.list_physical_devices('GPU'))
from_saved = True
save = False
TRAIN_DB_NAME = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_06_09/train_data_undersample_06.09.csv'
VAL_DB_NAME = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_06_09/val_data_undersample_06.09.csv'
TEST_DB_NAME = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_06_09/test_data_undersample_06.09.csv'

#TRAIN_DB_NAME = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_4_01/train_data_undersample_4_01.csv'
#VAL_DB_NAME = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_4_01/val_data_undersample_4_01.csv'
#TEST_DB_NAME = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data_split_4_01/test_data_undersample_4_01.csv'

if from_saved and os.path.exists(TRAIN_DB_NAME) and os.path.exists(VAL_DB_NAME) and os.path.exists(TEST_DB_NAME):
    train_set = pd.read_csv(TRAIN_DB_NAME)
    valid_set = pd.read_csv(VAL_DB_NAME)
    test_set = pd.read_csv(TEST_DB_NAME)
    train_set = untokenize(train_set)
    valid_set = untokenize(valid_set)
    test_set = untokenize(test_set)
    find_max_len(train_set)
    find_max_len(valid_set)
    find_max_len(test_set)
else:
    #db_file_path = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/clinvar_hgmd_benign_patho_no_gnomad_filt_09012023_with_seq_23_01.csv'
    #db_file_path = '/DCAIOCTO/ModelDataZoo/datasets/genomicsTrans_shared/data/data/pfam_added_to_fully_validated_data_from_04_11.csv'
    df = pd.read_csv(db_file_path).dropna().drop_duplicates()
    df = df.rename(columns={"Name": "name"})
    df['pname'] = df['name'].apply(lambda x: x.split('.')[0])
    if 'labels' not in df.columns:
        df['labels'] = np.where(df['category'] == 'patho', 1, 0)
    df_valid = df[df.astype(object).ne("-1").all(axis=1)]

    train_set, valid_set, test_set = better_split(df_valid)
    train_set = untokenize(train_set)
    valid_set = untokenize(valid_set)
    test_set = untokenize(test_set)

    if save:
        train_set.to_csv(TRAIN_DB_NAME)
        valid_set.to_csv(VAL_DB_NAME)
        test_set.to_csv(TEST_DB_NAME)

print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')


pretrained_model_generator, input_encoder = load_pretrained_model()
model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
    get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
    keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
]

finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['mutSequence'], train_set['labels'], valid_set['mutSequence'], valid_set['labels'], \
    seq_len = 1024, batch_size = 8, max_epochs_per_stage = 30, lr = 1e-04, begin_with_frozen_pretrained_layers = True, \
    lr_with_frozen_pretrained_layers = 1e-03, n_final_epochs = 1, final_seq_len = 2048, final_lr = 1e-03, callbacks = training_callbacks)

results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['mutSequence'], test_set['labels'], \
    start_seq_len = 1024, start_batch_size = 8)

print('Test-set performance:')
print(results)

print('Confusion matrix:')
print(confusion_matrix)

acc = sum(np.diag(confusion_matrix)) / len(test_set)
print(f'Overall test acc: {acc}')
