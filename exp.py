import argparse
import os
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import math
import numpy as np
import pickle
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len, encode_dataset
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from data_utils import extract_nm_name, fetch_seq_from_ncbi, extract_mut, add_mut_to_ref_seq
import tensorflow as tf
from pdb import set_trace as bp

# A local (non-global) binary output
OUTPUT_TYPE = OutputType(False, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

def closest_power_of_two(n):
    a = int(math.log2(n))
    if 2**a == n:
        return n
    return 2**(a + 1)


class ProteinBertWrapper(object):
    def __init__(self, args):
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.model_path = args.model_path
        if os.path.isdir(self.model_path):
            self.pkl_model_path = self.model_path + '/proteinbert_finetuned.pkl'
        else:
            self.pkl_model_path = self.model_path
        self.cls_threshold = args.cls_threshold
        self.num_epochs = args.num_epochs
        self.train_db_name = os.path.join(self.data_path, 'train_data.csv')
        self.val_db_name = os.path.join(self.data_path, 'val_data.csv')
        self.test_db_name = os.path.join(self.data_path, 'test_data.csv')
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.class2label = {0: 'Benign', 1: 'Pathogenic'}

    def get_label(self, all_labels):
        # Majority voting
        vals, cnt = np.unique(all_labels, return_counts=True)
        voted_label = np.argmax(cnt) if len(cnt) > 1 else vals[0]
        return voted_label
    
    def better_split(self, df, train_percent=.6, validate_percent=.2, seed=42):
        # labels: 'benign', 'patho'
        pnames = df['pname'].tolist()
        unames = np.unique(pnames)
        labels = df['labels'].tolist()
        p_to_lbl = dict([(key, []) for key in unames])
        for pname, lbl in zip(pnames, labels):
            p_to_lbl[pname].append(lbl)
        uname_to_label = dict([(key, self.get_label(p_to_lbl[key])) for key in unames])
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

    def augment_db(self, df):
        pass

    def untokenize(self, db):
        db['mutSequence'] = db['mutSequence'].apply(lambda x: x.replace(' ', ''))
        #db['refSequence'] = db['refSequence'].apply(lambda x: x.replace(' ', ''))
        return db

    def find_minmax_len(self, db):
        seqlen = [len(seq) for seq in db['mutSequence']]
        minlen = min(seqlen)
        maxlen = max(seqlen)
        print(f'Min len: {minlen} max len: {maxlen}') 
        return minlen, maxlen


    def read_data(self, read_train=True, read_val=True, read_test=True):
        data_path = self.data_path
        from_saved = True
        save = False
        if from_saved and os.path.exists(self.train_db_name) \
            and os.path.exists(self.val_db_name) \
            and os.path.exists(self.test_db_name):
            if read_train:
                self.train_set = pd.read_csv(self.train_db_name)
                self.train_set = self.untokenize(self.train_set)
                self.find_minmax_len(self.train_set)
            if read_val:
                self.valid_set = pd.read_csv(self.val_db_name)
                self.valid_set = self.untokenize(self.valid_set)
                self.find_minmax_len(self.valid_set)
            if read_test:
                self.test_set = pd.read_csv(self.test_db_name)       
                self.test_set = self.untokenize(self.test_set)
                self.find_minmax_len(self.test_set)
        else:
            db_file_path = os.path.join(data_path, 'all_data.csv')
            df = pd.read_csv(db_file_path).dropna().drop_duplicates()
            df = df.rename(columns={"Name": "name"})
            df['pname'] = df['name'].apply(lambda x: x.split('.')[0])
            if 'labels' not in df.columns:
                df['labels'] = np.where(df['category'] == 'patho', 1, 0)
            df_valid = df[df.astype(object).ne("-1").all(axis=1)]

            self.train_set, self.val_set, self.test_set = self.better_split(df_valid)
            self.train_set = self.untokenize(self.train_set)
            self.valid_set = self.untokenize(self.valid_set)
            self.test_set = self.untokenize(self.test_set)

        if save:
            self.train_set.to_csv(self.train_db_name)
            self.valid_set.to_csv(self.val_db_name)
            self.test_set.to_csv(self.test_db_name)

        '''
        print(f'{len(self.train_set)} training set records, ' \
            '{len(self.valid_set)} validation set records, ' \
                '{len(self.test_set)} test set records.')
        '''


    def train(self):
        self.read_data()
        print(tf.config.list_physical_devices('GPU'))
        
        pretrained_model_generator, input_encoder = load_pretrained_model()
        model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC,\
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

        training_callbacks = [
            keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-05, verbose=1),
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        ]

        '''
        finetune(model_generator, input_encoder, OUTPUT_SPEC, \
            self.train_set['mutSequence'], self.train_set['labels'], self.valid_set['mutSequence'], self.valid_set['labels'], \
            seq_len=1024, batch_size=8, max_epochs_per_stage=30, lr=1e-04, begin_with_frozen_pretrained_layers=True, \
            lr_with_frozen_pretrained_layers=1e-03, n_final_epochs=1, final_seq_len=2048, final_lr=1e-03, callbacks=training_callbacks)
        '''

        finetune(model_generator, input_encoder, OUTPUT_SPEC, \
            self.train_set['mutSequence'], self.train_set['labels'], self.valid_set['mutSequence'], self.valid_set['labels'], \
            seq_len=1024, batch_size=8, max_epochs_per_stage=self.num_epochs, lr=1e-04, begin_with_frozen_pretrained_layers=True, \
            lr_with_frozen_pretrained_layers=1e-03, n_final_epochs=1, final_seq_len=2048, final_lr=1e-03, callbacks=training_callbacks)

        self.test(model_generator, input_encoder)

        # save the model
        with open(self.pkl_model_path, 'wb') as f:
            pickle.dump(model_generator.model_weights, f)


    def load_pretrained(self):
        with open(self.pkl_model_path, 'rb') as f:
            saved_model_weights = pickle.load(f)  
        if len(saved_model_weights) == 3: # num_annotations, model_weights, optimizer_weights
            model_weights = saved_model_weights[1]
        else: #model_weights only
            model_weights = saved_model_weights
        saved_pretrained_model_generator, saved_input_encoder = load_pretrained_model()
        saved_model_generator = FinetuningModelGenerator(saved_pretrained_model_generator, OUTPUT_SPEC, 
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs, 
            dropout_rate=0.5,
            model_weights=model_weights)
        return saved_model_generator, saved_input_encoder

    def test(self, model_generator=None, input_encoder=None):
        self.read_data(read_train=False, read_val=False, read_test=True)
        if model_generator == None or input_encoder == None:
            model_generator, input_encoder = self.load_pretrained()
        results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC,\
            self.test_set['mutSequence'], self.test_set['labels'], \
            start_seq_len=1024, start_batch_size=8)

        print('Test-set performance:')
        print(results)

        print('Confusion matrix:')
        print(confusion_matrix)

        acc = sum(np.diag(confusion_matrix)) / len(self.test_set)
        print(f'Overall test acc: {acc}')


    def predict(self, nm_name, change): # TODO: multiple proteins?
        prot_name = extract_nm_name(nm_name)
        seq = fetch_seq_from_ncbi(prot_name)
        mut_seq, mut_pos = add_mut_to_ref_seq(seq, change)
        model_generator, input_encoder = self.load_pretrained()
        seq_len = min(1024, closest_power_of_two(len(mut_seq) + 2))# +2 because of begin and end tokens
        model = model_generator.create_model(seq_len)
        x = input_encoder.encode_X([mut_seq], seq_len)
        y_pred = model.predict(x, batch_size=1)
        predicted_class = int(y_pred >= self.cls_threshold)
        print(f'Predicted label: {self.class2label[predicted_class]}, raw score: {y_pred.flatten()}')

def main(args):
    proteinbert = ProteinBertWrapper(args)
    if args.mode == 'train':
        proteinbert.train()
    elif args.mode == 'test':
        proteinbert.test()
    elif args.mode == 'predict':
        proteinbert.predict(args.nm_name, args.change)
    else:
        print(f'Running mode {args.mode} not supported. Supported modes: train | test | predict')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train, test or infer a single sample for Protein pathogenicity')
    parser.add_argument('--mode', type=str, help='Running mode (train for training, test for testing, \
     predict for predicting a single sample)')
    parser.add_argument('--model-path', type=str, help='Path of trained model to load or save')
    parser.add_argument('--data-path', type=str, help='Data path')
    parser.add_argument('--nm-name', type=str, help='protein name for predict mode', default='NM_000355')
    parser.add_argument('--change', type=str, help='Mutation information', default='p.Arg259Pro')
    parser.add_argument('--cls-threshold', type=float, help='positive / negative cutoff threshold', default=0.5)
    parser.add_argument('--num-epochs', type=int, help='Number of epochs to run per stage', default=30)
    args = parser.parse_args()
    main(args)
