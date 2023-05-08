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
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from data_utils import extract_nm_name, fetch_seq_from_ncbi, extract_mut, add_mut_to_ref_seq
import tensorflow as tf
from pdb import set_trace as bp

def closest_power_of_two(n):
    a = int(math.log2(n))
    if 2**a == n:
        return n
    return 2**(a + 1)


class ProteinBertWrapper(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.model_path = args.model_path
        if os.path.isdir(self.model_path):
            self.pkl_model_path = self.model_path + '/proteinbert_finetuned.pkl'
        else:
            self.pkl_model_path = self.model_path

        self.out_pkl_dir_path = self.pkl_model_path if os.path.isdir(self.model_path) else os.path.basename(self.model_path)
        self.cls_threshold = args.cls_threshold
        self.sampling_policy = args.sampling_policy
        self.regenerate_data = args.regenerate_data
        self.use_pairs = args.use_pairs
        self.is_cat = args.is_cat

        # A local (non-global) binary output
        out_type = 'categorical' if self.is_cat else 'binary'
        self.OUTPUT_TYPE = OutputType(False, out_type)
        self.UNIQUE_LABELS = [0, 1]
        self.OUTPUT_SPEC = OutputSpec(self.OUTPUT_TYPE, self.UNIQUE_LABELS)
        
        self.train_db_name = os.path.join(self.data_path, 'train.csv')
        self.val_db_name = os.path.join(self.data_path, 'val.csv')
        self.test_db_name = os.path.join(self.data_path, 'test.csv')
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
        if self.sampling_policy == 'over':
            sampler = RandomOverSampler(sampling_strategy='minority')
        else:
            sampler = RandomUnderSampler(sampling_strategy='majority')
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
        db['RefSequence'] = db['RefSequence'].apply(lambda x: x.replace(' ', ''))
        return db

    def find_minmax_len(self, db):
        seqlen = [len(seq) for seq in db['mutSequence']]
        minlen = min(seqlen)
        maxlen = max(seqlen)
        print(f'Min len: {minlen} max len: {maxlen}') 
        return minlen, maxlen

    def read_data(self, read_train=True, read_val=True, read_test=True):
        data_path = self.data_path
        from_saved = not self.regenerate_data
        save = self.regenerate_data
        if from_saved and os.path.exists(self.train_db_name):
            if read_train and os.path.exists(self.train_db_name) and self.train_set is None:
                self.train_set = pd.read_csv(self.train_db_name)
                self.train_set = self.untokenize(self.train_set)
                self.find_minmax_len(self.train_set)
            if read_val and os.path.exists(self.val_db_name) and self.val_set is None:
                self.val_set = pd.read_csv(self.val_db_name)
                self.val_set = self.untokenize(self.val_set)
                self.find_minmax_len(self.val_set)
            if read_test and os.path.exists(self.test_db_name) and self.test_set is None:
                self.test_set = pd.read_csv(self.test_db_name)       
                self.test_set = self.untokenize(self.test_set)
                self.find_minmax_len(self.test_set)
        else:
            db_file_path = os.path.join(data_path, 'all_data.csv')
            df = pd.read_csv(db_file_path).dropna().drop_duplicates()
            df = df.rename(columns={'Name': 'name'})
            df['pname'] = df['name'].apply(lambda x: x.split('.')[0])
            if 'labels' not in df.columns:
                df['labels'] = np.where(df['category'] == 'patho', 1, 0)
            df_valid = df[df.astype(object).ne('-1').all(axis=1)]

            self.train_set, self.val_set, self.test_set = self.better_split(df_valid)
            self.train_set = self.untokenize(self.train_set)
            self.val_set = self.untokenize(self.val_set)
            self.test_set = self.untokenize(self.test_set)

        if save:
            self.train_set.to_csv(self.train_db_name)
            self.val_set.to_csv(self.val_db_name)
            self.test_set.to_csv(self.test_db_name)

        '''
        print(f'{len(self.train_set)} training set records, ' \
            '{len(self.val_set)} validation set records, ' \
                '{len(self.test_set)} test set records.')
        '''


    def train(self, epochs=5, batch_size=8):
        self.read_data()
        print(tf.config.list_physical_devices('GPU'))
        
        pretrained_model_generator, input_encoder = load_pretrained_model()
        model_generator = FinetuningModelGenerator(pretrained_model_generator, self.OUTPUT_SPEC,\
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

        training_callbacks = [
            keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-05, verbose=1),
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(self.out_pkl_dir_path + '/{epoch:02d}-{val_loss:.2f}.pkl', save_weights_only=True, save_best_only=False, save_freq='epoch')
        ]

        '''
        finetune(model_generator, input_encoder, OUTPUT_SPEC, \
            self.train_set['mutSequence'], self.train_set['labels'], self.val_set['mutSequence'], self.val_set['labels'], \
            seq_len=1024, batch_size=8, max_epochs_per_stage=30, lr=1e-04, begin_with_frozen_pretrained_layers=True, \
            lr_with_frozen_pretrained_layers=1e-03, n_final_epochs=1, final_seq_len=2048, final_lr=1e-03, callbacks=training_callbacks)
        '''

        if self.use_pairs:
            finetune(model_generator, input_encoder, self.OUTPUT_SPEC, \
            self.train_set['RefSequence'], self.train_set['labels'], self.val_set['RefSequence'], self.val_set['labels'], \
            seq_len=1024, batch_size=batch_size, max_epochs_per_stage=epochs, lr=1e-04, begin_with_frozen_pretrained_layers=True, \
            lr_with_frozen_pretrained_layers=1e-03, n_final_epochs=1, final_seq_len=2048, final_lr=1e-03, callbacks=training_callbacks, \
            train_seq_muts=self.train_set['mutSequence'], valid_seq_muts=self.val_set['mutSequence'])
        else:
            finetune(model_generator, input_encoder, self.OUTPUT_SPEC, \
            self.train_set['mutSequence'], self.train_set['labels'], self.val_set['mutSequence'], self.val_set['labels'], \
            seq_len=1024, batch_size=batch_size, max_epochs_per_stage=epochs, lr=1e-04, begin_with_frozen_pretrained_layers=True, \
            lr_with_frozen_pretrained_layers=1e-03, n_final_epochs=1, final_seq_len=2048, final_lr=1e-03, callbacks=training_callbacks)

        print('--------------------------------------------')
        print('---------------Training Done!---------------')
        print('--------------------------------------------')
        self.test(model_generator, input_encoder)

        # save the model
        with open(self.out_pkl_dir_path + 'model_last.pkl', 'wb') as f:
            pickle.dump(model_generator.model_weights, f)


    def load_pretrained(self):
        with open(self.pkl_model_path, 'rb') as f:
            saved_model_weights = pickle.load(f)  
        if len(saved_model_weights) == 3: # num_annotations, model_weights, optimizer_weights
            model_weights = saved_model_weights[1]
        else: #model_weights only
            model_weights = saved_model_weights
        saved_pretrained_model_generator, saved_input_encoder = load_pretrained_model()
        saved_model_generator = FinetuningModelGenerator(saved_pretrained_model_generator, self.OUTPUT_SPEC, 
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs, 
            dropout_rate=0.5,
            model_weights=model_weights)
        return saved_model_generator, saved_input_encoder

    def test(self, model_generator=None, input_encoder=None, batch_size=8):
        self.read_data(read_train=False, read_val=False, read_test=True)
        if model_generator == None or input_encoder == None:
            model_generator, input_encoder = self.load_pretrained()
        if self.use_pairs:
            results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, self.OUTPUT_SPEC,\
            self.test_set['RefSequence'], self.test_set['labels'], \
            start_seq_len=1024, start_batch_size=batch_size, seq_muts=self.test_set['mutSequence'])
        else:
            results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, self.OUTPUT_SPEC,\
            self.test_set['mutSequence'], self.test_set['labels'], \
            start_seq_len=1024, start_batch_size=batch_size)
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
        if self.use_pairs:
            seq_len = max(1024, closest_power_of_two(len(mut_seq)* 2 + 5)) # 2* begin and end + sep
        else:
            seq_len = max(1024, closest_power_of_two(len(mut_seq) + 2)) # +2 because of begin and end tokens
        model = model_generator.create_model(seq_len)
        if self.use_pairs:
            x = input_encoder.encode_X_pairs([seq], [mut_seq], seq_len)
        else:
            x = input_encoder.encode_X([mut_seq], seq_len)
        y_pred = model.predict(x, batch_size=1)
        predicted_class = int(y_pred >= self.cls_threshold)
        print(f'Predicted label: {self.class2label[predicted_class]}, raw score: {y_pred.flatten()}')

def main(args):
    SAMPLING_POLICIES = ['under', 'over']
    if args.sampling_policy not in SAMPLING_POLICIES:
        raise ValueError(f'Sampling policy {args.sampling_policy} not allowed. Valid options: {SAMPLING_POLICIES}')
    proteinbert = ProteinBertWrapper(args)
    if args.mode == 'gen':
        proteinbert.read_data(read_train=True, read_val=True, read_test=True)
    elif args.mode == 'train':
        proteinbert.train(args.num_epochs, args.batch_size)
    elif args.mode == 'test':
        proteinbert.test(args.batch_size)
    elif args.mode == 'predict':
        proteinbert.predict(args.nm_name, args.change)
    else:
        raise ValueError(f'Running mode {args.mode} not supported. Supported modes: gen | train | test | predict')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate data , train, test or infer a single sample for Protein pathogenicity')
    parser.add_argument('--mode', type=str, help='Running mode (gen for data generation, train for training, test for testing, \
     predict for predicting a single sample)')
    parser.add_argument('--model-path', type=str, help='Path of trained model to load or save', default='models/')
    parser.add_argument('--data-path', type=str, help='Data path')
    parser.add_argument('--nm-name', type=str, help='protein name for predict mode', default='NM_000355')
    parser.add_argument('--change', type=str, help='Mutation information', default='p.Arg259Pro')
    parser.add_argument('--cls-threshold', type=float, help='positive / negative cutoff threshold', default=0.5)
    parser.add_argument('--num-epochs', type=int, help='Number of epochs to run per stage', default=30)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=16)
    parser.add_argument('--regenerate-data', action='store_true', help='Regenerate train / val / test sets')
    parser.add_argument('--use-pairs', action='store_true', help='Use pairs or reference+mutation instead of mutation only')
    parser.add_argument('--sampling-policy', type=str, help='Sampling policy for imbalanced data, either over or under', default='under')
    parser.add_argument('--is-cat', action='store_true', help='Use categorical classification instead of binary output')
    args = parser.parse_args()
    main(args)
