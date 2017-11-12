from __future__ import division
import copy

import nltk
from collections import OrderedDict, defaultdict
import logging
import collections
import numpy as np
import string
import re
import astor
from itertools import chain

from nn.utils.io_utils import serialize_to_file, deserialize_from_file

import config
from lang.py.parse import get_grammar
from lang.py.unaryclosure import get_top_unary_closures, apply_unary_closures

# define actions
APPLY_RULE = 0
GEN_TOKEN = 1
COPY_TOKEN = 2
GEN_COPY_TOKEN = 3

ACTION_NAMES = {APPLY_RULE: 'APPLY_RULE',
                GEN_TOKEN: 'GEN_TOKEN',
                COPY_TOKEN: 'COPY_TOKEN',
                GEN_COPY_TOKEN: 'GEN_COPY_TOKEN'}

class Action(object):
    def __init__(self, act_type, data):
        self.act_type = act_type
        self.data = data

    def __repr__(self):
        data_str = self.data if not isinstance(self.data, dict) else \
            ', '.join(['%s: %s' % (k, v) for k, v in self.data.iteritems()])
        repr_str = 'Action{%s}[%s]' % (ACTION_NAMES[self.act_type], data_str)

        return repr_str


class Vocab(object):
    def __init__(self):
        self.token_id_map = OrderedDict()
        self.insert_token('<pad>')
        self.insert_token('<unk>')
        self.insert_token('<eos>')

    @property
    def unk(self):
        return self.token_id_map['<unk>']

    @property
    def eos(self):
        return self.token_id_map['<eos>']

    def __getitem__(self, item):
        if item in self.token_id_map:
            return self.token_id_map[item]

        logging.debug('encounter one unknown word [%s]' % item)
        return self.token_id_map['<unk>']

    def __contains__(self, item):
        return item in self.token_id_map

    @property
    def size(self):
        return len(self.token_id_map)

    def __setitem__(self, key, value):
        self.token_id_map[key] = value

    def __len__(self):
        return len(self.token_id_map)

    def __iter__(self):
        return self.token_id_map.iterkeys()

    def iteritems(self):
        return self.token_id_map.iteritems()

    def complete(self):
        self.id_token_map = dict((v, k) for (k, v) in self.token_id_map.iteritems())

    def get_token(self, token_id):
        return self.id_token_map[token_id]

    def insert_token(self, token):
        if token in self.token_id_map:
            return self[token]
        else:
            idx = len(self)
            self[token] = idx

            return idx


replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))


def tokenize(str):
    str = str.translate(replace_punctuation)
    return nltk.word_tokenize(str)


def gen_vocab(tokens, vocab_size=3000, freq_cutoff=5):
    word_freq = defaultdict(int)

    for token in tokens:
        word_freq[token] += 1

    print 'total num. of tokens: %d' % len(word_freq)

    words_freq_cutoff = [w for w in word_freq if word_freq[w] >= freq_cutoff]
    print 'num. of words appear at least %d: %d' % (freq_cutoff, len(words_freq_cutoff))

    ranked_words = sorted(words_freq_cutoff, key=word_freq.get, reverse=True)[:vocab_size-2]
    ranked_words = set(ranked_words)

    vocab = Vocab()
    for token in tokens:
        if token in ranked_words:
            vocab.insert_token(token)

    vocab.complete()

    return vocab


class DataEntry:
    def __init__(self, raw_id, query, parse_tree, code, actions, meta_data=None):
        self.raw_id = raw_id
        self.eid = -1
        # FIXME: rename to query_token
        self.query = query
        self.parse_tree = parse_tree
        self.actions = actions
        self.code = code
        self.meta_data = meta_data

    @property
    def data(self):
        if not hasattr(self, '_data'):
            assert self.dataset is not None, 'No associated dataset for the example'

            self._data = self.dataset.get_prob_func_inputs([self.eid])

        return self._data

    def copy(self):
        e = DataEntry(self.raw_id, self.query, self.parse_tree, self.code, self.actions, self.meta_data)

        return e


class DataSet:
    def __init__(self, annot_vocab, terminal_vocab, grammar, name='train_data'):
        self.annot_vocab = annot_vocab
        self.terminal_vocab = terminal_vocab
        self.name = name
        self.examples = []
        self.data_matrix = dict()
        self.grammar = grammar

    def add(self, example):
        example.eid = len(self.examples)
        example.dataset = self
        self.examples.append(example)

    def get_dataset_by_ids(self, ids, name):
        dataset = DataSet(self.annot_vocab, self.terminal_vocab,
                          self.grammar, name)
        for eid in ids:
            example_copy = self.examples[eid].copy()
            dataset.add(example_copy)

        for k, v in self.data_matrix.iteritems():
            dataset.data_matrix[k] = v[ids]

        return dataset

    @property
    def count(self):
        if self.examples:
            return len(self.examples)

        return 0

    def get_examples(self, ids):
        if isinstance(ids, collections.Iterable):
            return [self.examples[i] for i in ids]
        else:
            return self.examples[ids]

    def get_prob_func_inputs(self, ids):
        order = ['query_tokens', 'tgt_action_seq', 'tgt_action_seq_type',
                 'tgt_node_seq', 'tgt_par_rule_seq', 'tgt_par_t_seq']

        max_src_seq_len = max(len(self.examples[i].query) for i in ids)
        max_tgt_seq_len = max(len(self.examples[i].actions) for i in ids)

        logging.debug('max. src sequence length: %d', max_src_seq_len)
        logging.debug('max. tgt sequence length: %d', max_tgt_seq_len)

        data = []
        for entry in order:
            if entry == 'query_tokens':
                data.append(self.data_matrix[entry][ids, :max_src_seq_len])
            else:
                data.append(self.data_matrix[entry][ids, :max_tgt_seq_len])

        return data


    def init_data_matrices(self, max_query_length=70, max_example_action_num=100):
        logging.info('init data matrices for [%s] dataset', self.name)
        annot_vocab = self.annot_vocab
        terminal_vocab = self.terminal_vocab

        # np.max([len(e.query) for e in self.examples])
        # np.max([len(e.rules) for e in self.examples])

        query_tokens = self.data_matrix['query_tokens'] = np.zeros((self.count, max_query_length), dtype='int32')
        tgt_node_seq = self.data_matrix['tgt_node_seq'] = np.zeros((self.count, max_example_action_num), dtype='int32')
        tgt_par_rule_seq = self.data_matrix['tgt_par_rule_seq'] = np.zeros((self.count, max_example_action_num), dtype='int32')
        tgt_par_t_seq = self.data_matrix['tgt_par_t_seq'] = np.zeros((self.count, max_example_action_num), dtype='int32')
        tgt_action_seq = self.data_matrix['tgt_action_seq'] = np.zeros((self.count, max_example_action_num, 3), dtype='int32')
        tgt_action_seq_type = self.data_matrix['tgt_action_seq_type'] = np.zeros((self.count, max_example_action_num, 3), dtype='int32')

        for eid, example in enumerate(self.examples):
            exg_query_tokens = example.query[:max_query_length]
            exg_action_seq = example.actions[:max_example_action_num]

            for tid, token in enumerate(exg_query_tokens):
                token_id = annot_vocab[token]

                query_tokens[eid, tid] = token_id

            assert len(exg_action_seq) > 0

            for t, action in enumerate(exg_action_seq):
                if action.act_type == APPLY_RULE:
                    rule = action.data['rule']
                    tgt_action_seq[eid, t, 0] = self.grammar.rule_to_id[rule]
                    tgt_action_seq_type[eid, t, 0] = 1
                elif action.act_type == GEN_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab[token]
                    tgt_action_seq[eid, t, 1] = token_id
                    tgt_action_seq_type[eid, t, 1] = 1
                elif action.act_type == COPY_TOKEN:
                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[eid, t, 2] = src_token_idx
                    tgt_action_seq_type[eid, t, 2] = 1
                elif action.act_type == GEN_COPY_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab[token]
                    tgt_action_seq[eid, t, 1] = token_id
                    tgt_action_seq_type[eid, t, 1] = 1

                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[eid, t, 2] = src_token_idx
                    tgt_action_seq_type[eid, t, 2] = 1
                else:
                    raise RuntimeError('wrong action type!')

                # parent information
                rule = action.data['rule']
                parent_rule = action.data['parent_rule']
                tgt_node_seq[eid, t] = self.grammar.get_node_type_id(rule.parent)
                if parent_rule:
                    tgt_par_rule_seq[eid, t] = self.grammar.rule_to_id[parent_rule]
                else:
                    assert t == 0
                    tgt_par_rule_seq[eid, t] = -1

                # parent hidden states
                parent_t = action.data['parent_t']
                tgt_par_t_seq[eid, t] = parent_t

            example.dataset = self


class DataHelper(object):
    @staticmethod
    def canonicalize_query(query):
        return query
