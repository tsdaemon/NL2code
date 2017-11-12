import sys
import os
import gzip
import tqdm

from dataset import *
from lang.py.parse import parse_raw


def get_terminal_tokens(_terminal_str):
        tmp_terminal_tokens = _terminal_str.split(' ')
        _terminal_tokens = []
        for token in tmp_terminal_tokens:
            if token:
                _terminal_tokens.append(token)
            _terminal_tokens.append(' ')

        return _terminal_tokens[:-1]


def generate_grammars(data_train, data_dev, data_test, unary_closure_cutoff=None):

    print('Grammar...')
    parse_trees = [e['parse_tree'] for e in data_train + data_dev + data_test]

    # apply unary closures
    if unary_closure_cutoff is not None:
        unary_closures = get_top_unary_closures(parse_trees, k=0, freq=unary_closure_cutoff)
        for i, parse_tree in enumerate(parse_trees):
            apply_unary_closures(parse_tree, unary_closures)

    grammar = get_grammar(parse_trees)

    print('Annotation vocab...')
    annot_tokens = list(chain(*[e['query_tokens'] for e in data_train + data_dev + data_test]))
    annot_vocab = gen_vocab(annot_tokens, vocab_size=5000, freq_cutoff=3)

    terminal_token_seq = []

    print 'Terminal vocab...'
    # terminal vocab
    for entry in data_train + data_dev + data_test:
        parse_tree = entry['parse_tree']

        for node in parse_tree.get_leaves():
            if grammar.is_value_node(node):
                terminal_val = node.value
                terminal_str = str(terminal_val)

                terminal_tokens = get_terminal_tokens(terminal_str)

                for terminal_token in terminal_tokens:
                    assert len(terminal_token) > 0
                    terminal_token_seq.append(terminal_token)
    terminal_vocab = gen_vocab(terminal_token_seq, vocab_size=5000, freq_cutoff=3)

    return annot_vocab, terminal_vocab, grammar


def parse_br_dataset(data_dir, out_file):
    UNARY_CUTOFF_FREQ = 30
    print 'Preprocessing...'
    data_train, data_dev, data_test = preprocess_datasets(data_dir)
    print 'Generating grammar...'
    desc_vocab, terminal_vocab, grammar = generate_grammars(data_train, data_dev, data_test)

    all_examples = []

    can_fully_gen_num = 0
    empty_actions_count = 0

    train, ea, cf = parse_actions(data_train, desc_vocab, terminal_vocab, grammar, 'train_data')
    can_fully_gen_num += cf
    empty_actions_count += ea
    dev, ea, cf = parse_actions(data_dev, desc_vocab, terminal_vocab, grammar, 'dev_data')
    can_fully_gen_num += cf
    empty_actions_count += ea
    test, ea, cf = parse_actions(data_test, desc_vocab, terminal_vocab, grammar, 'test_data')
    can_fully_gen_num += cf
    empty_actions_count += ea

    # print statistics
    max_query_len = max(len(e.query) for e in all_examples)
    max_actions_len = max(len(e.actions) for e in all_examples)

    serialize_to_file([len(e.query) for e in all_examples], 'stat/query.br.len')
    serialize_to_file([len(e.actions) for e in all_examples], 'stat/actions.br.len')

    logging.info('examples that can be fully reconstructed: %d/%d=%f',
                 can_fully_gen_num, len(all_examples),
                 can_fully_gen_num / len(all_examples))
    logging.info('empty_actions_count: %d', empty_actions_count)
    logging.info('max_query_len: %d', max_query_len)
    logging.info('max_actions_len: %d', max_actions_len)

    print 'Initializing matrices...'
    train.init_data_matrices()
    dev.init_data_matrices()
    test.init_data_matrices()

    print 'Serializing...'
    serialize_to_file((train, dev, test), out_file)

    print 'Done!'
    return train, dev, test


def parse_actions(data, desc_vocab, terminal_vocab, grammar, dataset_name):
    dataset = DataSet(desc_vocab, terminal_vocab, grammar, dataset_name)
    empty_actions_count = 0
    can_fully_gen_num = 0
    print 'Creating dataset %s...' % dataset_name
    for entry in tqdm(data):
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']
        parse_tree = entry['parse_tree']

        rule_list, rule_parents = parse_tree.get_productions(include_value_node=True)

        actions = []
        can_fully_gen = True
        rule_pos_map = dict()

        for rule_count, rule in enumerate(rule_list):
            if not grammar.is_value_node(rule.parent):
                assert rule.value is None
                parent_rule = rule_parents[(rule_count, rule)][0]
                if parent_rule:
                    parent_t = rule_pos_map[parent_rule]
                else:
                    parent_t = 0

                rule_pos_map[rule] = len(actions)

                d = {'rule': rule, 'parent_t': parent_t, 'parent_rule': parent_rule}
                action = Action(APPLY_RULE, d)

                actions.append(action)
            else:
                assert rule.is_leaf

                parent_rule = rule_parents[(rule_count, rule)][0]
                parent_t = rule_pos_map[parent_rule]

                terminal_val = rule.value
                terminal_str = str(terminal_val)
                terminal_tokens = get_terminal_tokens(terminal_str)

                # assert len(terminal_tokens) > 0

                for terminal_token in terminal_tokens:
                    term_tok_id = terminal_vocab[terminal_token]
                    tok_src_idx = -1
                    try:
                        tok_src_idx = query_tokens.index(terminal_token)
                    except ValueError:
                        pass

                    d = {'literal': terminal_token, 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}

                    # cannot copy, only generation
                    # could be unk!
                    if tok_src_idx < 0:
                        action = Action(GEN_TOKEN, d)
                        if terminal_token not in terminal_vocab:
                            if terminal_token not in query_tokens:
                                # print terminal_token
                                can_fully_gen = False
                    else:  # copy
                        if term_tok_id != terminal_vocab.unk:
                            d['source_idx'] = tok_src_idx
                            action = Action(GEN_COPY_TOKEN, d)
                        else:
                            d['source_idx'] = tok_src_idx
                            action = Action(COPY_TOKEN, d)

                    actions.append(action)

                d = {'literal': '<eos>', 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}
                actions.append(Action(GEN_TOKEN, d))

        if len(actions) == 0:
            empty_actions_count += 1
            continue

        example = DataEntry(idx, query_tokens, parse_tree, code, actions,
                            {'raw_code': entry['raw_code'], 'str_map': entry['str_map']})

        if can_fully_gen:
            can_fully_gen_num += 1

        dataset.add(example)
        return dataset, empty_actions_count, can_fully_gen_num


def create_query(declaration, description):
    """
    creates query from function declaration and description
    """
    # remove def
    def_ = declaration.find("def")
    declaration = declaration[def_+4:]
    # form a description string from it
    declaration = declaration.replace("_", " ")
    declaration = declaration.replace("(", " (")
    declaration = declaration.replace(":", ".")

    # remove DCNL from description
    description = description.replace(" DCNL ", " ")

    query = declaration + " " + description

    # tokenize
    query_tokens = nltk.word_tokenize(query)

    new_query_tokens = []
    # break up function calls like foo.bar.func
    for token in query_tokens:
        new_query_tokens.append(token)
        i = token.find('.')
        if 0 < i < len(token) - 1:
            new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
            new_query_tokens.extend(new_tokens)

    query = ' '.join(new_query_tokens)

    return query


def canonicalize_code(code):
    from lang.py.parse import parse_raw, parse_tree_to_python_ast, canonicalize_code as make_it_compilable
    import astor, ast

    canonical_code = make_it_compilable(code)

    # sanity check
    parse_tree = parse_raw(canonical_code)
    gold_ast_tree = ast.parse(canonical_code).body[0]
    gold_source = astor.to_source(gold_ast_tree)
    ast_tree = parse_tree_to_python_ast(parse_tree)
    source = astor.to_source(ast_tree)

    assert gold_source == source, 'sanity check fails: gold=[%s], actual=[%s]' % (gold_source, source)

    return canonical_code


def preprocess_dataset(decl_file, desc_file, code_file):
    examples = []

    err_num = 0
    for idx, (decl, desc, code) in enumerate(zip(decl_file, desc_file, code_file)):
        decl = decl.strip()
        desc = desc.strip()
        code = code.strip()
        try:
            clean_query_tokens = create_query(decl, desc)
            clean_code = canonicalize_code(code)
            example = {'id': idx,
                       'query_tokens': clean_query_tokens,
                       'code': clean_code,
                       'raw_code': code,
                       'parse_tree': parse_raw(clean_code)}
            examples.append(example)

        except:
            print code
            err_num += 1

        idx += 1

    print 'error num: %d' % err_num
    print 'preprocess_dataset: cleaned example num: %d' % len(examples)
    decl_file.close()
    desc_file.close()
    code_file.close()
    return examples


def preprocess_datasets(data_dir):
    path = os.path.join(data_dir, 'repo_split.data_ps.')
    decl_path = path + 'declarations.'
    desc_path = path + 'descriptions.'
    body_path = path + 'bodies.'

    print 'Preprocessing train...'
    train_data = preprocess_dataset(open(decl_path + 'train', 'r'),
                                    open(desc_path + 'train', 'r'),
                                    gzip.open(body_path + 'train'))

    print 'Preprocessing dev...'
    dev_data = preprocess_dataset(open(decl_path + 'valid', 'r'),
                                  open(desc_path + 'valid', 'r'),
                                  open(body_path + 'valid', 'r'))

    print 'Preprocessing test...'
    test_data = preprocess_dataset(open(decl_path + 'test', 'r'),
                                   open(desc_path + 'test', 'r'),
                                   open(body_path + 'test', 'r'))

    return train_data, dev_data, test_data



if __name__ == '__main__':
    from nn.utils.generic_utils import init_logging
    init_logging('parse.log')

    data_dir = sys.argv[1]

    parse_br_dataset(data_dir, './data/br.bin')