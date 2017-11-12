import nltk
import sys

from dataset import *


def parse_django_dataset(annot_file, code_file, out_file):
    from lang.py.parse import parse_raw
    MAX_QUERY_LENGTH = 70
    UNARY_CUTOFF_FREQ = 30

    data = preprocess_dataset(annot_file, code_file)

    for e in data:
        e['parse_tree'] = parse_raw(e['code'])

    parse_trees = [e['parse_tree'] for e in data]

    # # apply unary closures
    # unary_closures = get_top_unary_closures(parse_trees, k=0, freq=UNARY_CUTOFF_FREQ)
    # for i, parse_tree in enumerate(parse_trees):
    #     apply_unary_closures(parse_tree, unary_closures)

    # build the grammar
    grammar = get_grammar(parse_trees)

    # # write grammar
    # with open('django.grammar.unary_closure.txt', 'w') as f:
    #     for rule in grammar:
    #         f.write(rule.__repr__() + '\n')

    # build grammar ...
    annot_tokens = list(chain(*[e['query_tokens'] for e in data]))
    annot_vocab = gen_vocab(annot_tokens, vocab_size=5000, freq_cutoff=3)

    terminal_token_seq = []
    empty_actions_count = 0

    # helper function begins
    def get_terminal_tokens(_terminal_str):
        tmp_terminal_tokens = _terminal_str.split(' ')
        _terminal_tokens = []
        for token in tmp_terminal_tokens:
            if token:
                _terminal_tokens.append(token)
            _terminal_tokens.append(' ')

        return _terminal_tokens[:-1]
        # return _terminal_tokens
    # helper function ends

    # first pass
    for entry in data:
        parse_tree = entry['parse_tree']

        for node in parse_tree.get_leaves():
            if grammar.is_value_node(node):
                terminal_val = node.value
                terminal_str = str(terminal_val)

                terminal_tokens = get_terminal_tokens(terminal_str)

                for terminal_token in terminal_tokens:
                    #assert len(terminal_token) > 0
                    terminal_token_seq.append(terminal_token)

    terminal_vocab = gen_vocab(terminal_token_seq, vocab_size=5000, freq_cutoff=3)
    assert '_STR:0_' in terminal_vocab

    train_data = DataSet(annot_vocab, terminal_vocab, grammar, 'train_data')
    dev_data = DataSet(annot_vocab, terminal_vocab, grammar, 'dev_data')
    test_data = DataSet(annot_vocab, terminal_vocab, grammar, 'test_data')

    all_examples = []

    can_fully_gen_num = 0

    # second pass
    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']
        str_map = entry['str_map']
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
                    if tok_src_idx < 0 or tok_src_idx >= MAX_QUERY_LENGTH:
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

        # train, valid, test
        if 0 <= idx < 16000:
            train_data.add(example)
        elif 16000 <= idx < 17000:
            dev_data.add(example)
        else:
            test_data.add(example)

        all_examples.append(example)

    # print statistics
    max_query_len = max(len(e.query) for e in all_examples)
    max_actions_len = max(len(e.actions) for e in all_examples)

    serialize_to_file([len(e.query) for e in all_examples], 'query.len')
    serialize_to_file([len(e.actions) for e in all_examples], 'actions.len')

    logging.info('examples that can be fully reconstructed: %d/%d=%f',
                 can_fully_gen_num, len(all_examples),
                 can_fully_gen_num / len(all_examples))
    logging.info('empty_actions_count: %d', empty_actions_count)
    logging.info('max_query_len: %d', max_query_len)
    logging.info('max_actions_len: %d', max_actions_len)

    train_data.init_data_matrices()
    dev_data.init_data_matrices()
    test_data.init_data_matrices()

    serialize_to_file((train_data, dev_data, test_data), out_file)


    return train_data, dev_data, test_data


def query_to_data(query, annot_vocab):
    query_tokens = query.split(' ')
    token_num = min(config.max_qeury_length, len(query_tokens))
    data = np.zeros((1, token_num), dtype='int32')

    for tid, token in enumerate(query_tokens[:token_num]):
        token_id = annot_vocab[token]

        data[0, tid] = token_id

    return data


QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")


def canonicalize_query(query):
    """
    canonicalize the query, replace strings to a special place holder
    """
    str_count = 0
    str_map = dict()

    matches = QUOTED_STRING_RE.findall(query)
    # de-duplicate
    cur_replaced_strs = set()
    for match in matches:
        # If one or more groups are present in the pattern,
        # it returns a list of groups
        quote = match[0]
        str_literal = quote + match[1] + quote

        if str_literal in cur_replaced_strs:
            continue

        # FIXME: substitute the ' % s ' with
        if str_literal in ['\'%s\'', '\"%s\"']:
            continue

        str_repr = '_STR:%d_' % str_count
        str_map[str_literal] = str_repr

        query = query.replace(str_literal, str_repr)

        str_count += 1
        cur_replaced_strs.add(str_literal)

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

    return new_query_tokens, str_map


def canonicalize_example(query, code):
    from lang.py.parse import parse_raw, parse_tree_to_python_ast, canonicalize_code as make_it_compilable
    import astor, ast

    query_tokens, str_map = canonicalize_query(query)
    canonical_code = code

    for str_literal, str_repr in str_map.iteritems():
        canonical_code = canonical_code.replace(str_literal, '\'' + str_repr + '\'')

    canonical_code = make_it_compilable(canonical_code)

    # sanity check
    parse_tree = parse_raw(canonical_code)
    gold_ast_tree = ast.parse(canonical_code).body[0]
    gold_source = astor.to_source(gold_ast_tree)
    ast_tree = parse_tree_to_python_ast(parse_tree)
    source = astor.to_source(ast_tree)

    assert gold_source == source, 'sanity check fails: gold=[%s], actual=[%s]' % (gold_source, source)

    return query_tokens, canonical_code, str_map


def preprocess_dataset(annot_file, code_file):
    f_annot = open('annot.all.canonicalized.txt', 'w')
    f_code = open('code.all.canonicalized.txt', 'w')

    examples = []

    err_num = 0
    for idx, (annot, code) in enumerate(zip(open(annot_file), open(code_file))):
        annot = annot.strip()
        code = code.strip()
        try:
            clean_query_tokens, clean_code, str_map = canonicalize_example(annot, code)
            example = {'id': idx, 'query_tokens': clean_query_tokens, 'code': clean_code,
                       'str_map': str_map, 'raw_code': code}
            examples.append(example)

            f_annot.write('example# %d\n' % idx)
            f_annot.write(' '.join(clean_query_tokens) + '\n')
            f_annot.write('%d\n' % len(str_map))
            for k, v in str_map.iteritems():
                f_annot.write('%s ||| %s\n' % (k, v))

            f_code.write('example# %d\n' % idx)
            f_code.write(clean_code + '\n')
        except:
            print code
            err_num += 1

        idx += 1

    f_annot.close()
    f_annot.close()

    # serialize_to_file(examples, 'django.cleaned.bin')

    print 'error num: %d' % err_num
    print 'preprocess_dataset: cleaned example num: %d' % len(examples)

    return examples


if __name__== '__main__':
    from nn.utils.generic_utils import init_logging
    init_logging('parse.log')

    out_file = 'data/django.cleaned.dataset.freq3.par_info.refact.space_only.order_by_ulink_len.bin'
    # 'data/django.cleaned.dataset.freq5.par_info.refact.space_only.unary_closure.freq{UNARY_CUTOFF_FREQ}.order_by_ulink_len.bin'.format(UNARY_CUTOFF_FREQ=UNARY_CUTOFF_FREQ)
    annot_file = sys.argv[1]
    code_file = sys.argv[2]

    parse_django_dataset(annot_file, code_file, out_file)
