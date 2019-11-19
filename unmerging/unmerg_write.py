import pickle
from SimpleGraph import Graph
from SentenceGraph import SentenceGraph
import Utils.ElementTreeUtils as ETUtils
from Utils.ProgressCounter import ProgressCounter
from Utils.InteractionXML.CorpusElements import CorpusElements
import Utils.Libraries.combine as combine
from StructureAnalyzer import StructureAnalyzer
import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
# import Utils.Range as Range
import types
import copy
import re
import pdb
import pickle
import tqdm
from collections import defaultdict, OrderedDict
import argparse
from multitask_LSTM import BiLSTM
import pprint

SIMPLE = ['Gene_expression', 'Transcription', 'Protain_catabolism', 'Localization', 'Phosphorylation']
REG = ['Negative_regulation', 'Positive_regulation', 'Regulation']
BIND = ['Binding']



def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def unmerging(d):
    '''
    d: a sentence. corpus_id, corpus_token, corpus_pos_tag, corpus_trigger_label, corpus_interaction_idx, corpus_interaction_label, corpus_span
    '''
    trigger_labels = d[3]
    int_idxs = d[4]
    int_labels = d[5]
    spans = d[6]
    tokens = d[1]
    assert len(spans) == len(tokens)
    assert len(trigger_labels) == len(tokens)
    assert len(int_idxs) == len(int_labels)

    arg_combs = []
    for idx in range(len(tokens)):
        # For predicted case, if the predicted trigger is None, then skip
        if trigger_labels[idx] == 'None':
            continue
        # find outgoing edges for current tokens
        # out_ints: each element is like ((l_idx, r_idx), int_label)
        out_ints = [i for i in zip(int_idxs, int_labels) if i[0][0] == idx]
        if len(out_ints) == 0:
            continue
        # else, meaning current tokens have outgoing edges
        trigger_label = trigger_labels[idx]
        assert trigger_label != 'None'
        arg_combs.extend(get_valid_combination(trigger_label, out_ints))
        # pdb.set_trace()
    return arg_combs
def get_valid_combination(trigger_label, out_ints, ignore=True):
    '''
    get ALL valid combs, will apply heuristics to get final combs later
    '''

    # if ignore:
    #     # only keep 'Theme' and 'Cause'
    #     for i in range(len(out_ints)):
    #         if out_ints[i][1] not in ['Theme', 'Cause']:
    #             out_ints[i][1] = 'None'

    # TODO: now only consider Theme and Cause
    # we might need to consider other args too in the future

    intsByType = defaultdict(list)
    for i in out_ints:
        intsByType[i[1]].append(i)

    arg_combs = []
    if trigger_label in SIMPLE:
        if ignore:
            # only one argument: (Theme, )
            # arg_combs.extend(intsByType['Theme'])
            for i in combinations(intsByType['Theme'], 1):
                arg_combs.append(i)
    elif trigger_label in BIND:
        if ignore:
            # arbitrary number of Theme args
            # TODO: Note that TEES maps all Theme2, Theme3 to Theme
            max_len = len(intsByType['Theme'])
            for length in range(1, max_len+1):
                for i in combinations(intsByType['Theme'], length):
                    arg_combs.append(i)
    elif trigger_label in REG:
        if ignore:
            # first generate all Theme-only args
            for i in combinations(intsByType['Theme'], 1):
                # first find (Theme, ) combinations
                arg_combs.append(i)
            if 'Cause' in intsByType:
                # NOTE: Cause-only edges are considered as non-event, hence not valid
                # for i in combinations(intsByType['Cause'], 1):
                #     # first find (Cause, ) combinations
                #     # we do this is because there are some inter-sent cases will miss the Theme edge, but leave the Casue edge
                #     # although this is not a valid event but the Cause edge should be helpful training signal and we keep them
                #     arg_combs.append(i)
                # then also find (Theme, Cause) combinations
                for i in combinations(intsByType['Theme'], 1):
                    for j in combinations(intsByType['Cause'], 1):
                        arg_combs.append((i[0], j[0]))
    return arg_combs
def heu_for_unmerge(arg_combs, d):
    '''
    Apply heuristics for different event types here, e.g. when both (Theme, Cause) and (Theme, ) combinations appear for current token
    Only keep the longest chain, i.e. (Theme, Cause)
    params:
        d: sent
        arg_combs: ALL valid combs of this sent
    '''
    final_combs = []
    trigger_types = d[3]
    for idx in range(len(trigger_types)):
        trigger_label = trigger_types[idx]
        cur_combs = [i for i in arg_combs if i[0][0][0] == idx]
        if trigger_label in ['None', 'Protein', 'Entity']:
            continue
        # inter-sent interactions, meaning a trigger no outgoing edges, skip
        if len(cur_combs) == 0:
            continue
        if trigger_label in SIMPLE:
            assert max([len(i) for i in cur_combs]) == 1
            # do nothing for SIMPLE types, because they all have only one Theme
            # final_combs.extend(cur_combs)
        elif trigger_label in BIND:
            # pass
            max_chain_len = max([len(i) for i in cur_combs])
            # only keep longest chain like (Theme, Theme) if any
            cur_combs = [i for i in cur_combs if len(i) == max_chain_len]
        elif trigger_label in REG:
            max_chain_len = max([len(i) for i in cur_combs])
            # only keep longest chain like (Theme, Cause) if any
            cur_combs = [i for i in cur_combs if len(i) == max_chain_len]
        final_combs.extend(cur_combs)
    return final_combs

def is_terminal(event):
    targets = []
    for k, v in event.items():
        if k.startswith('Theme') or k == 'Cause':
            targets.append(v)
    return all([i.startswith('T') for i in targets])


def map_theme_for_bind(cur_comb):
    '''
    Since TEES maps all Theme2, Theme3 args to Theme
    We now convert it back. o.w. there will be unexpected erorwhen constructing the event vocabulary
    since all Theme2, Theme3 will share the same Theme key
    '''
    edge_labels = [i[1] for i in cur_comb]
    assert len(set(edge_labels)) == 1
    if len(cur_comb) == 1:
        return cur_comb
    assert len(cur_comb) > 1 # it must have more than one Theme edge
    out_comb = []
    theme_id = 1
    for edge in cur_comb:
        out_edge = []
        out_edge.append(edge[0])
        if theme_id == 1:
            out_edge.append(edge[1])
        else:
            out_edge.append('{}{}'.format(edge[1], theme_id))
        theme_id +=1
        out_comb.append(tuple(out_edge))
    return out_comb


def writeA2(orig_docid, args, triggerIdBySpan, triggerTypeBySpan, tokenBySpan, all_events):
    triggerIdBySpan = OrderedDict(sorted(triggerIdBySpan.items(), key=lambda x:int(x[0].split('-')[0])))
    all_events = sorted(all_events, key=lambda x: int(x['ST_id'][1:]))
    with open('{}/{}.a2'.format(args.out_dir, orig_docid), 'w') as f:
        # first write triggers
        for span, trigger_id in triggerIdBySpan.items():
            span_l = span.split('-')[0]
            span_r = span.split('-')[1]
            T_line = '{}\t{} {} {}\t{}\n'.format(trigger_id, triggerTypeBySpan[span], span_l, span_r, tokenBySpan[span])
            f.write(T_line)

        # then write events
        for event in all_events:
            arg_str = ' '.join([':'.join([k, v]) for (k, v) in event.items() if k.startswith('Theme') or k == 'Cause'])
            E_line = '{}\t{}:{} '.format(event['ST_id'], event['trigger_type'], triggerIdBySpan[event['trigger_span']]) + \
                     arg_str + '\n'
            f.write(E_line)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--apply_heu', type=str2bool, default=True, help='apply the heuristics of taking the longest chain for unmerging')
    p.add_argument('--out_dir', type=str, default='GE11_a2_out_dev-pred-w-gold')
    args = p.parse_args()


    # with open('GE11_train_flat_w-span.pkl', 'r') as f:
    with open('../out_pkl/GE11_dev_EXP2_Tf10.59687856871_If10.801411142143_pred-w-goldTrue_pred-w-predFalse_lr0.002_drp0.1_ep12_hd60_seed42.pkl', 'r') as f:
        data = pickle.load(f)
    with open('GE11_dev_protIdBySpan.pkl', 'r') as f:
        # NOTE: the protIdBySpan is already processed by TEES
        # this is not the gold annotation from a1 files
        protIdBySpan = pickle.load(f)
    with open('GE11_dev_origIdById.pkl', 'r') as f:
        origIdById = pickle.load(f)

    # pdb.set_trace()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print 'Unmerging ...'
    for i in tqdm.tqdm(range(len(data))):
        arg_combs = unmerging(data[i])
        # pdb.set_trace()
        if len(arg_combs) != 0:
            # TODO: this is an error !!!!!!!!!
            # this will lose events from Gene_exp when there are both REG and Gene_exp
            # max_chain_len = max([len(a) for a in arg_combs])
            # when there are both (Theme, Cause) and (Theme), choose the longer one, i.e. (Theme, Cause)
            # final_combs save the final seletected combinations by our rules
            # these are actuallly the unmerged events
            # final_combs = [a for a in arg_combs if len(a) == max_chain_len]
            if args.apply_heu:
                final_combs = heu_for_unmerge(arg_combs, data[i])
            else:
                final_combs = arg_combs
            # pdb.set_trace()
            data[i] += tuple([final_combs], )
        else:
            data[i] += tuple([[]], )
        # print data[i][0]
        # if data[i][0] == 'GE09.d167.s1':
        #     pdb.set_trace()
        # if data[i][0] == 'GE09.d167.s2':
        #     pdb.set_trace()
    # pdb.set_trace()

    print 'Finding events ...'
    for doc_id in tqdm.tqdm(protIdBySpan.keys()): # doc_id: GE09.d1
        doc_sents = [i for i in data if '.'.join(i[0].split('.')[:-1]) == doc_id]
        # sort the doc with sentence order
        doc_sents = sorted(doc_sents, key=lambda x: int(x[0].split('.')[-1][1:]))

        doc_prots = protIdBySpan[doc_id]    # doc_prots: e.g. '1-3':'10086.T1'
        orig_docid = origIdById[doc_id]
        if doc_prots.values() == []:
            # doc does not have Proteins
            f = open('{}/{}.a2'.format(args.out_dir, orig_docid), 'w')
            f.close()
            continue
        # orig_docid = doc_prots.values()[0].split('.')[0]
        # find the proteins annotated in .a1 file
        # then decide the start id of trigger
        trigger_id = 1 + max([int(i.split('.')[1][1:]) for i in doc_prots.values()])
        # event_id = 1
        triggerIdBySpan = {}
        triggerTypeBySpan = {}
        tokenBySpan = {}
        eventIdsBySpan = defaultdict(list)
        # eventSTById = defaultdict(dict)

        # eventsBySpan = defaultdict(list)
        events = []
        event_id = 1
        print orig_docid
        # if orig_docid == 'PMC-2222968-05-Results-04':
        #     pdb.set_trace()
        for d in doc_sents:
            trigger_types = d[3]
            spans = d[6]
            tokens = d[1]
            int_idxs = d[4]
            int_labels = d[5]
            arg_combs = d[7]
            assert len(trigger_types) == len(spans)
            assert len(spans) == len(tokens)

            for idx in range(len(trigger_types)):
                if trigger_types[idx] in ['Protein', 'Entity', 'None']:
                    # skip triggers that are not events
                    # these include None, Protein, Entity
                    continue
                assert spans[idx] not in triggerIdBySpan
                triggerIdBySpan[spans[idx]] = 'T{}'.format(trigger_id)
                triggerTypeBySpan[spans[idx]] = trigger_types[idx]
                tokenBySpan[spans[idx]] = tokens[idx]
                trigger_id += 1

            # pdb.set_trace()

            for idx in range(len(trigger_types)):
                cur_combs = [i for i in arg_combs if i[0][0][0] == idx]
                # if spans[idx] == '':
                #     pdb.set_trace()
                for cur_comb in cur_combs:
                    event = {}
                    if trigger_types[idx] == 'Binding':
                        # map theme back to theme2 ... for Binding event
                        cur_comb = map_theme_for_bind(cur_comb)
                    event['trigger_span'] = spans[idx]
                    event['trigger_type'] = trigger_types[idx]
                    for edge in cur_comb:
                        assert edge[0][0] == idx
                        target_idx = edge[0][1]
                        target_span = spans[target_idx]
                        arg_role = edge[1]
                        if target_span in triggerIdBySpan:
                            # it is a nesting parent, its target is presented by span for now
                            event[arg_role] = target_span
                        elif target_span in doc_prots:
                            # it points to Protein, replace with the protein ID
                            event[arg_role] = doc_prots[target_span].split('.')[-1]
                        else:
                            # wrong prediction, in gold the target_span should be a event trigger but the predicted target_span position
                            # is None(not picked up by model, then this target_span will not be present in triggerIdBySpan, also not in doc_prots
                            continue
                            # pass

                    # if d[0] == 'GE09.d235.s11':
                    #     pdb.set_trace()
                    if is_terminal(event):
                        event['ST_id'] = 'E{}'.format(event_id)  # Assign event ids for terminal events
                        eventIdsBySpan[spans[idx]].append(event['ST_id'])
                        event_id += 1
                    else:
                        event['ST_id'] = 'X'  # Nesting events, Id To be determined later
                    # Think about this!!!!!!!!!!!
                    # if 'Theme' not in event and 'Cause' in event:
                    #     pdb.set_trace()
                    # NOTE: need to think about this !!!!!!!!!!!!!!!!!!!
                    # is it valid to have this artificial screening?
                    # predicted event might not have a theme edge, see above
                    # pdb.set_trace()
                    if not 'Theme' in event:
                        continue
                    events.append(event)

        # pdb.set_trace()

        event_cand_stack = [i for i in events if i['ST_id'] == 'X']
        new_events = []
        # print event_id
        while event_cand_stack:
            remove = [False] * len(event_cand_stack)
            # pre_len = len(remove)  #record state of remove mask
            for idx in range(len(event_cand_stack)):
                cur_event = event_cand_stack[idx]
                # NOTE: Need to think about it!!!!!!!!!!!
                # mis-classified event trigger type, the predicted events might say a Gene_expression will also nests other events
                # so this assertion is commented for now
                # assert cur_event['trigger_type'] in REG
                try:
                    theme_target_span = cur_event['Theme']
                except:
                    pdb.set_trace()
                cause_target_span = cur_event.get('Cause', None)

                # pdb.set_trace()
                if cause_target_span:
                    cause_target_ids = eventIdsBySpan.get(cause_target_span, None)
                else:
                    cause_target_ids = None
                theme_target_ids = eventIdsBySpan.get(theme_target_span, None)
                if theme_target_ids is not None and cause_target_ids is not None:
                    # both theme and cause point to (known) child trigger
                    new_combs = [(x, y) for x in theme_target_ids for y in cause_target_ids]
                    for i in range(len(new_combs)):
                        new_event = {}
                        new_event['trigger_type'] = cur_event['trigger_type']
                        new_event['trigger_span'] = cur_event['trigger_span']
                        assert cause_target_span
                        new_event['Cause'] = new_combs[i][1]
                        new_event['Theme'] = new_combs[i][0]
                        new_event['ST_id'] = 'E{}'.format(event_id)
                        eventIdsBySpan[cur_event['trigger_span']].append(new_event['ST_id'])
                        event_id += 1
                        new_events.append(new_event)
                        # the parent events have been found and added
                        remove[idx] = True
                elif theme_target_ids:
                    # only theme point to a (known) child event trigger
                    for i in range(len(theme_target_ids)):
                        new_event = {}
                        new_event['trigger_type'] = cur_event['trigger_type']
                        new_event['trigger_span'] = cur_event['trigger_span']
                        if cause_target_span:
                            new_event['Cause'] = cur_event['Cause']
                        new_event['Theme'] = theme_target_ids[i]
                        new_event['ST_id'] = 'E{}'.format(event_id)
                        eventIdsBySpan[cur_event['trigger_span']].append(new_event['ST_id'])
                        event_id += 1
                        new_events.append(new_event)
                        # the parent events have been found and added
                        remove[idx] = True
                else:
                    # target spans are unknown, meaning the child is not known yet
                    continue
            event_cand_stack = [event_cand_stack[i] for i in range(len(event_cand_stack)) if remove[i] == False]
            if set(remove) == set([False]): #and len(remove) == prev_len:
                # found the root(s), no more update
                break
        all_events = [event for event in events+new_events if event['ST_id'] != 'X']
        # if orig_docid == '10359895':
        writeA2(orig_docid, args, triggerIdBySpan, triggerTypeBySpan, tokenBySpan, all_events)







