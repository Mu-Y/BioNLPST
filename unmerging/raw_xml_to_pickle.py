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
#multiedges = True

"""
This program reads xml files and convert to flattened tokens, pos_tags, trigger_labels, interaction_labels, etc and picklize it.
Now not-handled-problems:
    0: The construction of gold unmerged graph, there will be duplicated e.g. 'Gene_expression' events. Why? Is this correct?
    1. multi-token triggers? Now take headOffset as the trigger span
    2. triggers with different trigger type not handled. See function flattencorpus for detail
"""

SIMPLE = ['Gene_expression', 'Transcription', 'Protain_catabolism', 'Localization', 'Phosphorylation']
REG = ['Negative_regulation', 'Positive_regulation', 'Regulation']
BIND = ['Binding']

def loadCorpus(corpus, parse, tokenization=None, removeNameInfo=False, removeIntersentenceInteractionsFromCorpusElements=False, merge=True):
    """
    Load an entire corpus through CorpusElements and add SentenceGraph-objects
    to its SentenceElements-objects.
    """
    # Corpus may be in file or not
    if type(corpus) == types.StringType:
        print "Loading corpus file", corpus
    corpusTree = ETUtils.ETFromObj(corpus)
    corpusRoot = corpusTree.getroot()
    # Use CorpusElements-class to access xml-tree
    corpusElements = CorpusElements(corpusRoot, parse, tokenization, tree=corpusTree, removeNameInfo=removeNameInfo, removeIntersentenceInteractions=removeIntersentenceInteractionsFromCorpusElements)
    print str(len(corpusElements.documentsById)) + " documents, " + str(len(corpusElements.sentencesById)) + " sentences"
    # Make sentence graphs
    duplicateInteractionEdgesRemoved = 0
    sentences = []
    # counter = ProgressCounter(len(corpusElements.sentences), "Make sentence graphs")
    # counter.showMilliseconds = True

    dupEntityCnt = 0
    for sentence in tqdm.tqdm(corpusElements.sentences[:]):
        # counter.update(1, "Making sentence graphs ("+sentence.sentence.get("id")+"): ")
        # No tokens, no sentence. No also no dependencies = no sentence.
        # Let's not remove them though, so that we don't lose sentences from input.
        if len(sentence.tokens) == 0: # or len(sentence.dependencies) == 0:
            #corpusElements.sentences.remove(sentence)
            sentence.sentenceGraph = None
            continue
        # for pair in sentence.pairs:  # pairs are all EMPTY lists, i.e. the pairs are not used for example builder in unmerging stage - Mu
        #     # gif-xml defines two closely related element types, interactions and
        #     # pairs. Pairs are like interactions, but they can also be negative (if
        #     # interaction-attribute == False). Sometimes pair-elements have been
        #     # (incorrectly) used without this attribute. To work around these issues
        #     # we take all pair-elements that define interaction and add them to
        #     # the interaction-element list.
        #     isInteraction = pair.get("interaction")
        #     if isInteraction == "True" or isInteraction == None:
        #         sentence.interactions.append(pair) # add to interaction-elements
        #         if pair.get("type") == None: # type-attribute must be explicitly defined
        #             pair.set("type", "undefined")
        # Construct the basic SentenceGraph (only syntactic information)
        graph = SentenceGraph(sentence.sentence, sentence.tokens, sentence.dependencies)
        # Add semantic information, i.e. the interactions
        # Note here the mapInteractions function has already skipped the duplicated interactions
        graph.mapInteractions(sentence.entities + [x for x in sentence.sentence.iter("span")], sentence.interactions)
        graph.interSentenceInteractions = sentence.interSentenceInteractions
        duplicateInteractionEdgesRemoved += graph.duplicateInteractionEdgesRemoved
        sentence.sentenceGraph = graph

        if merge:
            sentence.sentenceGraph.mergeInteractionGraph(True)
            entities = sentence.sentenceGraph.mergedEntities
            dupEntityCnt += len(sentence.sentenceGraph.entities) - len(entities)

        graph.parseElement = sentence.parseElement

        # if len(sentence.sentenceGraph.interactionGraph.edges) + graph.duplicateInteractionEdgesRemoved != len(sentence.sentenceGraph.interactions):
        #     pdb.set_trace()

        #graph.mapEntityHints()
    print "Skipped", duplicateInteractionEdgesRemoved, "duplicate interaction edges in SentenceGraphs"
    print "Duplicate entities skipped",dupEntityCnt
    return corpusElements

def sortInteractionsById(interactions):
        # The order of the interactions affects the order of the unmerging examples, and this
        # affects performance. It's not clear whether this is what really happens, or whether
        # the order of the interactions has some effect on the consistency of the unmerging
        # features (it shouldn't). However, in case it does, this function is left here for now,
        # although it shouldn't be needed at all. In any case the impact is minimal, for GE
        # 53.22 vs 53.28 on the development set.
        pairs = []
        for interaction in interactions:
            pairs.append( (int(interaction.get("id").split(".i")[-1]), interaction) )
        pairs.sort()
        return [x[1] for x in pairs]

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

def eventIsGold(entity, arguments, sentenceGraph, goldGraph, goldEntitiesByOffset, allGoldInteractions):
        offset = entity.get("headOffset")
        if not goldEntitiesByOffset.has_key(offset):
            return False
        eType = entity.get("type")
        goldEntities = goldEntitiesByOffset[offset] # goldEntities is the unmerged entities, a offset can have multiple entities at this position

        # Check all gold entities for a match
        for goldEntity in goldEntities:
            isGold = True

            # The entity type must match
            if goldEntity.get("type") != eType:
                isGold = False
                continue
            goldEntityId = goldEntity.get("id")

            # Collect the gold interactions

            goldInteractions = []
            for goldInteraction in allGoldInteractions: #goldGraph.interactions:
                if goldInteraction.get("e1") == goldEntityId and goldInteraction.get("event") == "True":
                    goldInteractions.append(goldInteraction)
            # Note that the obtained goldInteractions here is only associated with ONE of the gold entity tokens
            # that is, ONE of the unmerged GOLD token, which can have dupilated event trigger labels. - Mu
            # Also, for the case of Reg events where both (Theme, Cause) and (Theme) are annotated as events
            # the goldInteractions here actually contain both cases, because in each case the trigger has different ids
            # So if this goldEntity is (Theme, Cause), another goldEntity(in another loop of course) will be (Theme)
            # So that in that loop the (Theme) will also be considered as a true positive event

            # if entity.get('id') == 'GE09.d554.s5.e18':
            #     pdb.set_trace()

            # Argument count rules
            if len(goldInteractions) != len(arguments): # total number of edges differs
                isGold = False
                continue

            # count number of edges per type
            argTypeCounts = {}
            for argument in arguments:
                argType = argument.get("type")
                if not argTypeCounts.has_key(argType): argTypeCounts[argType] = 0
                argTypeCounts[argType] += 1
            # count number of gold edges per type
            goldTypeCounts = {}
            for argument in goldInteractions:
                argType = argument.get("type")
                if not goldTypeCounts.has_key(argType): goldTypeCounts[argType] = 0
                goldTypeCounts[argType] += 1
            # argument edge counts per type must match
            if argTypeCounts != goldTypeCounts:
                isGold = False
                continue
            # if entity.get('id') == 'GE09.d554.s5.e18':
            #     print 'pass 1 and 2'
            # Exact argument matching
            for argument in arguments: # check all edges
                e1 = argument.get("e1")
                e2 = argument.get("e2")
                if e2 not in sentenceGraph.entitiesById: # intersentence argument, assumed to be correct

                    # print("found inter-sentence event")
                    # pdb.set_trace()
                    found = True
                    continue
                e2Entity = sentenceGraph.entitiesById[e2]
                e2Offset = e2Entity.get("headOffset")
                e2Type = e2Entity.get("type")
                argType = argument.get("type")

                found = False
                for goldInteraction in goldInteractions:
                    if goldInteraction.get("type") == argType:
                        if goldInteraction.get("e2") in goldGraph.entitiesById: # if not, assume this goldInteraction is an intersentence interaction
                            goldE2Entity = goldGraph.entitiesById[goldInteraction.get("e2")]
                            if goldE2Entity.get("headOffset") == e2Offset and goldE2Entity.get("type") == e2Type:
                                found = True
                                # if entity.get('id') == 'GE09.d554.s5.e18':
                                    # print "pass 3"
                                break
                        # else:
                        #     print("")
                        #     pdb.set_trace()
                if found == False: # this edge did not have a corresponding gold edge
                    isGold = False
                    # if entity.get('id') == 'GE09.d554.s5.e18':
                        # print('failed 3')
                    break

            # Event is in gold
            if isGold:
                break

        return isGold
def buildExamplesFromGraph(sentenceGraph, goldGraph=None, structureAnalyzer=None, merge=True, debug=False):
        """
        Build examples for a single sentence. Returns a list of examples.
        See Core/ExampleUtils for example format.
        """
        # self.multiEdgeFeatureBuilder.setFeatureVector(resetCache=True)
        # self.triggerFeatureBuilder.initSentence(sentenceGraph)

        # exampleIndex = 0
        # exampleCounter = defaultdict(dict) # exampleCounter['Binding']: {"tp":xxx, "fp": xxx}
        # undirected = sentenceGraph.dependencyGraph.toUndirected()
        # paths = undirected

        # Get argument order
        # self.interactionLenghts = self.getInteractionEdgeLengths(sentenceGraph, paths)

        # Map tokens to character offsets
        tokenByOffset = {}
        for i in range(len(sentenceGraph.tokens)):
            token = sentenceGraph.tokens[i]
            if goldGraph != None: # check that the tokenizations match
                goldToken = goldGraph.tokens[i]
                assert token.get("id") == goldToken.get("id") and token.get("charOffset") == goldToken.get("charOffset")
            tokenByOffset[token.get("charOffset")] = token.get("id")

        # Map gold entities to their head offsets
        goldEntitiesByOffset = {}
        if goldGraph != None:
            for entity in goldGraph.entities:
                offset = entity.get("headOffset")
                assert offset != None
                if not goldEntitiesByOffset.has_key(offset):
                    goldEntitiesByOffset[offset] = []
                goldEntitiesByOffset[offset].append(entity)

        # if self.styles["no_merge"]:
        #     mergeInput = False
        #     entities = sentenceGraph.entities
        # else:
        # Entered here - Mu
        # The entities here include both named entities(Protein) and event triggers
        # The purpose of merging the entities is to convert the original gold annotation, where
        # a trigger can have multiple trigger annotations, to the merged version.
        if merge:
            mergeInput = True
            assert sentenceGraph.mergedEntityToDuplicates == None # make sure here the sentenceGraph is unmerged(entities)
            sentenceGraph.mergeInteractionGraph(True)
            assert sentenceGraph.mergedEntityToDuplicates != None # make sure now the sentenceGraph is the merged graph
            # assert goldGraph.mergedEntityToDuplicates == None # make sure gold graph is unmerged
            entities = sentenceGraph.mergedEntities

        mergedEntitiesByOffset = {}
        for entity in entities: # entities is mergedEntities
            offset = entity.get('headOffset')
            assert offset != None
            if not mergedEntitiesByOffset.has_key(offset):
                mergedEntitiesByOffset[offset] = []
            mergedEntitiesByOffset[offset].append(entity)


        return tokenByOffset, goldEntitiesByOffset, mergedEntitiesByOffset, entities, sentenceGraph
def printEventStats(triggers):
    """
    params:
        triggers are the unmerged triggers(excluding Proteins)
        this is a list
    """
    eventCounter = {}
    for trigger in triggers:
        eventType = trigger.get('type')
        if eventType not in eventCounter:
            eventCounter[eventType] = 1
        else:
            eventCounter[eventType] += 1
    print "------ Event Statistics ------"
    for k, v in eventCounter.items():
        print "Event Type {}, count {}".format(k, v)
    return eventCounter

def flattenCorpus(corpus):
    """
    given a corpus element, return all flattened tokens, pos_tags, trigger_labels, interactions, interaction_labels
    the returned corpus_proteinOridIdBySpan: corpus_proteinOridIdBySpan[doc_id] = {span:origProtId}, where the span is
    the absolute span of the head token(not the entire protein entity span for multi-token proteins), and origProtId is
    the original protein id in a1 files
    """
    corpus_ids = []
    corpus_tokens = []
    corpus_spans = []
    corpus_pos_tags = []
    corpus_trigger_labels = []
    corpus_interaction_idxs = []
    corpus_interaction_labels = []
    corpus_proteinOrigIdBySpan = defaultdict(dict) # dict of dict, dict[doc_id][abs_span] = proteinOrigId
    # init the dict with key as doc_id
    for doc_id in corpus.documentsById.keys():
        corpus_proteinOrigIdBySpan[doc_id] = dict()

    # pdb.set_trace()

    for sentence in corpus.sentences:
        id = sentence.sentenceGraph.sentenceElement.get('id')
        doc_id = '.'.join(id.split('.')[:-1])
        pos_tags = [i.attrib['POS'] for i in sentence.tokenizationElement.findall('token')]
        tokens = [i.attrib['text'] for i in sentence.tokenizationElement.findall('token')]
        offsets = [i.attrib['charOffset'] for i in sentence.tokenizationElement.findall('token')]
        trigger_labels = []
        sentenceGraph = sentence.sentenceGraph
        tokenByOffset, goldEntitiesByOffset, mergedEntitiesByOffset, entities, sentenceGraph = buildExamplesFromGraph(sentenceGraph)
        sent_start = int(sentence.sentenceGraph.sentenceElement.get('charOffset').split('-')[0])
        # construct trigger labels
        for (offset, token) in zip(offsets, tokens):
            if offset in mergedEntitiesByOffset:
                if len(mergedEntitiesByOffset[offset]) > 1:
                    # if len(set([i.get('type') for i in mergedEntitiesByOffset[offset]])) != len([i.get('type') for i in mergedEntitiesByOffset[offset]]):
                    #     pdb.set_trace()
                    ### A token has multiple-type trigger labels
                    ### TODO: now the way to handle it is to just take the first entity in the dict
                    ### this will lose some events
                    # trigger_type = '--'.join(sorted([i.get('type') for i in mergedEntitiesByOffset[offset]]))
                    trigger_type = mergedEntitiesByOffset[offset][0].get('type')
                else:
                    trigger_type = mergedEntitiesByOffset[offset][0].get('type')

                if trigger_type == 'Protein':
                    # TODO: now only consider Protein, i.e. ingore Entity
                    # meaning that corpus_proteinOrigIdBySpan does not contain Entity
                    span_l = int(offset.split('-')[0])
                    span_r = int(offset.split('-')[1])
                    abs_span = '-'.join([str(span_l+sent_start), str(span_r+sent_start)])
                    corpus_proteinOrigIdBySpan[doc_id][abs_span] = mergedEntitiesByOffset[offset][0].get('origId')

                trigger_labels.append(trigger_type)
            else:
                trigger_labels.append('None')
        # TODO: Change Protein and Entity to 'None'
        # Maybe need to modify later - let's do this in the joint_model.py
        # for i, label in enumerate(trigger_labels):
        #     if label in ['Protein', 'Entity']:
        #         trigger_labels[i] = 'None'


        assert len(pos_tags) == len(tokens)
        assert len(tokens) == len(trigger_labels), pdb.set_trace()

        # a dirty hack to hanle multi-type triggers:
        # just take the first one in the dict. Now each span has only one-type entities
        # this will lose some events, although very few. should be of minor importance
        entities = [mergedEntitiesByOffset[offset][0] for offset in mergedEntitiesByOffset]

        # construct interactions
        interactions = []
        for entity in entities:
            # if entity.get('type') not in ['Protein', 'Entity', 'None']:
                # Only construct interacitons for Non-protein, i.e. triggers
            # if interactions is None:
            #     pdb.set_trace()
            # pdb.set_trace()
            interactions.extend([x for x in sentenceGraph.getOutInteractions(entity, True)])
        if interactions != []:
            # found interactions in current sentence
            pair_spans = [(x[0].get('headOffset'), x[1].get('headOffset')) for x in interactions]
            interaction_labels = [x[2].get('type') for x in interactions]
            pair_idxs = [(offsets.index(x[0]), offsets.index(x[1])) for x in pair_spans]

            corpus_interaction_idxs.append(pair_idxs)
            corpus_interaction_labels.append(interaction_labels)
        else:
            # No interactions appear in current sentence
            # just append empty lists
            corpus_interaction_idxs.append([])
            corpus_interaction_labels.append([])
        corpus_tokens.append(tokens)
        # compute the absolute span of all tokens(w.r.t. the document, not sentence)
        sent_start = int(sentence.sentenceGraph.sentenceElement.get('charOffset').split('-')[0])
        offsets = [(int(i.split('-')[0]), int(i.split('-')[1])) for i in offsets]
        offsets = ['-'.join([str(i[0]+sent_start), str(i[1]+sent_start)]) for i in offsets]
        corpus_spans.append(offsets)
        corpus_pos_tags.append(pos_tags)
        corpus_trigger_labels.append(trigger_labels)
        corpus_ids.append(id)
        # if sentenceGraph.sentenceElement.get('id') == "GE09.d167.s1":
        #     pdb.set_trace()
    return (zip(corpus_ids, corpus_tokens, corpus_pos_tags, corpus_trigger_labels, corpus_interaction_idxs, corpus_interaction_labels, corpus_spans), corpus_proteinOrigIdBySpan)

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
        # find outgoing edges for current tokens
        # out_ints: each element is like ((l_idx, r_idx), int_label)
        out_ints = [i for i in zip(int_idxs, int_labels) if i[0][0] == idx]
        if len(out_ints) == 0:
            continue
        # else, meaning current tokens have outgoing edges
        trigger_label = trigger_labels[idx]
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
        if trigger_label in ['None', 'Protein', 'None', 'Entity']:
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

# def is_terminal(cur_combs, spans, triggerIdBySpan):
#     flags = []
#     for cur_comb in cur_combs:
#         target_spans = [spans[edge[0][1]] for edge in cur_comb]
#         flags.append(all([target_span not in triggerIdBySpan for target_span in target_spans]))
#     # all events associated with one trigger should either 1)all be terminal; 2) all be non-terminal
#     assert len(set(flags)) == 1, pdb.set_trace()
#     return all(flags)

def is_terminal(event):
    targets = []
    for k, v in event.items():
        if k.startswith('Theme') or k == 'Cause':
            targets.append(v)
    return all([i.startswith('T') for i in targets])


# def pretty_print(d, indent=0):
#     for key, value in d.items():
#         print('\t' * indent + str(key))
#         if isinstance(value, dict):
#             pretty_print(value, indent+1)
#         elif isinstance(value, list):
#             for
#             pretty_print()
#         else:
#             print('\t' * (indent+1) + str(value))
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

# class Event():
#     def __init__(self, event_id=None, cur_comb=None, trigger_span = None, trigger_type=None):
#         self.event_id = event_id
#         self.trigger_span = trigger_span
#         self.trigger_type=trigger_type
#     def set_args(self, cur_comb):

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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parse', type=str, default='McCC')
    p.add_argument('--tokenization', type=str, default=None)
    p.add_argument('--corpus_file', type=str, default='../reparse_from_installed_GE09/GE09-devel.xml')
    p.add_argument('--merge', action='store_true')
    p.add_argument('--apply_heu', action='store_true', help='apply the heuristics of taking the longest chain for unmerging')
    p.add_argument('--out_dir', type=str, default='a2_out_dev')
    args = p.parse_args()

    # structureAnalyzer = StructureAnalyzer()
    # structureAnalyzer.analyze(args.corpus_file)
    # print >> sys.stderr, "--- Structure Analysis ----"
    # print >> sys.stderr, structureAnalyzer.toString()

    # print "Loading unmerged gold corpus..."
    # corpus = loadCorpus(args.corpus_file, args.parse, tokenization=args.tokenization,
    #                     removeNameInfo=False, removeIntersentenceInteractionsFromCorpusElements=True,
    #                     merge=False)
    # # sentenceGraph will be merged at the internal call of buildExampleFromGraph function
    # # within the flattenCorpus function
    # data, protIdBySpan= flattenCorpus(corpus)

    # pdb.set_trace()
    # with open('GE09_dev_flat_w-span.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    # with open('GE09_dev_protIdBySpan.pkl', 'wb') as f:
    #     pickle.dump(protIdBySpan, f)
    with open('GE09_dev_flat_w-span.pkl', 'r') as f:
        data = pickle.load(f)
    with open('GE09_dev_protIdBySpan.pkl', 'r') as f:
        protIdBySpan = pickle.load(f)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(len(data)):
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
    for doc_id in protIdBySpan.keys(): # doc_id: GE09.d1
        doc_sents = [i for i in data if '.'.join(i[0].split('.')[:-1]) == doc_id]
        # sort the doc with sentence order
        doc_sents = sorted(doc_sents, key=lambda x: int(x[0].split('.')[-1][1:]))

        doc_prots = protIdBySpan[doc_id]
        if doc_prots.values() == []:
            # doc does not have Proteins
            continue
        orig_docid = doc_prots.values()[0].split('.')[0]
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
        for d in doc_sents:
            trigger_types = d[3]
            spans = d[6]
            tokens = d[1]
            int_idxs = d[4]
            int_labels = d[5]
            arg_combs = d[7]
            assert len(trigger_types) == len(spans)
            assert len(spans) == len(tokens)

            # first write the triggers
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
                            event[arg_role] = target_span
                        else:
                            event[arg_role] = doc_prots[target_span].split('.')[-1]
                    # if d[0] == 'GE09.d235.s11':
                    #     pdb.set_trace()
                    if is_terminal(event):
                        event['ST_id'] = 'E{}'.format(event_id)  # Assign event ids for terminal events
                        eventIdsBySpan[spans[idx]].append(event['ST_id'])
                        event_id += 1
                    else:
                        event['ST_id'] = 'X'  # Nesting events, Id To be determined later
                    events.append(event)

                    # eventsBySpan[spans[idx]].append(event)
        # terminal_events = {}
        # event_id = 1
        # for span, l in eventsBySpan.items():
        #     for dic in l:
        #         if is_terminal(dic):
        #             str_key = ' '.join([':'.join([k,v]) for (k,v) in dic.items()])
        #             terminal_events[str_key] = 'E{}'.format(event_id)
        #             event_id += 1

        # print events
        # pdb.set_trace()

        event_cand_stack = [i for i in events if i['ST_id'] == 'X']
        new_events = []
        # print event_id
        while event_cand_stack:
            remove = [False] * len(event_cand_stack)
            for idx in range(len(event_cand_stack)):
                cur_event = event_cand_stack[idx]
                assert cur_event['trigger_type'] in REG
                theme_target_span = cur_event['Theme']
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
            if set(remove) == set([False]):
                # found the root(s), no more update
                break
        all_events = [event for event in events+new_events if event['ST_id'] != 'X']
        # if orig_docid == '10359895':
        writeA2(orig_docid, args, triggerIdBySpan, triggerTypeBySpan, tokenBySpan, all_events)
            # pdb.set_trace()
                        # for k, v in cur_event.items():
                        #     new_event = {}
                        #     if k.startswith('Theme') or k == 'Cause':
                        #         if cur_event[k] in eventIdsBySpan: # cur_event[k] is targt span
                        #             # child of this event has previously been found

                        #             target_event_ids = eventIdsBySpan[cur_event[k]]






                # print(terminal_events)
                # print(eventsBySpan)
                # pickle.dump(eventsBySpan, open('initial_dict2.pkl', 'w'))
                # pickle.dump(doc_prots, open('doc_prots2.pkl', 'w'))
                # for k, v in eventsBySpan.items():
                #     for i in range(len(v)):
                #         # str_key = ' '.join([':'.join([kk,vv]) for (kk,vv) in v[i].items()])
                #         # v[i] = terminal_events.get(str_key, v[i])
                #         for kk, vv in v[i].items():
                #             if vv.startswith('T'):
                #                 continue
                #             # using get() function is to hanle triggers that are corresponding to inter-sent events
                #             # for inter-sent events, their trigger spans are in triggerIdBySpan but NOT in eventsBySpan
                #             v[i][kk] = eventsBySpan.get(vv, triggerIdBySpan[vv])
                # print(eventsBySpan)
                # # pp = pprint.PrettyPrinter(indent=4)
                # # pp.pprint(eventsBySpan.items())
                # pickle.dump(eventsBySpan, open('test_dict2.pkl', 'w'))
                # pdb.set_trace()


def get_event_ST_string(out_combs, triggerIdBySpan, eventIdsBySpan, doc_prots, spans, trigger_id, event_id):
    '''
    given a list of out_combs like [(((11, 15), 'Theme'),), (((11, 14), 'Theme'),)]
    return a list of event string in BioNLP ST format like ['Theme:T3', 'Theme:T4']
    '''

    cur_event_id = event_id
    eventSTById = defaultdict(dict)
    ST_events = []


    # for source_span in eventIdsBySpan:

    for i, event in enumerate(out_combs):
        for edge in event:
            arg_role = edge[1]
            source_idx = edge[0][0]
            source_span = spans[source_idx]
            target_idx = edge[0][1]
            target_span = spans[target_idx]
            if target_span in triggerIdBySpan:
                # target is a trigger, this is a nested event
                eventIdsBySpan[target_span].append('E{}'.format(cur_event_id))
                cur_event_id += 1
            else:
                assert target_span in doc_prots
                eventSTById['E{}'.format(cur_event_id)][arg_role] = doc_prots[target_span]

    event_ST = {}
    cur_event_id = event_id
    for event in out_combs:
        for edge in event:
            arg_role = edge[1]
            target_idx = edge[0][1]
            target_span = spans[target_idx]
            if target_span in triggerIdBySpan:
                # target is a trigger, this is a nested event
                pass
            else:
                assert target_span in doc_prots[target_span]
                event_ST[arg_role] = doc_prots[target_span]
        event_ST_string = ''
        # sorted, reverse=True to garuantee output in Theme, Cause order
        for arg_role in sorted(event_ST, reverse=True):
            event_ST_string = ' '.join([event_ST_string, '{}:{}'.format(arg_role, event_ST[arg_role])])





