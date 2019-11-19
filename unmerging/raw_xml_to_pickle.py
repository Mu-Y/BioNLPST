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
def buildExamplesFromGraph(sentenceGraph, goldGraph=None, structureAnalyzer=None, merge=True, debug=False, test=False):
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
        # Note that this is not used any more - Mu 11/14
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
        if test is False:
            if merge:
                mergeInput = True
                assert sentenceGraph.mergedEntityToDuplicates == None # make sure here the sentenceGraph is unmerged(entities)
                sentenceGraph.mergeInteractionGraph(True)
                assert sentenceGraph.mergedEntityToDuplicates != None # make sure now the sentenceGraph is the merged graph
                # assert goldGraph.mergedEntityToDuplicates == None # make sure gold graph is unmerged
                entities = sentenceGraph.mergedEntities
        else:
            # for test case, there is no need to merge entity(trigger words)
            # just obtain the original annotated entities(Proteins)
            entities = sentenceGraph.entities

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

def flattenCorpus(corpus, test=False):
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
    corpus_docId2OrigId = {}  # 'GE09.d1' --> '10086'

    # init the dict with key as doc_id
    for doc_id in corpus.documentsById.keys():
        corpus_proteinOrigIdBySpan[doc_id] = dict()

        if doc_id not in corpus_docId2OrigId:
            corpus_docId2OrigId[doc_id] = corpus.documentsById[doc_id].get('origId')

    with open('GE09_test_origIdById.pkl', 'wb') as f:
        pickle.dump(corpus_docId2OrigId, f)

    # pdb.set_trace()

    for sentence in corpus.sentences:
        id = sentence.sentenceGraph.sentenceElement.get('id')
        doc_id = '.'.join(id.split('.')[:-1])
        pos_tags = [i.attrib['POS'] for i in sentence.tokenizationElement.findall('token')]
        tokens = [i.attrib['text'] for i in sentence.tokenizationElement.findall('token')]
        offsets = [i.attrib['charOffset'] for i in sentence.tokenizationElement.findall('token')]
        trigger_labels = []
        sentenceGraph = sentence.sentenceGraph
        # if test is False, the mergedEntitiesByOffset key will be a list holding both (merged) event triggers and proteins, the entities will be mergedEntites
        # if test is True, the mergedEntitiesByOffset key will be a list of Proteins only, 'merge' will event not be relavant, entities is also mergedEntities
        tokenByOffset, goldEntitiesByOffset, mergedEntitiesByOffset, entities, sentenceGraph = buildExamplesFromGraph(sentenceGraph, test=test)
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
    if test is True:
        # for test case, there should be no int pair idxs and no int labels
        assert all([i == [] for i in corpus_interaction_idxs])
        assert all([i == [] for i in corpus_interaction_labels])
    return (zip(corpus_ids, corpus_tokens, corpus_pos_tags, corpus_trigger_labels, corpus_interaction_idxs, corpus_interaction_labels, corpus_spans), corpus_proteinOrigIdBySpan)





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
    p.add_argument('--parse', type=str, default='McCC')
    p.add_argument('--tokenization', type=str, default=None)
    p.add_argument('--corpus_file', type=str, default='../reparse_from_installed_GE09/GE09-test.xml')
    p.add_argument('--merge', type=str2bool, default=True)
    args = p.parse_args()

    # structureAnalyzer = StructureAnalyzer()
    # structureAnalyzer.analyze(args.corpus_file)
    # print >> sys.stderr, "--- Structure Analysis ----"
    # print >> sys.stderr, structureAnalyzer.toString()

    print "Loading unmerged gold corpus..."
    corpus = loadCorpus(args.corpus_file, args.parse, tokenization=args.tokenization,
                        removeNameInfo=False, removeIntersentenceInteractionsFromCorpusElements=True,
                        merge=False)
    # sentenceGraph will be merged at the internal call of buildExampleFromGraph function
    # within the flattenCorpus function
    data, protIdBySpan= flattenCorpus(corpus, test=True)

    pdb.set_trace()
    # with open('GE09_test_flat_w-span.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    # with open('GE09_test_protIdBySpan.pkl', 'wb') as f:
    #     pickle.dump(protIdBySpan, f)

    # pdb.set_trace()


