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
from collections import defaultdict
import argparse

#multiedges = True

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
def buildExamplesFromGraph(sentenceGraph, args, goldGraph=None, structureAnalyzer=None, debug=False):
        """
        Build examples for a single sentence. Returns a list of examples.
        See Core/ExampleUtils for example format.
        """
        # self.multiEdgeFeatureBuilder.setFeatureVector(resetCache=True)
        # self.triggerFeatureBuilder.initSentence(sentenceGraph)

        exampleIndex = 0
        exampleCounter = defaultdict(dict) # exampleCounter['Binding']: {"tp":xxx, "fp": xxx}
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
        mergeInput = True
        assert sentenceGraph.mergedEntityToDuplicates == None # make sure here the sentenceGraph is unmerged(entities)
        sentenceGraph.mergeInteractionGraph(True)
        assert sentenceGraph.mergedEntityToDuplicates != None # make sure now the sentenceGraph is the merged graph
        assert goldGraph.mergedEntityToDuplicates == None # make sure gold graph is unmerged
        entities = sentenceGraph.mergedEntities
        dupEntityCnt = len(sentence.sentenceGraph.entities) - len(entities)
        # self.exampleStats.addValue("Duplicate entities skipped", len(sentenceGraph.entities) - len(entities))
        # pdb.set_trace()
        # if len(sentenceGraph.entities) != len(sentenceGraph.mergedEntities):
        #     pdb.set_trace()

        # Up to here, the merged graph has been built. for one sentence - Mu
        # sentenceGraph_return = sentenceGraph
        # with open('./GE09_train_graph/merged-'+ sentenceGraph.sentenceElement.get('id'), 'wb') as f:
        #     pickle.dump(sentenceGraph, f)
        # with open('./GE09_train_graph/gold-'+ goldGraph.sentenceElement.get('id'), 'wb') as f:
        #     pickle.dump(goldGraph, f)

        # if sentenceGraph.sentenceElement.get('id') == 'GE09.d167.s1':
        #     pdb.set_trace()
        exampleIndex = 0
        for entity in entities: # sentenceGraph.mergedEntities:
            if type(entity) in types.StringTypes: # dummy entity for intersentence interactions
                continue

            eType = entity.get("type")
            assert eType != None, entity.attrib
            eType = str(eType)

            interactions = [x[2] for x in sentenceGraph.getOutInteractions(entity, mergeInput)]
            interactions = sortInteractionsById(interactions)
            interactionCounts = defaultdict(int)
            validInteractionsByType = defaultdict(list)
            for interaction in interactions: # interactions are outgoing edges for the current merged entity - Mu
                if interaction.get("event") != "True":
                    continue
                e1 = sentenceGraph.entitiesById[interaction.get("e1")]
                if interaction.get("e2") in sentenceGraph.entitiesById:
                    e2 = sentenceGraph.entitiesById[interaction.get("e2")]
                    if interaction.get("type") in structureAnalyzer.getValidEdgeTypes(e1.get("type"), e2.get("type")):
                        validInteractionsByType[interaction.get("type")].append(interaction)
                else: # intersentence
                    print("found inter-sent")
                    validInteractionsByType[interaction.get("type")].append(interaction)
                interactionCounts[interaction.get("type")] += 1
            interactionCountString = ",".join([key + "=" + str(interactionCounts[key]) for key in sorted(interactionCounts.keys())])
            # if sentenceGraph.sentenceElement.get('id') == 'GE09.d167.s1':
            #     pdb.set_trace()
            #argCombinations = self.getArgumentCombinations(eType, interactions, entity.get("id"))
            intCombinations = []
            validIntTypeCount = 0
            maxArgCount = 0
            if debug:
                print >> sys.stderr, entity.get("id"), entity.get("type"), "int:" + interactionCountString, "validInt:" + str(validInteractionsByType)
            # pdb.set_trace()
            # if 'Theme' in validInteractionsByType.keys() and 'Cause' in validInteractionsByType:
                # pdb.set_trace()
            for intType in sorted(validInteractionsByType.keys()): # for each argument type the event can have
                validIntTypeCount += 1
                intCombinations.append([])
                minArgs, maxArgs = structureAnalyzer.getArgLimits(entity.get("type"), intType)
                if maxArgs > maxArgCount:
                    maxArgCount = maxArgs
                #if maxArgs > 1: # allow any number of arguments for cases like Binding
                #    maxArgs = len(validInteractionsByType[intType])
                for combLen in range(minArgs, maxArgs+1): # for each valid argument count, get all possible combinations. note that there may be zero-lenght combination
                    for singleTypeArgCombination in combinations(validInteractionsByType[intType], combLen):
                        intCombinations[-1].append(singleTypeArgCombination)
                # e.g. theme:[a,b], cause:[d] = [[(), (d,)], [(a,), (b,)]] - Mu
            # pdb.set_trace()
            # intCombinations now contains a list of lists, each of which has a tuple for each valid combination
            # of one argument type. Next, we'll make all valid combinations of multiple argument types
            if debug:
                print >> sys.stderr, " ", "intCombinations", intCombinations
            argCombinations = combine.combine(*intCombinations)
            if debug:
                print >> sys.stderr, " ", "argCombinations", argCombinations
            for i in range(len(argCombinations)):
                argCombinations[i] = sum(argCombinations[i], ())

            # Up to here, all possible interaction combinations are found - Mu
            # Note this is for each trigger - Mu
            #sum(argCombinations, []) # flatten nested list
            # argCombinations_return = argCombinations
            # pdb.set_trace()
            if debug:
                print >> sys.stderr, " ", "argCombinations flat", argCombinations

            # if len(sentenceGraph.entities) != len(sentenceGraph.mergedEntities) and len(argCombinations) != 0:

            if argCombinations ==[()]:
               if entity.get('type') not in ['Protein', 'Entity']:
                   # meaning that this is a event trigger and also it has no outgoing edges
                   # due to possibbly removed inter-sentence interactions
                   # so skip this to prevent generating a false positive
                   # TODO: need to think about this - how to deal with the inter-sentence interactions? view it as an error?
                   continue

            for argCombination in argCombinations:
                # Originally binary classification
                # if sentenceGraph.sentenceElement.get('id') == 'GE09.d167.s1':
                #     pdb.set_trace()

                # filter out the combinations where the mandatory 'Theme' argument is not presented
                # this can be due to inter-sentence interaction, like the case in the Phosphorylation in GE09.d169.s2
                if 'Theme' not in [i.get('type') for i in argCombination]:
                    continue
                category = None
                if args.apply_alg:
                    if entity.get('type') in ['Negative_regulation', 'Positive_regulation', 'Regulation']:
                        maxArgCombinationLen = max([len(i) for i in argCombinations])
                        if len(argCombination) != maxArgCombinationLen:
                            # meaning that for Regulation classes, there are plausible association of both
                            # (Theme, Cause) and (Theme). And we always choose (Theme, Cause) and ignore (Theme)
                            continue
                    elif entity.get('type') in ['Binding']:
                        maxArgCombinationLen = max([len(i) for i in argCombinations])
                        if len(argCombination) != maxArgCombinationLen:
                            # meaning that for binding events, only take the longest ones.
                            continue
                    elif entity.get('type') in ['Localization', 'Phosphorylation']:
                        maxArgCombinationLen = max([len(i) for i in argCombinations])
                        if len(argCombination) != maxArgCombinationLen:
                            # meaning that for binding events, only take the longest ones.
                            continue
                # else:
                #     continue

                # if not entity.get('type') in ['Gene_expression', 'Transcription', 'Protain_catabolism']:
                #     continue
                # if not entity.get('type') in ['Localization', 'Phosphorylation']:
                #     continue
                # if not entity.get('type') in ['Binding']:
                #     continue
                # if not entity.get('type') in ['Negative_regulation', 'Positive_regulation', 'Regulation']:
                #     continue

                # if entity.get('type') in ['Negative_regulation', 'Positive_regulation', 'Regulation']:
                # if entity.get('type') in ['Binding']:
                if goldGraph != None:
                    isGoldEvent = eventIsGold(entity, argCombination, sentenceGraph, goldGraph, goldEntitiesByOffset, goldGraph.interactions)
                    #if eType == "Binding":
                    #    print argCombination[0].get("e1"), len(argCombination), isGoldEvent
                else:
                    isGoldEvent = False
                # Named (multi-)class
                if isGoldEvent:
#                    category = "zeroArg"
#                    if validIntTypeCount == 1:
#                        category = "singleArg" # event has 0-1 arguments (old simple6)
#                    if validIntTypeCount > 1:
#                        category = "multiType" # event has arguments of several types, 0-1 of each (old Regulation)
#                    if maxArgCount > 1:
#                        category = "multiArg" # event can have 2-n of at least one argument type (old Binding)
                    # if self.styles["binary"]:
                    #     category = "pos"
                    # else: # Entered here, since self.styles["binary"] is None - Mu
                    category = entity.get("type")

                    assert category != None
                else:
                    category = "neg"
                # self.exampleStats.beginExample(category)
                if category != "neg":
                    if category not in exampleCounter:
                        exampleCounter[category] = {"tp":1, "fp":0}
                    else:
                        exampleCounter[category]["tp"] += 1
                else:
                    # the unmerging category generates a False Positive
                    eventType = entity.get("type")
                    if eventType not in exampleCounter:
                        exampleCounter[eventType] = {"tp":0, "fp":1}
                    else:
                        exampleCounter[eventType]["fp"] += 1

                # For debugging - investigate why for single argument event there is false positives
                if category == 'neg' and entity.get("type") == 'Positive_regulation':
                    pdb.set_trace()
                #     print entity.get('id')
                    # if entity.get('id') == 'GE09.d554.s5.e18':
                        # pdb.set_trace()

                #issues = defaultdict(int)
                ## early out for proteins etc.
                #if validIntTypeCount == 0 and entity.get("given") == "True":
                #    self.exampleStats.filter("given-leaf:" + entity.get("type"))
                #    if self.debug:
                #        print >> sys.stderr, " ", category +"("+eType+")", "arg combination", argCombination, "LEAF"
                # TODO: Check this line below, it remove some of the neg classes.
                #elif structureAnalyzer.isValidEntity(entity) or structureAnalyzer.isValidEvent(entity, argCombination, self.documentEntitiesById, noUpperLimitBeyondOne=self.styles["no_arg_count_upper_limit"], issues=issues):
                #    if self.debug:
                #        print >> sys.stderr, " ", category, "arg combination", argCombination, "VALID"
                #    argString = ""
                #    for arg in argCombination:
                #        argString += "," + arg.get("type") + "=" + arg.get("id")
                #    extra = {"xtype":"um","e":entity.get("id"),"i":argString[1:],"etype":eType,"class":category}
                #    extra["allInt"] = interactionCountString
                #    assert type(extra["etype"]) in types.StringTypes, extra
                #    assert type(extra["class"]) in types.StringTypes, category
                #    assert type(extra["i"]) in types.StringTypes, argString
                #    example = self.buildExample(sentenceGraph, paths, entity, argCombination, interactions)
                #    example[0] = sentenceGraph.getSentenceId()+".x"+str(exampleIndex)
                #    example[1] = self.classSet.getId(category)
                #    example[3] = extra
                #    #examples.append( example )
                #    ExampleUtils.appendExamples([example], outfile)
                #    exampleIndex += 1
                #else: # not a valid event or valid entity
                #    if len(issues) == 0: # must be > 0 so that it gets filtered
                #        if not structureAnalyzer.isValidEntity(entity):
                #            issues["INVALID_ENTITY:"+eType] += 1
                #        else:
                #            issues["UNKNOWN_ISSUE_FOR:"+eType] += 1
                #    for key in issues:
                #        self.exampleStats.filter(key)
                #    if self.debug:
                #        print >> sys.stderr, " ", category, "arg combination", argCombination, "INVALID", issues
                #self.exampleStats.endExample()

        #return examples
        # if 'Phosphorylation' in exampleCounter:
        #     pdb.set_trace()
        return exampleIndex, exampleCounter, dupEntityCnt#, sentenceGraph_return, argCombinations_return
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
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parse', type=str, default='McCC')
    p.add_argument('--tokenization', type=str, default=None)
    p.add_argument('--corpus_file', type=str, default='../reparse_from_installed_GE09/GE09-devel.xml')
    p.add_argument('--merge', action='store_true')
    p.add_argument('--apply_alg', action='store_true', help='apply the heuristics of taking the longest chain for unmerging')
    args = p.parse_args()

    structureAnalyzer = StructureAnalyzer()
    structureAnalyzer.analyze(args.corpus_file)
    print >> sys.stderr, "--- Structure Analysis ----"
    print >> sys.stderr, structureAnalyzer.toString()

    corpus = loadCorpus(args.corpus_file, args.parse, tokenization=args.tokenization,
                        removeNameInfo=False, removeIntersentenceInteractionsFromCorpusElements=True,
                        merge=False)

    triggers = [[j for j in i.entities if j.get('type') != 'Protein' and j.get('type') != 'Entity'] for i in corpus.sentences]
    triggers = [i for j in triggers for i in j]
    entities = [[j for j in i.entities if j.get('type') == 'Protein' or j.get('type') == 'Entity'] for i in corpus.sentences]
    entities = [i for j in entities for i in j]
    interactions = [i.interactions for i in corpus.sentences]
    interactions = [i for j in interactions for i in j]
    print "------In original annotation:------"
    print "Total triggers {}, Total named entities {}, Total interactions {}".format(len(triggers), format(len(entities)), format(len(interactions)))
    eventStatsCounter = printEventStats(triggers)

    corpus_gold = loadCorpus(args.corpus_file, args.parse, tokenization=args.tokenization,
                        removeNameInfo=False, removeIntersentenceInteractionsFromCorpusElements=False,
                        merge=False)
    # sentence = [i for i in corpus.sentences if i.sentence.get('id') == 'GE09.d167.s1'][0]
    # pdb.set_trace()

    corpusExampleCounter = defaultdict(dict)
    dupEntityCnt = 0
    for sentence, sentence_gold in zip(corpus.sentences, corpus_gold.sentences):
        sentenceGraph = sentence.sentenceGraph
        # goldGraph = copy.deepcopy(sentence.sentenceGraph)
        goldGraph = sentence_gold.sentenceGraph
        assert sentenceGraph.sentenceElement.get('id') == goldGraph.sentenceElement.get('id')

        _, sentenceExampleCounter, sentDupEntityCnt = buildExamplesFromGraph(sentenceGraph, args, goldGraph=goldGraph, structureAnalyzer=structureAnalyzer, debug=False)
        dupEntityCnt += sentDupEntityCnt
        for k, v in sentenceExampleCounter.items():
            if k not in corpusExampleCounter:
                corpusExampleCounter[k]["tp"] = v["tp"]
                corpusExampleCounter[k]["fp"] = v["fp"]
            else:
                corpusExampleCounter[k]["tp"] += v["tp"]
                corpusExampleCounter[k]["fp"] += v["fp"]
    for k, v in corpusExampleCounter.items():
        if k == 'Protein' or k == 'Entity':
            # skip non-event entities
            continue
        precision = 1.0 * corpusExampleCounter[k]["tp"] / (corpusExampleCounter[k]["tp"] + corpusExampleCounter[k]["fp"])
        recall = 1.0 * corpusExampleCounter[k]["tp"] / eventStatsCounter[k]
        f1 = 2*precision*recall / (precision+recall)
        print "{}:{}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}".format(k, v, precision, recall, f1)

    if args.merge:
        mergedTriggers = [[j for j in i.sentenceGraph.mergedEntities if type(j) not in types.StringTypes and j.get('type') != 'Protein' and j.get('type') != 'Entity'] for i in corpus.sentences]
        n_mergedTriggers = sum([len(i) for i in mergedTriggers])
        mergedInteractions = [i.sentenceGraph.interactionGraph.edges for i in corpus.sentences]
        n_mergedInteractions = sum([len(i) for i in mergedInteractions])
        interSentenceInteractions = [i.sentenceGraph.interSentenceInteractions for i in corpus.sentences]
        n_interSentenceInteractions = sum([len(i) for i in interSentenceInteractions])
        print "------In merged graph:------"
        print "Total merged triggers {}, Total merged interactions {}, Total inter-sentence interactions {}".format(n_mergedTriggers, n_mergedInteractions, n_interSentenceInteractions)
        print "Duplicated entity removed {}".format(dupEntityCnt)

    # pdb.set_trace()


