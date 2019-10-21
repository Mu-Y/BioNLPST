import pickle
from SimpleGraph import Graph
import Utils.ElementTreeUtils as ETUtils
from Utils.ProgressCounter import ProgressCounter
from Utils.InteractionXML.CorpusElements import CorpusElements
import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
# import Utils.Range as Range
import types
import copy
import re
import pdb
import pickle
import argparse

#multiedges = True

def loadCorpus(corpus, parse, tokenization=None, removeNameInfo=False, removeIntersentenceInteractionsFromCorpusElements=True):
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
    for sentence in corpusElements.sentences[:]:
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
        graph.mapInteractions(sentence.entities + [x for x in sentence.sentence.iter("span")], sentence.interactions)
        graph.interSentenceInteractions = sentence.interSentenceInteractions
        duplicateInteractionEdgesRemoved += graph.duplicateInteractionEdgesRemoved
        sentence.sentenceGraph = graph

        graph.parseElement = sentence.parseElement

        #graph.mapEntityHints()
    print "Skipped", duplicateInteractionEdgesRemoved, "duplicate interaction edges in SentenceGraphs"
    return corpusElements

if __name == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parse', type=str, default='McCC')
    p.add_argument('--tokenization', type=str, default=None)
    p.add_argument('--corpus_file', type=str, default='../reparse_from_installed_GE09/GE09-train.xml')
    args = p.parse_args()

    corpus = loadCorpus(args.corpus_file, args.parse, tokenization=args.tokenization, removeNameInfo=False, removeIntersentenceInteractionsFromCorpusElements=True)
    pdb.set_trace()

