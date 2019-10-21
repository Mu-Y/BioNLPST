try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os
import argparse
import pickle
from collections import defaultdict
import pdb


class GEDoc:
    def __init__(self, doc_origId):
        self.id = doc_origId
        self.sents = []
        self.token_spans = []
        self.token_labels = []
        self.triggers = []
        self.trigger_head_spans = [] # same as token_spans
        self.trigger_full_spans = [] # full spans include head spans
        self.trigger_types = []
        self.pos_tags = []
    def addSent(self, sent):
        self.sents.append(sent)
    def addToken_span(self, token_span):
        self.token_spans.append(token_span)
    def addTrigger(self, trigger):
        self.triggers.append(trigger)
    def addTrigger_head_span(self, trigger_head_span):
        self.trigger_head_spans.append(trigger_head_span)
    def addTrigger_full_span(self, trigger_full_span):
        self.trigger_full_spans.append(trigger_full_span)
    def addTrigger_type(self, trigger_type):
        self.trigger_types.append(trigger_type)
    def addPos_tag(self, pos_tag):
        self.pos_tags.append(pos_tag)
    def addToken_label(self, token_label):
        self.token_labels.append(token_label)


def trigger_merge_BIO(GEDoc, args):
    '''
    Post processing on trigger labels,
    token_labels are added to GEDoc obj in this step.
    It does:
        1. Handle multi-token and token-splitted triggers
        2. Merge the trigger types with "---" if the very same token has multiple Different trigger labels
        3. Convert to BIO(optionally)
    '''
    # First, handle 2: Merge trigger types with '---'
    for trigger_head_spans, trigger_types, token_spans in zip(GEDoc.trigger_head_spans,GEDoc.trigger_types, GEDoc.token_spans):
        assert len(trigger_head_spans) == len(trigger_types)
        if len(trigger_types) == 0:
            token_labels = ['neg'] *len(token_spans)
            GEDoc.addToken_label(token_labels)
            continue
        headSpan2triggerType = defaultdict(list)
        # trigger_head_spans, trigger_types: sent level
        # trigger_head_span, trigger_type: token level
        for trigger_head_span, trigger_type in zip(trigger_head_spans, trigger_types):
            if trigger_type not in headSpan2triggerType[trigger_head_span]:
                # Ensure *Different* trigger types
                headSpan2triggerType[trigger_head_span].append(trigger_type)
        # generate new trigger types by merging DIfferent trigger types
        # for a same token.
        for k,v in headSpan2triggerType.items():
            if len(v) > 1:
                # merging trigger types
                headSpan2triggerType[k] = '---'.join(sorted(v))
            else:
                headSpan2triggerType[k] = v[0]
        # Update new trigger types
        trigger_types = [headSpan2triggerType[i] for i in trigger_head_spans]
        token_labels = [headSpan2triggerType.get(token_span, 'neg')  for token_span in token_spans]
        GEDoc.addToken_label(token_labels)
        # pdb.set_trace()

    # pdb.set_trace()

    ## TODO: Then, handle 1: multi-token trigger
    ## Currently, only head token is assigned the trigger label, eval is also conducted only on head token
    # doc level
    # assert len(GEDoc.trigger_head_spans) == len(GEDoc.trigger_full_spans)
    # for trigger_head_spans, trigger_full_spans, token_spans in zip(GEDoc.trigger_head_spans, GEDoc.trigger_full_spans, GEDoc.token_spans):
    #     trigger_labels = [headSpan2triggerType.get(token_span, 'neg')  for token_span in token_spans]
    #     pdb.set_trace()
    #     # sent level
    #     assert len(trigger_head_spans) == len(trigger_full_spans)
    #     if len(trigger_full_spans) == 0:
    #         continue

    ## TODO: BIO? However TEES does not do BIO for trigger detection
    # return None

def read_doc_from_xml(doc):
    '''
    read a GEDoc obj from the xml tree
    add its tokens, token_spans, pos_tags, triggers, trigger_spans etc.
    input "doc" is a xml doc element
    '''
    GDoc = GEDoc(doc.attrib['origId'])
    print("Reading {}".format(GDoc.id))
    for sent in doc.findall('sentence'):
        triggers = [entity.attrib['text'] for entity in sent.findall('entity') if entity.get('event')=='True']
        # use headOffset, for multi-token triggers or token-splitted triggers
        # this head span is only part of the entire trigger - Handle Later
        trigger_head_spans = [entity.attrib['headOffset'] for entity in sent.findall('entity') if entity.get('event')=='True']
        trigger_full_spans = [entity.attrib['charOffset'] for entity in sent.findall('entity') if entity.get('event')=='True']
        # this trigger type only corresponds to the head span
        # also it may assign different types to the SAME token - Handles this two later
        trigger_types = [entity.attrib['type'] for entity in sent.findall('entity') if entity.get('event')=='True']
        GDoc.addTrigger(triggers)
        GDoc.addTrigger_head_span(trigger_head_spans)
        GDoc.addTrigger_full_span(trigger_full_spans)
        GDoc.addTrigger_type(trigger_types)
        for tokenization in sent.findall('analyses/tokenization'):
            sent_tokens = [i.attrib['text'] for i in tokenization.findall('token')]
            token_spans = [i.attrib['charOffset'] for i in tokenization.findall('token')]
            pos_tags = [i.attrib['POS'] for i in tokenization.findall('token')]
            assert len(sent_tokens) == len(token_spans)
            assert len(sent_tokens) == len(pos_tags)
            GDoc.addSent(sent_tokens)
            GDoc.addToken_span(token_spans)
            GDoc.addPos_tag(pos_tags)
    return GDoc

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in_file', type=str)
    p.add_argument('--out_file', type=str)
    args = p.parse_args()

    tree = ET.ElementTree(file=args.in_file)
    root = tree.getroot()
    docs = []
    for doc in root.findall('document'):
        GDoc = read_doc_from_xml(doc)
        trigger_merge_BIO(GDoc, args)
        docs.append(GDoc)
    with open(args.out_file, 'wb') as f:
        pickle.dump(docs, f)
    print('pickle file saved as {} for {}.'.format(args.out_file, args.in_file))
    # docs = pickle.load(open(args.out_file, 'rb'))
    # # test_doc = [i for i in docs if i.id == '10092091'][0]
    # test_doc = [i for i in docs if i.id == '9794238'][0]
    # pdb.set_trace()

    # trigger_merge_BIO(test_doc, args)

