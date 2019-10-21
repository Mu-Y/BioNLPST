import os
import argparse
from collections import defaultdict
import pdb

def get_doc_labelset(doc_path):
    '''
    trigger2label comes from all the Event annotation(Exx)
    span2label comes from all the Trigger annotation(Txx)
    '''
    trigger2label = {}
    span2trigger = defaultdict(list)
    with open(doc_path, 'r') as f:
        for line in f:
            if line.startswith('T'):
                cur_trigger = line.split()[0]
                if line.split()[1] == 'Entity':
                    # skip Entity triggers, because these will not function as
                    # Event triggers under current annotation
                    continue
                cur_span = (line.split()[2], line.split()[3])
                # if cur_span in span2trigger:
                span2trigger[cur_span].append(cur_trigger)
                # else:
                #     span2trigger[cur_span] = cur_trigger

            elif line.startswith('E'):
                raw_label = line.split()[1]
                cur_trigger = raw_label.split(':')[1]
                cur_label = raw_label.split(':')[0]
                assert cur_trigger.startswith('T')
                if cur_trigger in trigger2label:
                    # if this trigger has multiple trigger type annotatoin, append its type label
                    trigger2label[cur_trigger] = trigger2label[cur_trigger] + '---' + cur_label
                else:
                    trigger2label[cur_trigger] = cur_label
    return trigger2label, span2trigger


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--doc_dir', type=str)
    args = p.parse_args()

    filenames = os.listdir(args.doc_dir)
    filenames = [i for i in filenames if i.endswith('.a2')]
    for filename in filenames:
        doc_id = filename.split('.')[0]
        out_filename = doc_id + '.trigger_labels'
        print('generating {}'.format(doc_id))
        trigger2label, span2trigger = get_doc_labelset(os.path.join(args.doc_dir, filename))
        # print(trigger2label)
        # print(span2trigger)
        trigger_labels = []
        for span, trigger_list in span2trigger.items():
            # find ALL associated trigger types for this trigger
            trigger_label = '---'.join([trigger2label[i] for i in trigger_list])
            trigger_label_sorted = '---'.join(sorted(trigger_label.split('---')))
            trigger_labels.append(trigger_label_sorted)
        # pdb.set_trace()
        with open(os.path.join(args.doc_dir, out_filename), 'w') as f:
            for trigger_label in trigger_labels:
                f.write(trigger_label+'\n')
    # print(trigger_labels)
    # trigger2label, span2trigger = get_doc_labelset(args.doc_dir)
    # trigger_labels = []
    # for span, trigger_list in span2trigger.items():
    #     # find ALL associated trigger types for this trigger
    #     trigger_label = '---'.join([trigger2label[i] for i in trigger_list])
    #     trigger_label_sorted = '---'.join(sorted(trigger_label.split('---')))
    #     trigger_labels.append(trigger_label_sorted)
    # print(trigger_labels)





