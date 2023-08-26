import nltk
import pandas as pd
import copy
from nltk.tokenize import sent_tokenize
import os
import json
import collections

def find_writing_sessions(dataset_dir):
    paths = [
        os.path.join(dataset_dir, path)
        for path in os.listdir(dataset_dir)
        if path.endswith('jsonl')
    ]
    return paths


def read_writing_session(path):
    events = []
    with open(path, 'r') as f:
        for event in f:
            events.append(json.loads(event))
    return events

def apply_ops(doc, mask, ops, source):
    original_doc = doc
    original_mask = mask

    new_doc = ''
    new_mask = ''
    for i, op in enumerate(ops):

        # Handle retain operation
        if 'retain' in op:
            num_char = op['retain']

            retain_doc = original_doc[:num_char]
            retain_mask = original_mask[:num_char]

            original_doc = original_doc[num_char:]
            original_mask = original_mask[num_char:]

            new_doc = new_doc + retain_doc
            new_mask = new_mask + retain_mask

        # Handle insert operation
        elif 'insert' in op:
            insert_doc = op['insert']

            insert_mask = 'U' * len(insert_doc)  # User
            if source == 'api':
                insert_mask = 'A' * len(insert_doc)  # API

            if isinstance(insert_doc, dict):
                if 'image' in insert_doc:
                    print('Skipping invalid object insertion (image)')
                else:
                    print('Ignore invalid insertions:', op)
                    # Ignore other invalid insertions
                    # Debug if necessary
                    pass
            else:
                new_doc = new_doc + insert_doc
                new_mask = new_mask + insert_mask

        # Handle delete operation
        elif 'delete' in op:
            num_char = op['delete']

            if original_doc:
                original_doc = original_doc[num_char:]
                original_mask = original_mask[num_char:]
            else:
                new_doc = new_doc[:-num_char]
                new_mask = new_mask[:-num_char]

        else:
            # Ignore other operations
            # Debug if necessary
            print('Ignore other operations:', op)
            pass

    final_doc = new_doc + original_doc
    final_mask = new_mask + original_mask
    return final_doc, final_mask

def populate_currentdoc(events):
    prompt = events[0]['currentDoc'].strip()

    text = prompt
    mask = 'P' * len(prompt)  # Prompt

    events_with_currentdoc = copy.deepcopy(events)
    for i, event in enumerate(events):
        if 'ops' in event['textDelta']:
            ops = event['textDelta']['ops']
            source = event['eventSource']
            text, mask = apply_ops(text, mask, ops, source)
        events_with_currentdoc[i]['currentDoc'] = copy.deepcopy(text)

    return events_with_currentdoc

def get_text_and_mask(events, event_id, remove_prompt=True):
    prompt = events[0]['currentDoc'].strip()

    text = prompt
    mask = 'P' * len(prompt)  # Prompt
    for event in events[:event_id]:
        if 'ops' not in event['textDelta']:
            continue
        ops = event['textDelta']['ops']
        source = event['eventSource']
        text, mask = apply_ops(text, mask, ops, source)

    if remove_prompt:
        if 'P' not in mask:
            print('=' * 80)
            print('Could not find the prompt in the final text')
            print('-' * 80)
            print('Prompt:', prompt)
            print('-' * 80)
            print('Final text:', text)
        else:
            end_index = mask.rindex('P')
            text = text[end_index + 1:]
            mask = mask[end_index + 1:]

    return text, mask


def identify_author(mask):
    if 'P' in mask:
        return 'prompt'
    elif 'U' in mask and 'A' in mask:
        return 'user_and_api'
    elif 'U' in mask and 'A' not in mask:
        return 'user'
    elif 'U' not in mask and 'A' in mask:
        return 'api'
    else:
        raise RuntimeError(f'Could not identify author for this mask: {mask}')


def classify_sentences_by_author(text, mask):
    sentences_by_author = collections.defaultdict(list)
    for sentence_id, sentence in enumerate(sent_tokenize(text.strip())):
        if sentence not in text:
            print(f'Could not find sentence in text: {sentence}')
            continue
        index = text.index(sentence)
        sentence_mask = mask[index:index + len(sentence)]
        author = identify_author(sentence_mask)
        sentences_by_author[author].append({
            'sentence_id': sentence_id,
            'sentence_mask': sentence_mask,
            'sentence_author': author,
            'sentence_text': sentence,
        })
    return sentences_by_author


creative = pd.read_csv('matedataSurvey/creative_metadata.csv')['session_id'].to_list()
argument = pd.read_csv('matedataSurvey/argumentative_metadata.csv')['session_id'].to_list()
dataset_dir = './coauthor-v1.0/'
paths = find_writing_sessions(dataset_dir)
for path in paths:
    if path[16:48] in creative:
        print('creative: ', path[16:48])
        events = read_writing_session(path)
        events_with_currentdoc = populate_currentdoc(events)
        session = pd.DataFrame(events_with_currentdoc)
        session.to_csv('./{}/{}.csv'.format('creative', path[16:48]))
    if path[16:48] in argument:
        print('argument: ', path[16:48])
        events = read_writing_session(path)
        events_with_currentdoc = populate_currentdoc(events)
        session = pd.DataFrame(events_with_currentdoc)
        session.to_csv('./{}/{}.csv'.format('argumentative', path[16:48]))

