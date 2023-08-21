import nltk
import pandas as pd
import copy
import os
import json
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def insert(df: pd.DataFrame, index: int) -> bool:
    if df['eventName'][index] == 'text-insert':
        return True


def delete(df: pd.DataFrame, index: int) -> bool:
    if df['eventName'][index] == 'text-delete':
        return True


def revise(df: pd.DataFrame, index: int) -> bool:
    if df['eventName'][index] == 'text-insert' and post_text_identifier(df, index):
        return True
    if df['eventName'][index] == 'text-delete' and post_text_identifier(df, index):
        return True


def post_text_identifier(df: pd.DataFrame, index: int) -> bool:
    """
    Identify if there is text behind the current cursor

    """
    if df['currentCursor'][index] < len(df['currentDoc'][index]):
        return True


def sentences_with_range(df: pd.DataFrame, index: int) -> list:
    """
    Get the CURRENT event's segmented sentences and their position range.

    """
    sent_positions = []
    doc = df['currentDoc'][index]
    sentences = sent_tokenize(doc)
    for i in range(len(sentences)):
        sent_positions.append((sentences[i], doc.index(sentences[i]),
                               doc.index(sentences[i]) + len(sentences[i]), i))
    return sent_positions  # [(sentence1, start, end, index of doc)...]


def get_gpt_inti_sent(df: pd.DataFrame) -> list:
    """
    Initial GPT all accepted sentences.

    """
    index_sent = []
    for index in range(len(df)):
        if (df['eventName'][index] == 'text-insert') and (df['eventSource'][index] == 'api'):
            index_sent.append(eval(df['textDelta'][index])['ops'][1]['insert'])
    return index_sent


def revised_sentence_identifier(df: pd.DataFrame, index: int) -> list:
    """
    Identify the revised sentence and its location based on the CURRENT cursor

    """
    res = []
    if df['textDelta'].isnull()[index]:
        pass
    else:
        start_cursor = eval(df['textDelta'][index])['ops'][0]['retain']
        for sentence in sentences_with_range(df, index - 1):
            if sentence[1] <= start_cursor < sentence[2]:
                res.append(sentence)  # (sentence, start, end, index of sent)
        for sentence_revised in sentences_with_range(df, index):
            if sentence_revised[1] <= df['currentCursor'][index] < sentence_revised[2]:
                res.append(sentence_revised)  # (sentence, start, end, index of sent)
        return res


def in_gpt_range(df: pd.DataFrame, index: int, gpt_sent: list) -> list:
    """
    Judge if the cursor is in gpt sentence range.

    """
    if revise(df, index) and revised_sentence_identifier(df, index) !=[]:
        sentence = revised_sentence_identifier(df, index)[0][0]
        # update_gpt_sent=gpt_sent_update(df, index, gpt_sent)
        if sentence in gpt_sent:
            return [True, gpt_sent.index(sentence)]
    else:
        return [False, 100]


def gpt_sent_update(df: pd.DataFrame, index: int, gpt_sent: list) -> list:
    """
    Gather the all updated gpt list after each event of revision in gpt cursor range.

    """
    if revise(df, index):
        bool_index = in_gpt_range(df, index, gpt_sent)
        # print("bool_index: ", bool_index)
        update = bool_index[0]
        sent_index = bool_index[1]

        if update:
            # condition 1: what if the user deletes the all content of GPT? that's deep modification
            if delete(df, index) and eval(df['textDelta'][index])['ops'][1]['delete'] == len(gpt_sent[sent_index]):
                gpt_sent[sent_index] = 'None'
            else:
                # condition 2: revise without all deleting
                gpt_sent[sent_index] = revised_sentence_identifier(df, index)[1][0]
        return gpt_sent


def get_revised_gpt_sent(df: pd.DataFrame, initial_gpt_sent: list) -> list:
    """
    Entirely updated gpt-sentence list

    """
    # print(df)
    for index in range(len(df)):
        if revise(df, index):
            initial_gpt_sent = gpt_sent_update(df, index, initial_gpt_sent)
    return initial_gpt_sent


def original_final_gpt_sent_identifier(df: pd.DataFrame, index: int, original_gpt_list: list,
                                       final_gpt_list: list) -> list:
    """
    Generate the sentence pair (original vs finished) by index
    :return: sentence pair
    """

    if revise(df, index):
        # print(get_revised_gpt_sent(df[:index],original_gpt_list))
        bool_index = in_gpt_range(df, index, get_revised_gpt_sent(df[:index], original_gpt_list))
        # print(bool_index)
        # print(get_revised_gpt_sent(df[:index], original_gpt_list))
        update = bool_index[0]
        sent_index = bool_index[1]
        if update:
            return [True,
                    [original_gpt_list[sent_index], get_revised_gpt_sent(df[:index], original_gpt_list)[sent_index]]]
        # TODO make a real recogniser of original and finalised sentences


def modify_low(df: pd.DataFrame, index: int, original_gpt_list: list, final_gpt_list: list) -> bool:
    if original_final_gpt_sent_identifier(df, index, original_gpt_list, final_gpt_list) != None:
        sent_pair = original_final_gpt_sent_identifier(df, index, original_gpt_list, final_gpt_list)[1]
        if sent_pair != None or []:
            sent_pair_unmodify = sent_pair.copy()
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embeddings = model.encode(sent_pair)
            for i in range(len(sent_pair)):
                sent_pair[i] = nltk.pos_tag(word_tokenize(sent_pair[i]))

            sent1_sent2 = list(set(sent_pair[0]) - set(sent_pair[1]))
            sent2_sent1 = list(set(sent_pair[1]) - set(sent_pair[0]))
            symmetric_difference = list(set(sent_pair[0]).symmetric_difference(set(sent_pair[1])))
            # case 1: Punctuation; conjunction; numeral; determiner; existential there; preposition;
            #         foreign word; modal auxiliary; pre-determiner; genitive marker;
            #         particle; symbol; infinitive marker;interjection;verb, past tense;
            #         verb, present participle; verb, past participle; verb, present tense, 3rd person singular;
            #         verb, present tense, not 3rd person singular; WH-determiner; WH-pronoun; WH-pronoun, possessive
            punctuation = ['$', '"', '.', '\'', '(', ')', ',', '--', ':', 'CC', 'CD', 'DT', 'EX',
                           'FW', 'IN', 'LS', 'MD', 'PDT', 'POS', 'SYM', 'TO', 'UH', 'VB', 'VBD',
                           'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', '``']
            for token in symmetric_difference:
                for symbol in punctuation:
                    if symbol in token:
                        return True

            # case 2: Capitalization
            for token in symmetric_difference:
                if token[0][0].isupper():
                    return True

            # case 3: connecting phrases
            conn_phrase = ['in addition to', 'in addition', 'in thi case', 'even if', 'for one thing',
                           'in fact', 'in other words', 'in summary', 'so that', 'in spite of', 'to summarise',
                           'as well as', 'because of', 'as a result', 'due to', 'similar to', 'for example',
                           'for instance']
            for phrase in conn_phrase:
                if phrase not in sent_pair_unmodify[0].lower() and phrase in sent_pair_unmodify[1].lower():
                    return True
                    # case 4: redundancy removal
            if sent2_sent1 == [] and sent1_sent2 != [] and cosine_similarity(embeddings)[0][1] > 0.9:
                return True

            # case 5: Analogy; Specification

            if sent2_sent1 != [] and sent1_sent2 == [] and cosine_similarity(embeddings)[0][1] > 0.9:
                return True
            # Synonym Replacement; Formality Reduction;
            if cosine_similarity(embeddings)[0][1] > 0.85:
                return True


def modify_high(df: pd.DataFrame, index: int, original_gpt_list: list, final_gpt_list: list) -> bool:
    if original_final_gpt_sent_identifier(df, index, original_gpt_list, final_gpt_list) != None:
        sent_pair = original_final_gpt_sent_identifier(df, index, original_gpt_list, final_gpt_list)[1]
        if sent_pair != None or []:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embeddings = model.encode(sent_pair)
            #     for i in range(len(sent_pair)):
            #         sent_pair[i]=nltk.pos_tag(word_tokenize(sent_pair[i]))
            #     sent1_sent2=list(set(sent_pair[0])-set(sent_pair[1]))
            #     sent2_sent1=list(set(sent_pair[1])-set(sent_pair[0]))
            #     symmetric_difference=list(set(sent_pair[0]).symmetric_difference(set(sent_pair[1])))

            # case 1: subject change

            # case 2: extension

            # case 3: truncation

            # case 4: modifier

            # case 5: entity substitution

            # case 6: plot modification TODO hard

            return cosine_similarity(embeddings)[0][1] < 0.75


def behavioural_code_identifier(df: pd.DataFrame, index: int, original_gpt_list: list, final_gpt_list: list) -> list:
    behaviour_seq = []
    # 1.Insert
    if insert(df, index):
        behaviour_seq.append('insert')
    # 2.Delete
    if delete(df, index):
        behaviour_seq.append('delete')
    # 3.Revise
    # insert revise
    if revise(df, index):
        behaviour_seq.append('revise')

    # 4.Relocate
    # TODO: rearrange the position of suggestions

    # TODO: consider the second condition (put suggestion as the start of new paragraph)

    # TODO: insert or remove a sentence before a GPT-suggestion NOTE: distinguish the connecting phrase addition

    # 5.Reflect
    if index / len(df) > 0.8 and post_text_identifier(df, index) and (
            insert(df, index) or delete(df, index) or revise(df, index)):
        behaviour_seq.append('reflect')

    # 6.Seek Suggestion
    if df['eventName'][index] == 'suggestion-get':
        behaviour_seq.append('seekSugg')
        return behaviour_seq
    # 7.Dismiss Suggestion
    # TODO: implement from other samples

    # 8.Accept Suggestion
    if df['eventSource'][index] == 'api' and df['eventName'][index] == 'text-insert':
        behaviour_seq.append('acceptSugg')

    # 9.Modify Suggestion - low
    if modify_low(df, index, original_gpt_list, final_gpt_list):
        behaviour_seq.append('lowModification')

    # # 10.Modify Suggestion - high
    if modify_high(df, index, original_gpt_list, final_gpt_list):
        behaviour_seq.append('highModification')
    return behaviour_seq


def creative_code_identifier(df: pd.DataFrame, index: int, sentences: list, sent_positions: list) -> set:
    creative_seq = set()

    entity = set()
    conditions = []
    location = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
    person = ['B-PER', 'I-PER']
    bes = ['is', 'are', 'were', 'was']
    contrast = ['but', 'however', 'whereas']
    believe = ['believe', 'think', 'wonder']
    goals = ['have to', 'must', 'hope', 'goal']
    # get prompt
    prompt = df['currentDoc'][0]

    prompt_last = sent_tokenize(prompt)[-1]

    # get prompt index
    prompt_index = sentences.index(prompt_last)
    # identify which sentence is within the index of cursor

    sent_index = []
    if df['textDelta'].isnull()[index]:
        pass
    else:
        current_cursor = eval(df['textDelta'][index])['ops'][0]['retain']
        for sentence in sent_positions:
            if sentence[1] <= current_cursor <= sentence[2]:
                sent_index.append(sentence)  # (sentence, start, end, index of sent)

    if len(sent_index) != 0:
        # print('sent_index: ', sent_index)
        sentence = sentences[sent_index[0][3]]
        # segment and tags
        pos_tags = nltk.pos_tag(word_tokenize(sentence))
        # print('pos_tags:', pos_tags)
        # sentence NER
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        sent_NER = nlp(sentence)
        # print('NER:', sent_NER)
        # case 1: Narrative progression
        # 1.1 prompt continuity
        if sent_index[0][3] == prompt_index + 1:
            creative_seq.add('narrativeProg')
            conditions.append(1.1)
        # 1.2 Scene Transition
        if sent_NER != [] and (token['entity'] in location for token in sent_NER):
            creative_seq.add('narrativeProg')
            conditions.append(1.2)
        # 1.3 Climax
        if '!' in sentence:
            creative_seq.add('narrativeProg')
            conditions.append(1.3)
        # 1.4 Character Introduction
        if sent_NER != []:
            for token in sent_NER:
                if (token['entity'] in person) and token['word'] not in entity:
                    entity.add(token['word'])
                    creative_seq.add('narrativeProg')
                    conditions.append(1.4)
        # 1.5 Conclusion
        if sentence == sentences[-1]:
            creative_seq.add('narrativeProg')
            conditions.append(1.5)
        # case 2: Stylistic Devices
        # 2.1 Metonymy

        for token in pos_tags:
            if 'POS' in token:
                creative_seq.add('stylisticDev')
                conditions.append(2.1)
                creative_seq.add('characterization')  # 3.3 Mystery
                conditions.append(3.3)
            # 2.2 Metaphor

            if 'NN' in token and (be in sentence for be in bes):  # TODO
                creative_seq.add('stylisticDev')
                conditions.append(2.2)
                # 3.3 Mystery & 3.4 Dialogue
            if '``' in token:
                creative_seq.add('characterization')
                conditions.append(3.3)
                # 2.3 Contrast/turning point

        if (word in sentence for word in contrast):
            creative_seq.add('stylisticDev')
            conditions.append(2.3)
            # 2.4 Echo
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        for sentence1 in sentences:
            if cosine_similarity(model.encode([sentence, sentence1]))[0][1] > 0.85:
                creative_seq.add('stylisticDev')
                conditions.append(2.4)
                # 2.5 Internal Monologue
        if sent_NER != [] and (token['entity'] in person for token in
                               sent_NER) and (word in sentence for word in believe):
            creative_seq.add('stylisticDev')
            conditions.append(2.5)
            # case 3: Characterization
        # 3.1 Stake

        if (goal in sentence for goal in goals):
            creative_seq.add('characterization')
            conditions.append(3.1)
            # 3.2 Reaction

        # 3.3 Mystery--- see above 2.1
        if '?' in sentence:
            creative_seq.add('characterization')
            conditions.append(3.3)

        # 3.4 Dialogue-- see above

        # 3.5 Shared Emotions Introduction
    # print('creative_seq: ', creative_seq)
    # print('conditions: ', conditions)
    return creative_seq


def argue_code_identifier(df: pd.DataFrame, index: int, sentences: list, sent_positions: list) -> set:
    argue_seq = set()

    # get prompt
    prompt = df['currentDoc'][0]

    prompt_last = sent_tokenize(prompt)[-1]

    # get prompt index
    prompt_index = sentences.index(prompt_last)
    # identify which sentence is within the index of cursor

    sent_index = []
    if df['textDelta'].isnull()[index]:
        pass
    else:
        current_cursor = eval(df['textDelta'][index])['ops'][0]['retain']
        for sentence in sent_positions:
            if sentence[1] <= current_cursor <= sentence[2]:
                sent_index.append(sentence)  # (sentence, start, end, index of sent)

    if sent_index is not None:
        if len(sent_index) != 0:
            sentence = sentences[sent_index[0][3]]
            # segment and tags
            pos_tags = nltk.pos_tag(word_tokenize(sentence))
            # sentence NER
            tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            nlp = pipeline("ner", model=model, tokenizer=tokenizer)
            sent_NER = nlp(sentence)

            # case 1:

            # case 2:

            # case 3:

            # case 4:
    return argue_seq


data = pd.read_csv('../07c50.csv')

original_gpt = get_gpt_inti_sent(data)
# final_gpt = get_revised_gpt_sent(data, original_gpt)
doc = data['currentDoc'][len(data) - 1]
sentences = nltk.sent_tokenize(doc)

sent_positions = []
for i in range(len(sentences)):
    sent_positions.append((sentences[i], doc.index(sentences[i]),
                           doc.index(sentences[i]) + len(sentences[i]), i))

# for i in range(len(data)):
#     print(behavioural_code_identifier(data,i, original_gpt, final_gpt), creative_code_identifier(data, i,sentences, sent_positions),i)

cols = ["eventName",
        "currentDoc",
        "eventSource",
        "insert",
        "delete",
        "revise",
        "relocate",
        "reflect",
        "seekSugg",
        "acceptSugg",
        "dismissSugg",
        "lowModification",
        "highModification",
        "narrativeProg",
        "stylisticDev",
        "characterization",
        "argument",
        "evidence",
        "counterArg",
        "rebuttal"]
df2 = pd.DataFrame(columns=cols)
new_row = set()

for i in range(len(data)):
    if data['eventName'][i] == 'text-insert':
        if eval(data['textDelta'][i])['ops'][1]['insert'] == '\n':
            continue

    print(i)
    in_which_gpt = original_final_gpt_sent_identifier(data, i, original_gpt_list=original_gpt, final_gpt_list=None)
    code_list = []
    # print(i, in_which_gpt, data['eventName'][i])
    if in_which_gpt is not None and in_which_gpt[0] or (
            data['eventName'][i] == 'text-insert' and data['eventSource'][i] == 'api') or (
            data['eventName'][i] == 'suggestion-get'):
        final_gpt = get_revised_gpt_sent(data[:i], original_gpt)
        code_list = behavioural_code_identifier(data, i, original_gpt, final_gpt) + list(
            creative_code_identifier(data, i, sentences, sent_positions))
        # Columns to fill with the same content from the existing dataset (data)
        given_list_2 = ['eventName', 'currentDoc', 'eventSource']
        # Accessing the values from the last row of the existing dataset (data) for the specified columns
        values_to_copy = data[given_list_2].iloc[i].to_dict()
        values_to_copy['currentDoc'] = values_to_copy['currentDoc'][len(data['currentDoc'][0]):]
        # Creating the new row with the specified values and 1 in the given columns
        new_row = {col: values_to_copy[col] if col in given_list_2 else (1 if col in code_list else 0) for col in
                   df2.columns}

        # Appending the new row to the new DataFrame (df2)
        df2 = df2.append(new_row, ignore_index=True)

# df2.fillna(0, inplace=True)
df2.to_csv('sample_07c50_creative.csv')
