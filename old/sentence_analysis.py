import sys

import nltk
import pandas as pd
import copy
import os
import json
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline
import stanza
import time


def insert(df: pd.DataFrame, index: int) -> bool:
    return df['eventName'][index] == 'text-insert'


def delete(df: pd.DataFrame, index: int) -> bool:
    return df['eventName'][index] == 'text-delete'


def revise(df: pd.DataFrame, index: int) -> bool:
    return post_text_identifier(df, index) and (insert(df, index) or delete(df, index))


def post_text_identifier(df: pd.DataFrame, index: int) -> bool:
    """
    Identify if there is text behind the current cursor

    """
    return df['currentCursor'][index] < len(df['currentDoc'][index])


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


def get_init_sent(df: pd.DataFrame) -> list:
    # tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    # model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")
    # nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
    endingmark = ('.', '!', '?')
    prompt = df['currentDoc'][0]
    init_sent_author = []
    init_sentences = sent_tokenize(prompt)
    for sentence in init_sentences:
        init_sent_author.append([sentence, 'prompt'])

    for index in range(1, len(df)):
        # print(index)
        # new_sentences = sent_tokenize(df['currentDoc'][index])
        last_sentence = sentences_with_range(df, index)[-1]
        # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # embeddings = model.encode([init_sentences[-1],last_sentence])
        # print(last_sentence)
        # if df['textDelta'].isnull()[index]:
        #     continue
        # case 1, add the initial state of each last sentence TODO final condition to set up.
        if last_sentence[0].endswith(endingmark) \
                and (last_sentence[0] not in init_sent_author[-1]) \
                and revise(df, index) == False \
                and df['eventName'][index] != 'cursor-backward' \
                and df['eventName'][index] != 'cursor-forward' \
                and df['currentCursor'][index] <= last_sentence[2] + 1:
            if df['eventSource'][index] == 'api':
                init_sent_author.append([last_sentence[0], 'api'])
            else:
                init_sent_author.append([last_sentence[0], 'user'])
        else:
            continue
        # # case 2, insert the new sentence, add it to the initial list as well.
        # if revise(df, index):
        #     if len(sent_tokenize(df['currentDoc'])[i]) > len(sent_tokenize(df['currentDoc'])[i-1]):

    return init_sent_author


def sent_update(df: pd.DataFrame, index: int, sent_list: list) -> list:
    """
    Gather the all updated gpt list after each event of revision in gpt cursor range.

    """
    if revise(df, index):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        sent_from_last = sent_tokenize(df['currentDoc'][index - 1])
        sent_current = sent_tokenize(df['currentDoc'][index])
        last_current = list(set(sent_from_last) - set(sent_current))
        current_last = list(set(sent_current) - set(sent_from_last))
        symmetric_difference = list(set(sent_current).symmetric_difference(set(sent_from_last)))
        # print(last_current, current_last)
        # case 1 remove the sentence
        if len(symmetric_difference) != 2:
            ranking = []
            seq = []
            for i in range(len(current_last)):
                for j in range(len(last_current)):
                    embedding = model.encode([current_last[i], last_current[j]])
                    ranking.append([cosine_similarity(embedding)[0][1]])
                    seq.append([i, j])
            if len(last_current) > len(current_last):
                indexes = seq[ranking.index(max(ranking))]

                for i in range(len(last_current)):
                    for j in range(len(sent_list)):
                        if last_current[i] in sent_list[j] and i == indexes[1]:
                            sent_list[j][0] = current_last[indexes[0]]

                        if last_current[i] in sent_list[j] and i != indexes[1]:
                            sent_list[j][0] = 'None'

                # for i in range(len(sent_list)):
                #     if last_current[indexes[1]] in sent_list[i]:
                #         sent_list[i][0] = current_last[indexes[0]]
                #     else:
                #         sent_list[i][0] = 'None'
                return sent_list

            if last_current != [] and current_last == []:
                # print(index)
                # print(last_current)
                # print(sent_from_last)
                for element in last_current:
                    for i in range(len(sent_list)):
                        if element in sent_list[i]:
                            sent_list[i][0] = 'None'
                            return sent_list

            if len(current_last) > len(last_current):
                pass

        # TODO case 2 has merged to the initial sentence list in "get_init_sent"
        # # case 2 add new sentence (relocate) Flaw: until a complete new sentence comes out, the new adds always are tokenized in the later sentence.
        # if current_last !=[] and last_current ==[]:
        #     later_sentence= sent_current[sent_current.index(current_last)+1]
        #     sent_index=sent_list.index(later_sentence)
        #     sent_list.insert(sent_index-1,current_last[0])
        #     return sent_list  # (relocate)
        # case 3 revise
        if len(symmetric_difference) == 2:
            # print(symmetric_difference)
            for i in range(len(sent_list)):
                if symmetric_difference[0] in sent_list[i]:
                    sent_list[i][0] = symmetric_difference[1]
                    return sent_list
                if symmetric_difference[1] in sent_list[i]:
                    sent_list[i][0] = symmetric_difference[0]
                    # print(sent_list)
                    return sent_list
    return sent_list


def get_revised_sent(df: pd.DataFrame, initial_sent: list) -> list:
    """
    Entirely updated gpt-sentence list

    """
    # print(df)
    repo = initial_sent.copy()
    for index in range(len(df)):
        if revise(df, index):
            repo = sent_update(df, index, repo)
    return repo


def original_final_revise_sent_identifier(df: pd.DataFrame, index: int, original_list: list,
                                          final_list: list) -> list:
    """
    Generate the sentence pair (original vs finished) by index
    :return: sentence pair
    """

    if revise(df, index):
        preindex_sent_list = get_revised_sent(df[:index - 1], original_list)
        index_sent_list = get_revised_sent(df[:index], original_list)

        for i in range(len(preindex_sent_list)):
            if preindex_sent_list[i] != index_sent_list[i]:
                return [original_list[i], final_list[i]]


def relocate(df: pd.DataFrame, index: int, original_list: list, final_list: list) -> bool:
    if original_final_revise_sent_identifier(df, index, original_list, final_list) != None:
        sent_pair = original_final_revise_sent_identifier(df, index, original_list, final_list)[1]
        if sent_pair != []:
            if sent_pair[2] == 'None':
                return True


def compose(df: pd.DataFrame, index: int) -> bool:
    if revise(df, index) == False:
        if df['eventName'][index] == 'text-insert':
            return True


def modify_low(df: pd.DataFrame, index: int, original_list: list, final_list: list) -> bool:
    if original_final_revise_sent_identifier(df, index, original_list, final_list) != None:
        sent_pair = original_final_revise_sent_identifier(df, index, original_list, final_list)[1]
        if sent_pair != []:
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
            #         particle; symbol; infinitive marker; interjection;verb, past tense;
            #         verb, present participle; verb, past participle; verb, present tense, 3rd person singular;
            #         verb, present tense, not 3rd person singular; WH-determiner; WH-pronoun; WH-pronoun, possessive
            punctuation_tense_infinitive = ['$', '"', '.', '\'', '(', ')', ',', '--', ':', 'CC', 'CD', 'DT', 'EX',
                                            'FW', 'IN', 'LS', 'MD', 'PDT', 'POS', 'SYM', 'TO', 'UH', 'VB', 'VBD',
                                            'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', '``']
            for token in symmetric_difference:
                for symbol in punctuation_tense_infinitive:
                    if symbol in token:
                        return True

            # case 2: capitalization
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
            if sent2_sent1 == [] and sent1_sent2 != [] and cosine_similarity(embeddings)[0][1] >= 0.8:
                return True

            # case 5: Analogy; Specification

            if sent2_sent1 != [] and sent1_sent2 == [] and cosine_similarity(embeddings)[0][1] >= 0.8:
                return True
            # Synonym Replacement; Formality Reduction;
            if cosine_similarity(embeddings)[0][1] >= 0.8:
                return True


def modify_high(df: pd.DataFrame, index: int, original_list: list, final_list: list) -> bool:
    entity = set()
    location = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
    modifiers = ['RB', 'JJ']
    persons = ['B-PER', 'I-PER']

    if original_final_revise_sent_identifier(df, index, original_list, final_list) != None:
        sent_pair = original_final_revise_sent_identifier(df, index, original_list, final_list)[1]
        if sent_pair != None or []:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embeddings = model.encode(sent_pair)
            # for i in range(len(sent_pair)):
            #     sent_pair[i] = nltk.pos_tag(word_tokenize(sent_pair[i]), tagset='universal')
            # sent1_sent2 = list(set(sent_pair[0]) - set(sent_pair[1]))
            # sent2_sent1 = list(set(sent_pair[1]) - set(sent_pair[0]))
            # symmetric_difference = list(set(sent_pair[0]).symmetric_difference(set(sent_pair[1])))
            # if cosine_similarity(embeddings)[0][1] < 0.8:
            #     # case 1: subject change TODO change it's name
            #     if (person in symmetric_difference for person in persons):
            #         return True
            #     # case 2: extension
            #     if len(sent1_sent2) == 0 and len(sent2_sent1) != 0:
            #         return True
            #     # case 3: truncation
            #     if len(sent1_sent2) != 0 and sent2_sent1 == 0:
            #         return True
            #     # case 4: modifier TODO
            #
            #     # case 5: entity substitution also use entity set to identify
            #
            #     # case 6: plot modification TODO hard

            return cosine_similarity(embeddings)[0][1] < 0.8


def behavioural_code_identifier(df: pd.DataFrame, index: int, original_list: list, final_list: list) -> list:
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
    if relocate(df, index, original_list, final_list):
        behaviour_seq.append('relocate')
    # 5.Reflect
    if index / len(df) > 0.9 and post_text_identifier(df, index) and (
            insert(df, index) or delete(df, index) or revise(df, index)):
        behaviour_seq.append('reflect')

    # 6.Seek Suggestion
    if df['eventName'][index] == 'suggestion-get':
        behaviour_seq.append('seekSugg')
        return behaviour_seq
    # 7.Dismiss Suggestion
    # TODO: implement from other samples
    if df['eventName'][index] == 'suggestion-close' and df['eventSource'][index] == 'user':
        behaviour_seq.append('dismissSugg')

    if df['eventName'][index] == 'suggestion-get' and df['eventName'][index + 1] != 'suggestion-open':
        behaviour_seq.append('dismissSugg')
    # 8.Accept Suggestion
    if df['eventSource'][index] == 'api' and df['eventName'][index] == 'text-insert':
        behaviour_seq.append('acceptSugg')

    # 9.Modify Suggestion - low
    if modify_low(df, index, original_list, final_list):
        behaviour_seq.append('lowModification')

    # # 10.Modify Suggestion - high
    if modify_high(df, index, original_list, final_list):
        behaviour_seq.append('highModification')

    # 11.Compose
    if compose(df, index):
        behaviour_seq.append('compose')

    if df['currentTemperature'][index] < 0.5:
        behaviour_seq.append('lowTemp')

    if df['currentTemperature'][index] > 0.5:
        behaviour_seq.append('highTemp')
    return behaviour_seq


def execute(file_name: str, genre: str):
    data = pd.read_csv(file_name)

    original = get_init_sent(data)
    original1 = copy.deepcopy(original)
    final = get_revised_sent(data, original1)

    cols = ["eventName",
            "currentDoc",
            "eventSource",
            "insert",
            "delete",
            "revise",
            "reviseSugg",
            "relocate",
            "reflect",
            "seekSugg",
            "acceptSugg",
            "dismissSugg",
            "lowModification",
            "highModification",
            "compose",
            "highTemp",
            "lowTemp"]
    df2 = pd.DataFrame(columns=cols)
    new_row = set()
    ops = ['text-insert', 'text-delete', 'suggestion-get']
    for i in range(len(data)):

        if data['eventName'][i] == 'text-insert':
            if eval(data['textDelta'][i])['ops'][1]['insert'] == '\n':
                continue

        # print(i, in_which_gpt, data['eventName'][i])
        if data['eventName'][i] in ops or (
                data['eventName'][i] == 'suggestion-close' and data['eventSource'][i] == 'user'):
            # final = get_revised_sent(data, original)
            # print('final: ', final)
            code_list = behavioural_code_identifier(data, i, original, final)
            # Columns to fill with the same content from the existing dataset (data)
            given_list_2 = ['eventName', 'currentDoc', 'eventSource']
            # Accessing the values from the last row of the existing dataset (data) for the specified columns
            values_to_copy = data[given_list_2].iloc[i].to_dict()
            values_to_copy['currentDoc'] = values_to_copy['currentDoc']  # [len(data['currentDoc'][0]):]
            # Creating the new row with the specified values and 1 in the given columns
            new_row = {col: values_to_copy[col] if col in given_list_2 else (1 if col in code_list else 0) for col in
                       df2.columns}

            # Appending the new row to the new DataFrame (df2)
            df2 = df2.append(new_row, ignore_index=True)
            print(i)
    df2.fillna(0, inplace=True)
    df2.to_csv('{}_{}.csv'.format(file_name, genre))


def main():

    execute('../a068c.csv', 'creative')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
