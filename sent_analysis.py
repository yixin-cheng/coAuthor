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
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline
import stanza
import time
import collections
from sklearn.metrics.pairwise import linear_kernel


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
    last_end = 0  # To keep track of the last found index
    for i in range(len(sentences)):
        last_end = 0  # To keep track of the last found index
        for i in range(len(sentences)):
            start = doc.index(sentences[i], last_end)
            end = start + len(sentences[i])
            sent_positions.append((sentences[i], start, end, i))
            last_end = end  # Update the last_end for the next search
    return sent_positions  # [(sentence1, start, end, index of doc)...]


def is_valid_sentence(last_sentence, init_sent_author, df, index):
    # tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    # model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")
    # nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
    delimiters = ('.', '!', '?')
    # Check if the last sentence ends with an ending mark
    valid_ending = last_sentence[0].endswith(delimiters)

    # Check if the last sentence is not the same as the last sentence from the prompt
    not_in_prompt = last_sentence[0] not in init_sent_author[-1]

    # Check if the last sentence is not a revision and not a cursor movement event
    not_revision = not revise(df, index)
    not_cursor_back = df['eventName'][index] != 'cursor-backward'
    not_cursor_forward = df['eventName'][index] != 'cursor-forward'

    # Check if the cursor is at or just after the end of the sentence
    cursor_position = df['currentCursor'][index] <= last_sentence[2] + 1

    return valid_ending and not_in_prompt and not_revision and not_cursor_back and not_cursor_forward and cursor_position


def get_init_sent(df: pd.DataFrame) -> list:
    # Initial sentences from the prompt
    prompt = df['currentDoc'][0]
    init_sentences = sent_tokenize(prompt)
    init_sent_author = [[sentence, 'prompt', prompt.index(sentence), prompt.index(sentence) + len(sentence)]
                        for sentence in init_sentences]

    for index in range(1, len(df)):
        last_sentence = sentences_with_range(df, index)[-1]
        if is_valid_sentence(last_sentence, init_sent_author, df, index):
            source = 'api' if df['eventSource'][index] == 'api' else 'user'
            init_sent_author.append([last_sentence[0], source, last_sentence[1], last_sentence[2]])
    return init_sent_author


def compute_similarity(sentences):
    """
    Compute pairwise cosine similarity using TF-IDF.
    This version is robust against empty documents and those with only stop words.
    """
    vectorizer = TfidfVectorizer(stop_words=None)  # Do not remove stop words
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Handle the case where TF-IDF cannot produce a valid vocabulary
    if tfidf_matrix.shape[1] == 0:
        return [[1.0 if i == j else 0.0 for j in range(len(sentences))] for i in range(len(sentences))]

    return cosine_similarity(tfidf_matrix)


def most_similar_pair(sent_a, sent_b):
    """
    Return the indices of the most similar pair of sentences.
    """
    # print(sent_a)
    # print(sent_b)
    combined_sentences = sent_a + sent_b
    # print(combined_sentences)
    similarity_matrix = compute_similarity(combined_sentences)
    # print(similarity_matrix)
    max_similarity = -1
    pair = (-1, -1)
    for i in range(len(sent_a)):
        for j in range(len(sent_b)):
            similarity = similarity_matrix[i, len(sent_a) + j]
            if similarity > max_similarity:
                max_similarity = similarity
                pair = (i, j)
    return pair


# Re-defining the functions

def sent_update(df: pd.DataFrame, index: int, sent_list: list) -> list:
    if revise(df, index):
        last_sent_range = sentences_with_range(df, index - 1)
        current_sent_range = sentences_with_range(df, index)

        sent_from_last = [element[0] for element in last_sent_range]
        sent_current = [element[0] for element in current_sent_range]

        symmetric_difference = list(set(sent_current).symmetric_difference(set(sent_from_last)))

        # case 1 merge/remove the sentence
        if len(symmetric_difference) != 2:
            last_diff = list(set(sent_from_last) - set(sent_current))
            current_diff = list(set(sent_current) - set(sent_from_last))

            # 1.1 remove
            if last_diff and not current_diff:
                for element in last_diff:
                    for i, sent_element in enumerate(sent_list):
                        if element in sent_element:
                            sent_list[i][0] = 'None'
                            sent_list[i][2] = -1
                            sent_list[i][3] = -1
                            return sent_list

            # 1.2 merge
            if last_diff and current_diff:
                i, j = most_similar_pair(current_diff, last_diff)
                for k, sent_element in enumerate(sent_list):
                    if last_diff[j] in sent_element:
                        sent_list[k][0] = current_diff[i]
                        for element in current_sent_range:
                            if current_diff[i] in element:
                                sent_list[k][2] = element[1]
                                sent_list[k][3] = element[2]
                # Set unmatched sentences from last_diff to 'None'
                for l in range(len(last_diff)):
                    if l != j:
                        for k, sent_element in enumerate(sent_list):
                            if last_diff[l] in sent_element:
                                sent_list[k][0] = 'None'
                                sent_list[k][2] = -1
                                sent_list[k][3] = -1
                return sent_list

        # case 2 is already handled in the get_init_sent function

        # case 3 revise
        if len(symmetric_difference) == 2:
            # print(symmetric_difference)
            for i in range(len(sent_list)):
                if symmetric_difference[0] in sent_list[i]:
                    sent_list[i][0] = symmetric_difference[1]
                    for element in current_sent_range:
                        if symmetric_difference[1] in element:
                            sent_list[i][2] = element[1]
                            sent_list[i][3] = element[2]
                    return sent_list
                if symmetric_difference[1] in sent_list[i]:
                    sent_list[i][0] = symmetric_difference[0]
                    for element in current_sent_range:
                        if symmetric_difference[0] in element:
                            sent_list[i][2] = element[1]
                            sent_list[i][3] = element[2]
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
                                          final_list: list, similarity: list) -> list:
    """
    Generate the sentence pair (original vs finished) by index
    :return: sentence pair
    """

    if revise(df, index):

        # preindex_sent_list = get_revised_sent(df[:index - 1], original_list)
        index_sent_list = get_revised_sent(df[:index], original_list)

        for i in range(len(index_sent_list)):
            if index_sent_list[i][2] <= df['currentCursor'][index] <= index_sent_list[i][3]:
                return [original_list[i][0], final_list[i][0], original_list[i][1], similarity[i]]


def get_index(df: pd.DataFrame, index: int) -> int:
    sent_with_index = sentences_with_range(df, index)
    # print(sent_with_index)
    # print(df['currentCursor'][index])
    for i in range(len(sent_with_index)):
        # if df['textDelta'].isnull()[index]:
        if sent_with_index[i][1] <= df['currentCursor'][index] <= sent_with_index[i][2]:
            return sent_with_index[i][3]


def relocate(df: pd.DataFrame, index: int, sent_pair: list) -> bool:
    # case 1: remove a sentence
    if sent_pair[1] == 'None':
        return True


def compose(df: pd.DataFrame, index: int) -> bool:
    if revise(df, index) == False:
        if df['eventName'][index] == 'text-insert' and df['eventSource'][index] == 'user':
            return True


def modify_low(df: pd.DataFrame, index: int, sent_pair: list) -> bool:
    if revise(df, index):
        if sent_pair[2] == 'api' and sent_pair[3] >= 0.8:
            return True


def modify_high(df: pd.DataFrame, index: int, sent_pair: list) -> bool:
    if revise(df, index):
        if sent_pair[2] == 'api' and sent_pair[3] < 0.8:
            return True


def pair_similarity(original_list: list, final_list: list) -> list:
    res = []
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for i in range(len(original_list)):
        sent_pair = [original_list[i][0], final_list[i][0]]
        embeddings = model.encode(sent_pair)
        res.append(cosine_similarity(embeddings)[0][1])

    return res


def behavioural_code_identifier(df: pd.DataFrame, index: int, original_list: list, final_list: list,
                                similarity: list) -> list:
    behaviour_seq = []
    sent_pair1 = original_final_revise_sent_identifier(df, index, original_list, final_list, similarity)

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

    # 5.Reflect
    if index / len(df) > 0.9 and post_text_identifier(df, index) and (
            insert(df, index) or delete(df, index) or revise(df, index)):
        behaviour_seq.append('reflect')

    # 6.Seek Suggestion
    if df['eventName'][index] == 'suggestion-get':
        behaviour_seq.append('seekSugg')
        return behaviour_seq
    # 7.Dismiss Suggestion
    if df['eventName'][index] == 'suggestion-close' and df['eventSource'][index] == 'user':
        behaviour_seq.append('dismissSugg')

    if df['eventName'][index] == 'suggestion-get' and df['eventName'][index + 1] != 'suggestion-open':
        behaviour_seq.append('dismissSugg')
    # 8.Accept Suggestion
    if df['eventSource'][index] == 'api' and df['eventName'][index] == 'text-insert':
        behaviour_seq.append('acceptSugg')
    if sent_pair1 is not None and sent_pair1 != []:
        # print(sent_pair1)
        # 9.Relocate
        if relocate(df, index, sent_pair=sent_pair1):
            behaviour_seq.append('relocate')
        # 10.Modify Suggestion - low
        if modify_low(df, index, sent_pair=sent_pair1):
            behaviour_seq.append('lowModification')

        # 11.Modify Suggestion - high
        if modify_high(df, index, sent_pair=sent_pair1):
            behaviour_seq.append('highModification')

    # 12.Compose
    if compose(df, index):
        behaviour_seq.append('compose')

    return behaviour_seq


def execute(file_name: str, genre: str):
    data = pd.read_csv(file_name)
    original = get_init_sent(data)
    error_doc = []
    original1 = copy.deepcopy(original)
    final = get_revised_sent(data, original1)
    similarity_list = pair_similarity(original_list=original, final_list=final)
    # print('similarity: ', similarity_list)
    # print('original: ', original)
    # print('final: ', final)
    # for i in range(len(original)):
    #     if original[i][1]=='api'and similarity_list[i]>=0.8:
    #         print([original[i],final[i],similarity_list[i]])

    cols = ["eventName",
            "currentDoc",
            "eventSource",
            "sentIndex",
            "compose",
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
            "highTemp",
            "lowTemp"]
    df2 = pd.DataFrame(columns=cols)
    ops = ['text-insert', 'text-delete', 'suggestion-get']
    for i in range(len(data)):
        # print(i)
        if data['eventName'][i] == 'text-insert' and len(eval(data['textDelta'][i])['ops']) == 2 and 'insert' in \
                eval(data['textDelta'][i])['ops'][1]:
            if eval(data['textDelta'][i])['ops'][1]['insert'] == '\n':
                continue

        # print(i, in_which_gpt, data['eventName'][i])
        if data['eventName'][i] in ops or (
                data['eventName'][i] == 'suggestion-close' and data['eventSource'][i] == 'user'):
            # final = get_revised_sent(data, original)
            # print('final: ', final)
            code_list = behavioural_code_identifier(data, i, original, final, similarity=similarity_list) + [
                ('lowTemp' if data['currentTemperature'][i] < 0.5 else 'highTemp')]
            # Columns to fill with the same content from the existing dataset (data)
            given_list_2 = ['eventName', 'currentDoc', 'eventSource', 'sentIndex']
            # Accessing the values from the last row of the existing dataset (data) for the specified columns
            values_to_copy = data[given_list_2[:3]].iloc[i].to_dict()
            values_to_copy['currentDoc'] = values_to_copy['currentDoc']  # [len(data['currentDoc'][0]):]
            values_to_copy['sentIndex'] = get_index(data, i)
            # print(get_index(data, i))
            # Creating the new row with the specified values and 1 in the given columns
            new_row = {col: values_to_copy[col] if col in given_list_2 else (1 if col in code_list else 0) for col in
                       df2.columns}

            # Appending the new row to the new DataFrame (df2)
            df2 = df2.append(new_row, ignore_index=True)
            # print(i)
    df2.fillna(0, inplace=True)
    df2.to_csv('./{}/{}.csv'.format(genre, file_name[11:43]))  # [11:43] [16:48]
    print(error_doc)


def main():
    directory = './creative/'
    files = os.listdir(directory)
    files_mapping = os.listdir('./creativeMapping/')
    index=0
    for file in files:
        if file not in files_mapping:
            print('file: ', file)
            start_time = time.time()
            execute(file_name=directory + file, genre='creativeMapping')
            print("--- %s seconds ---" % (time.time() - start_time))
        index+=1
    # execute('a068c.csv', genre='creativeMapping')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
