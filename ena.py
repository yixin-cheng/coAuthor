import pandas as pd
import copy
import os
from functools import reduce

# df_creative = pd.read_csv('creative.csv')
# df_argument = pd.read_csv('argumentative.csv')
# directory = './ena/'
# list_data = []
#
# directory_creative='./creative/'
# directory_argument='./argumentative/'
# creative=os.listdir(directory_creative)
# argument=os.listdir(directory_argument)

def integrate (meta, dir, to_dir, genre):
    df_genre=pd.read_csv(meta)
    files = os.listdir(dir)
    for file in files[:20]:
        index=[0]
        user = ''
        time=''
        data = pd.read_csv(dir + file)
        seq=1
        res=[]
        sent_index=data['sentIndex'][0]
        res.append(seq)
        # del data[data.columns[0]]
        for i in range(len(df_genre)):
            if df_genre['session_id'][i] == file[:-4]:
                user = df_genre['worker_id'][i]
                time=df_genre['timestamp'][i]
        for i in range(1, len(data)):
            if data['sentIndex'][i]!=sent_index:
                seq+=1
                sent_index=data['sentIndex'][i]
            res.append(seq)
            index.append(i)
        data.insert(0, "session_id", file[:-4], True)
        # data.insert(1,"timeStamp", time, True)
        data.insert(1, 'worker_id', user, True)
        data.insert(2, "genre", genre, True)
        data.insert(3, "event_id", index, True)
        data.insert(7, "sentSeq", res, True)
        data.to_csv('./{}/{}.csv'.format(to_dir,file[:5]), index=False)

integrate('creative.csv', './creativeMapping/', 'ena', 'creative')    # creative

integrate('argumentative.csv', './argumentativeMapping/', 'ena', 'argumentative')  # argument


def merge_csvs(directory_, output_file):
    file_list=os.listdir(directory_)
    # Read each CSV into a DataFrame and store all DataFrames in a list
    dfs = [pd.read_csv(directory_ + file) for file in file_list]
    # Concatenate all DataFrames into one
    merged_df = pd.concat(dfs, ignore_index=True)

    print(merged_df)
    # Save to a new CSV file
    merged_df.to_csv(output_file, index=False)


merge_csvs('./ena/', 'ena_update.csv')
#
# print(len(pd.read_csv('ena_update.csv')))

# for file in files:
#     data_file=pd.read_csv(directory+file)
#     list_data.append(data_file)
# # data1=pd.concat(data for data in list_data)
# data1=reduce(lambda x, y: pd.merge(x, y), list_data)
# print(data1)
# data1.to_csv('ena.csv', index=False)





