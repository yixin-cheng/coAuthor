import pandas as pd
import copy
import os
from functools import reduce


# df_creative = pd.read_csv('creative_metadata.csv')
# df_argument = pd.read_csv('argumentative_metadata.csv')
# directory = './ena/'
# list_data = []
#
# directory_creative='./creative/'
# directory_argument='./argumentative/'
# creative=os.listdir(directory_creative)
# argument=os.listdir(directory_argument)

def integrate(meta, dir, to_dir, genre):
    df_meta = pd.read_csv(meta)
    files = os.listdir(dir)
    for file in files:
        index = [0]
        user = ''
        satisfaction = ''
        confidence = ''
        ownershipSurvey = ''
        ownershipMetadata = ''
        data = pd.read_csv(dir + file)
        # data=data.drop('sentSeq', axis=0)
        seq = 1
        res = []
        sent_index = data['sentIndex'][0]
        res.append(seq)
        for i in range(len(df_meta)):
            if df_meta['session_id'][i].strip() == file[:-4]:
                user = df_meta['worker_id'][i]
                ownershipMetadata = df_meta['ownership_metadata'][i]

        for i in range(1, len(data)):
            if data['sentIndex'][i] != sent_index:
                seq += 1
                sent_index = data['sentIndex'][i]
            res.append(seq)
            index.append(i)
        data.insert(0, "session_id", file[:-4], True)
        # data.insert(1,"timeStamp", time, True)
        data.insert(1, 'worker_id', user, True)
        data.insert(2, "genre", genre, True)
        # data.insert(3, "satisfaction", satisfaction, True)
        # data.insert(4, "confidence", confidence, True)
        # data.insert(5, "ownershipSurvey", ownershipSurvey, True)
        data.insert(6, "ownershipMetadata", ownershipMetadata, True)
        data.insert(8, "event_id", index, True)
        # data.insert(12, "sentSeq", res, True)
        data['sentSeq']=res
        data.to_csv('./{}/{}.csv'.format(to_dir, file[:5]), index=False)

#
# integrate('./results/creative_metadata_results.csv', './creativeMappingNew/', 'enaNew',
#           'creative')  # creative
#
# integrate('./results/argu_metadata_results.csv', './argumentativeMappingNew/', 'enaNew',
#           'argumentative')  # argument


def merge_csvs(directory_, output_file):
    file_list = os.listdir(directory_)
    # Read each CSV into a DataFrame and store all DataFrames in a list
    dfs = [pd.read_csv(directory_ + file) for file in file_list]
    # Concatenate all DataFrames into one
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop('currentDoc', axis=1)
    print(merged_df)
    # Save to a new CSV file
    merged_df.to_csv(output_file, index=False)


# merge_csvs('./enaNew/', 'ena_all_new_v2.csv')
#

# files1 =os.listdir('creative')
#
# file4= os.listdir('creativeMappingNew')
#
#
#
# print(files1 ==file4)
#
#
# file2 =os.listdir('argumentative')
#
# file3 = os.listdir('argumentativeMappingNew')
# print(len(file2))
# print(len(file3))
# print(file2==file3)
# for file in file4:
#     for file_sub in files1:
#         if file in file_sub:
#             os.rename('./creativeMappingNew/'+file, './creativeMappingNew/'+file_sub)

# print(len(pd.read_csv('ena_update.csv')))

# for file in files:
#     data_file=pd.read_csv(directory+file)
#     list_data.append(data_file)
# # data1=pd.concat(data for data in list_data)
# data1=reduce(lambda x, y: pd.merge(x, y), list_data)
# print(data1)
# data1.to_csv('ena.csv', index=False)
author=set()
data1=pd.read_csv('./results/argu_metadata_results.csv')

data2=pd.read_csv('./results/creative_metadata_results.csv')

print(len(set(data1['worker_id'].unique()).union(set(data2['worker_id'].unique()))))
