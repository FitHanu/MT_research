# Description: script will merge data from multiple zip files into one and reduce it by percentage
#Read & Write files, get device compatible batch size
import os
import time
import argparse

import pandas as pd
import numpy as np
import csv
import constants as const

from utils import (
    get_dirname_by_path,
    unzip_corpus, get_data_filepath_by_language_id,
    get_corpus_name_by_path
)

COL_SRC = const.SRC_COLUMN_NAME
COL_TGT = const.TGT_COLUMN_NAME

# Process Variables
RESULT_PATH = "./data"
SRC_LANG_ID = ""
TGT_LANG_ID = ""
ZIP_FILE_PATHS = []

PATH_SEP = os.path.sep

# init dataframes
main_df = pd.DataFrame(columns=[COL_SRC, COL_TGT])
corpus_dict = {}



def filter_data(df, corpus_no):
    """ Filter training data in data frame """
    # Delete nan
    print(f"Filtering data for corpus no: {corpus_no}")
    df = df.dropna()
    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])
    # Drop duplicates
    df = df.drop_duplicates()
    print("--- Duplicates Deleted\t\t\t--> Rows:", df.shape[0])
    # Drop copy-source rows
    df["Source-Copied"] = df[COL_SRC] == df[COL_TGT]
    df = df.set_index(['Source-Copied'])
    try:  # To avoid (KeyError: '[True] not found in axis') if there are no source-copied cells
        df = df.drop([True])  # Boolean, not string, do not add quotes
    except:
        pass
    df = df.reset_index()
    df = df.drop(['Source-Copied'], axis=1)
    print("--- Source-Copied Rows Deleted\t\t--> Rows:", df.shape[0])
    # Drop too-long rows (source or target)
    # Based on your language, change the values "2" and "200"
    df["Too-Long"] = ((df[COL_SRC].str.count(' ') + 1) > (
            df[COL_TGT].str.count(' ') + 1) * 2) | \
                     ((df[COL_TGT].str.count(' ') + 1) > (
                             df[COL_SRC].str.count(' ') + 1) * 2) | \
                     ((df[COL_SRC].str.count(' ') + 1) > 200) | \
                     ((df[COL_TGT].str.count(' ') + 1) > 200)
    df = df.set_index(['Too-Long'])
    try:  # To avoid (KeyError: '[True] not found in axis') if there are no too-long cells
        df = df.drop([True])  # Boolean, not string, do not add quotes
    except:
        pass
    df = df.reset_index()
    df = df.drop(['Too-Long'], axis=1)
    print("--- Too Long Source/Target Deleted\t--> Rows:", df.shape[0])
    # Remove HTML and normalize
    # Use str() to avoid (TypeError: expected string or bytes-like object)
    # Note: removing tags should be before removing empty cells because some cells might have only tags and become empty.
    df = df.replace(r'<.*?>|&lt;.*?&gt;|&?(amp|nbsp|quot);|{}', ' ', regex=True)
    df = df.replace(r'  ', ' ', regex=True)  # replace double-spaces with one space
    print("--- HTML Removed\t\t\t--> Rows:", df.shape[0])
    # Replace empty cells with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # Delete nan (already there, or generated from the previous steps)
    df = df.dropna()
    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])
    df = df.reset_index(drop=True)
    #
    # # Shuffle the data
    # df = df.sample(frac=1).reset_index(drop=True)
    # print("--- Rows Shuffled\t\t\t--> Rows:", df.shape[0])
    return df


def handle_write_result_files():
    """ Write processed result files """
    global main_df, corpus_dict, RESULT_PATH
    handle_write_bi_direction_files(main_df, const.FULL_DATA_FILE_NAME, "")



def handle_write_bi_direction_files(data_frame, file_name, sub_dir):
    """ Write bi-directional files """
    global SRC_LANG_ID, TGT_LANG_ID, RESULT_PATH

    df_dic = data_frame.to_dict(orient='list')

    source_file = (RESULT_PATH + PATH_SEP + sub_dir + PATH_SEP + file_name + "." + SRC_LANG_ID)
    target_file = (RESULT_PATH + PATH_SEP + sub_dir + PATH_SEP + file_name + "." + TGT_LANG_ID)

    with open(source_file, "w") as file:
        file.write("\n".join(line for line in df_dic[COL_SRC]))
        file.write("\n")
    print("File Saved:", source_file)

    with open(target_file, "w") as file:
        file.write("\n".join(line for line in df_dic[COL_TGT]))
        file.write("\n")
    print("File Saved:", target_file)


def process():
    """ MAIN PROCESS WORKFLOW """
    global main_df, corpus_dict, SRC_LANG_ID, TGT_LANG_ID, RESULT_PATH, ZIP_FILE_PATHS
    corpus_no = 0

    for zip_path in ZIP_FILE_PATHS:
        corpus_no += 1
        print(f"Processing corpus no {corpus_no} : {os.path.basename(zip_path)}")
        directory_path = get_dirname_by_path(zip_path)
        unzip_corpus(zip_path, directory_path)
        src_file = get_data_filepath_by_language_id(directory_path, SRC_LANG_ID)
        tgt_file = get_data_filepath_by_language_id(directory_path, TGT_LANG_ID)
        corpus_dict[corpus_no] = get_corpus_name_by_path(src_file)
        print(f"Source file: {src_file}")
        print(f"Target file: {tgt_file}")
        filter_merge_data(src_file, tgt_file, corpus_no)

    print(f"Total corpus: {corpus_no}")
    print(corpus_dict)
    print(f"Rows after processed {corpus_no} corpus: {main_df.shape[0]}")
    handle_write_result_files()


def filter_merge_data(source_file, target_file, corpus_no):
    """ Process & Split & Merge data to main dataset """
    global main_df
    df_source = pd.read_csv(source_file, names=[COL_SRC], sep="\0", quoting=csv.QUOTE_NONE, skip_blank_lines=False,
                            on_bad_lines="skip")
    df_target = pd.read_csv(target_file, names=[COL_TGT], sep="\0", quoting=csv.QUOTE_NONE, skip_blank_lines=False,
                            on_bad_lines="skip")
    if len(df_source) != len(df_target):
        raise ValueError("Source and target files have different number of lines")
    #Concat two data columns
    df_corpus = pd.concat([df_source, df_target], axis=1)
    #Filter data
    df_corpus = filter_data(df_corpus, corpus_no)

    #Concat corpus label for file output handling
    print(f"Processed corpus dataframe shape (rows, columns): {df_corpus.shape}")
    print(f"5 rows from head of corpus {corpus_no}: ")
    print(df_corpus.head(5))
    print(f"-------------------------------------------------------------------")
    print(f"5 rows from tail of corpus {corpus_no}: ")
    print(df_corpus.tail(5))
    main_df = pd.concat([main_df, df_corpus], axis=0)
    print(f"Total rows after merging corpus {corpus_no}: {main_df.shape[0]}")



def get_arg():
    parser = argparse.ArgumentParser(description='Process Arguments')
    parser.add_argument('--src', type=str, help='Source lang ID', required=True)
    parser.add_argument('--tgt', type=str, help='Target lang ID', required=True)
    parser.add_argument('--result_path', type=str, default="./data", help='Files Result Path')
    parser.add_argument('zip_file_paths', nargs='*', help='Additional zip file paths (array input)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arg()
    SRC_LANG_ID = args.src
    TGT_LANG_ID = args.tgt
    RESULT_PATH = args.result_path
    ZIP_FILE_PATHS = args.zip_file_paths
    print(f"params  :                             \n"
          f"--------------------------------------\n"
          f"src     : {SRC_LANG_ID}               \n"
          f"tgt     : {TGT_LANG_ID}               \n"
          f"save_to : {RESULT_PATH}               \n"
          f"files   :                              ")
    i = 0
    for path in ZIP_FILE_PATHS:
        i += 1
        print(f" {i}      : {path}")
    print(f"--------------------------------------\n")
    start_time = time.time()
    process()
    end_time = time.time()
    print(f"Done data processing in {end_time - start_time} seconds")
