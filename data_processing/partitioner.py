import os
import time
import argparse

import pandas as pd
import csv
import constants as const

COL_SRC = const.SRC_COLUMN_NAME
COL_TGT = const.TGT_COLUMN_NAME

# Process Variables
DEV_ROWS = const.DEFAULT_DEV_ROW
TEST_ROWS = const.DEFAULT_TEST_ROW
RESULT_PATH = "./data"
SRC_FILE = ""
TGT_FILE = ""

PATH_SEP = os.path.sep


def handle_write_result_files(df):
    """ Write processed result files """
    global SRC_FILE, TGT_FILE, DEV_ROWS, TEST_ROWS, RESULT_PATH
    df_dev = df.sample(n = DEV_ROWS)
    df_train = df.drop(df_dev.index)

    df_test = df_train.sample(n = TEST_ROWS)
    df_train = df_train.drop(df_test.index)
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    print(f"Writing files: train: {df_train.shape[0]}, dev: {df_dev.shape[0]}, test: {df_test.shape[0]}")
    handle_write_bi_direction_files(df_train, "data.train", "")
    handle_write_bi_direction_files(df_dev, "data.dev", "")
    handle_write_bi_direction_files(df_test, "data.test", "")




def handle_write_bi_direction_files(data_frame, file_name, sub_dir):
    """ Write bi-directional files """
    global RESULT_PATH

    df_dic = data_frame.to_dict(orient='list')

    source_file = (RESULT_PATH + PATH_SEP + "source_" + file_name)
    target_file = (RESULT_PATH + PATH_SEP + "source_" + file_name)

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    with open(source_file, "w+") as file:
        file.write("\n".join(line for line in df_dic[COL_SRC]))
        file.write("\n")
    print("File Saved:", source_file)

    with open(target_file, "w+") as file:
        file.write("\n".join(line for line in df_dic[COL_TGT]))
        file.write("\n")
    print("File Saved:", target_file)


def process():
    """ MAIN PROCESS WORKFLOW """
    df_source = pd.read_csv(SRC_FILE, names=[COL_SRC], sep="\0", quoting=csv.QUOTE_NONE, skip_blank_lines=False,
                            on_bad_lines="skip")
    df_target = pd.read_csv(TGT_FILE, names=[COL_TGT], sep="\0", quoting=csv.QUOTE_NONE, skip_blank_lines=False,
                            on_bad_lines="skip")
    if len(df_source) != len(df_target):
        raise ValueError("Source and target files have different number of lines")
    # Concat two data columns

    df_corpus = pd.concat([df_source, df_target], axis=1)
    handle_write_result_files(df_corpus)

def get_arg():
    parser = argparse.ArgumentParser(description='Process Arguments')
    parser.add_argument('--src', type=str, help='Source File', required=True)
    parser.add_argument('--tgt', type=str, help='Target File', required=True)
    parser.add_argument('--dev', type=int, default=const.DEFAULT_DEV_ROW, help='Number of output .dev rows')
    parser.add_argument('--test', type=int, default=0, help='Number of output .test rows')
    parser.add_argument('--result_path', type=str, default="./data", help='Files Result Path')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arg()
    SRC_FILE = args.src
    TGT_FILE = args.tgt
    DEV_ROWS = args.dev
    TEST_ROWS = args.test
    RESULT_PATH = args.result_path
    print(f"PARTITIONING TRAIN DEV TEST...")
    print(f"params  :                             \n"
          f"--------------------------------------\n"
          f"src     : {SRC_FILE}                  \n"
          f"tgt     : {TGT_FILE}                  \n"
          f"dev     : {DEV_ROWS}                  \n"
          f"test    : {TEST_ROWS}                 \n"
          f"save_to : {RESULT_PATH}               \n"
          )
    print(f"--------------------------------------\n")
    start_time = time.time()
    process()
    end_time = time.time()
    print(f"Done data splitting in {end_time - start_time} seconds")
