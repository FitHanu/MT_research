# POS tagging English sentence for Machine Translation
# Command: python3 tagger.py <source_file> <lang_id> <result_file_name> <batch_size> <cpu_cores_num>
import sys
import argparse
import constants as const
from utils import get_file_name_by_path, check_extension_is_one_of
from core import process


def get_arg():
    parser = argparse.ArgumentParser(description='Process Arguments')
    parser.add_argument('--src', type=str, help='Tagged File', required=True)
    parser.add_argument('--lang_id', type=str, help='Subworded File', required=True)
    parser.add_argument('--result_name', type=str, default='', help='')
    parser.add_argument('--batch_size', type=str, default=const.DEFAULT_TAGGING_BATCH_SIZE, help='')
    parser.add_argument('--core_num', type=str, default=const.DEFAULT_TAGGING_CORE_NUM, help='')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_arg()
    source_file_path = args.src
    language_id = args.lang_id
    result_file_name = args.result_name  
    if (result_file_name == ''):
        result_file_name = get_file_name_by_path(source_file_path)
    batch_size = args.batch_size
    number_of_cores = args.core_num
    if not check_extension_is_one_of(source_file_path, const.ALLOWED_LANGUAGE_IDS) and not language_id in const.ALLOWED_LANGUAGE_IDS:
        print("Unallowed language ID: " + source_file_path + " Please give a valid language ID: " + str(const.ALLOWED_LANGUAGE_IDS))
        exit(1)
    process(source_file_path, result_file_name, language_id, int(batch_size), int(number_of_cores))