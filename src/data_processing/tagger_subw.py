import argparse
import constants as const
from utils import get_file_name_by_path, read_file, write_pos_tagged_file
import spacy
import os
import sys
from core import map_tag_to_new_format
import pandas as pd
import csv
import sentencepiece as spm
"""
I Wrote this script as separate module along with tagger.py for quick and maintainability

Params:
+ src_lang_id: str
+ tgt_lang_id: str
+ result_path: str

Step 1. Load initial data as arrays
Step 2. For each sentence from initial data
    Step 2.1. Tokenize sentence
    Step 2.2. BPE sentence as temp
    Step 2.3. Tag sentence as temp
    Step 2.4. Initialize empty sentence
        Step 2.4.1. For each token in BPE'd sentence
        Step 2.4.2.         
"""
class TaggerSubword:

    def __init__(self, src, language_id, subw_file, batch_size, core_num):
        self.nlp = spacy.blank(language_id)
        self.vi_flag = False
        self.batch_size = batch_size
        self.core_num = core_num
        self.pd = self.load_data(src, subw_file)
        
        print(self.pd.shape())
        print(self.pd.head(10))
    
    def load_data(self, src, subw_file):
        # Decode src file to arrays of sentences
        with open(src, 'r', encoding='utf-8') as f:
            src_sentences = [line.strip().split() for line in f]

        # Load SentencePiece model and decode subworded file
        sp = spm.SentencePieceProcessor()
        sp.Load(subw_file)
        subword_sentences = [sp.DecodePieces(line.strip().split()) for line in open(subw_file, 'r', encoding='utf-8')]

        # Concat source and subword sentences
        if len(src_sentences) != len(subword_sentences):
            raise ValueError("Source and subword files have different number of lines")
        return pd.DataFrame({'source': src_sentences, 'subword': subword_sentences})



    def load_model(self, language_id):
        spacy.blank(language_id)
        cache_dir = const.MODEL_CACHE_DIR
        model_name = str(const.SPACY_TAGGING_MODELS[language_id])
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        print("Tagging model: " + model_name)
        print("Loading model ...")
        try:
            self.nlp = spacy.load(os.path.join(cache_dir, model_name))
        except OSError:
            print("Model not initialized. Downloading model: " + model_name + "...")

            # Handle vi model exclusively
            if language_id == "vi":
                # Model downloaded from installation step
                # spacy.cli.download(model_path)
                nlp = spacy.load(model_name)
                nlp.to_disk(os.path.join(cache_dir, model_name))
            else:
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name)
                nlp.to_disk(os.path.join(cache_dir, model_name))

    def pos_tag_sentences(self, sentences, _batch_size):
        tag_sentences = []
        sentences_len = len(sentences)
        log_num = const.LOG_PER_NUMBER_OF_LINE_TAGGING

        for i, doc in enumerate(self.nlp.pipe(sentences)):
            if  log_num >= sentences_len - i:
                sys.stdout.write(f"\rNumber Of Sentences Processed: {i + 1}/{sentences_len}")
                sys.stdout.flush()
            if i % log_num == 0:
                sys.stdout.write(f"\rNumber Of Sentences Processed: {i}/{sentences_len}")
                sys.stdout.flush()
            if self.vi_flag:
                for token in doc:
                    pos_tag = const.VI_TAG_MAPPING[token.tag_]
                tag_processed_sentence = " ".join([map_tag_to_new_format(token.tag_) for token in doc])
            else:
                tag_processed_sentence = " ".join([token.pos_ for token in doc])
            tag_sentences.append(tag_processed_sentence)
        return tag_sentences

    def process(self, source_file_path, language_id, subword_file_path, result_file_name, batch_size, number_of_cores):
        self.load_model(language_id)
        sentences = read_file(source_file_path)
        tagged_sentences = self.pos_tag_sentences(sentences, batch_size)
        write_pos_tagged_file(tagged_sentences, result_file_name)


def process(source_file_path, language_id, subword_file_path, result_file_name, batch_size, number_of_cores):
    print("Processing ...")


def get_arg():
    parser = argparse.ArgumentParser(description='Process Arguments, please provide matching tag & subw files for accurate result')
    parser.add_argument('--src', type=str, help='Original File', required=True)
    parser.add_argument('--lang_id', type=str, help='Language ID', required=True)
    parser.add_argument('--subw_file', type=str, help='Subworded File', required=True)
    parser.add_argument('--result', type=str, default='', help='')
    parser.add_argument('--batch_size', type=str, default=const.DEFAULT_TAGGING_BATCH_SIZE, help='')
    parser.add_argument('--core_num', type=str, default=const.DEFAULT_TAGGING_CORE_NUM, help='')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_arg()
    source_file_path = args.src
    subword_file_path = args.subw_file
    result_file_name = args.result
    language_id = args.lang_id
    if (result_file_name == ''):
        result_file_name = get_file_name_by_path(source_file_path)
    batch_size = args.batch_size
    number_of_cores = args.core_num
    tagger = TaggerSubword(source_file_path, language_id, subword_file_path, batch_size, number_of_cores)