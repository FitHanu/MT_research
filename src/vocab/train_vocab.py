import sentencepiece as spm
import argparse


ALLOWED_VOCAB_TYPES = ['bpe', 'unigram']

def process(source_path, target_path, vocab_size, model_type):
    SRC_ARGS_STRING, TGT_ARGS_STRING = join_params(source_path, target_path, vocab_size, model_type)

    spm.SentencePieceTrainer.Train(SRC_ARGS_STRING)
    print("Done training SentencePiece model for the Source data!")

    spm.SentencePieceTrainer.Train(TGT_ARGS_STRING)
    print("Done training SentencePiece model for the Target data!")


def join_params(src_path, tgt_path, vocal_size, model_type):
    params = {
        '--input': src_path,
        '--model_prefix': 'src_' + model_type,
        '--split_digits': 'true'
    }
    SRC_ARGS_STRING = ' '.join(f"{key}={value}" for key, value in params.items())
    params_2 = {
        '--input': tgt_path,
        '--model_prefix': 'tgt_' + model_type,
        '--split_digits': 'true'
    }
    TGT_ARGS_STRING = ' '.join(f"{key}={value}" for key, value in params_2.items())
    if (vocal_size > 0):
        SRC_ARGS_STRING += f' --vocab_size={vocal_size}'
        TGT_ARGS_STRING += f' --vocab_size={vocal_size}'
    else:
        SRC_ARGS_STRING += ' --hard_vocab_limit=false'
        TGT_ARGS_STRING += ' --hard_vocab_limit=false'

    print(f"source train arg string: {SRC_ARGS_STRING}")    
    print(f"target train arg string: {TGT_ARGS_STRING}")
    return SRC_ARGS_STRING, TGT_ARGS_STRING


def get_arg():
    parser = argparse.ArgumentParser(description='Process Arguments')
    parser.add_argument('--src', type=str, help='Source Path', required=True)
    parser.add_argument('--tgt', type=str, help='Target Path', required=True)
    parser.add_argument('--type', type=str, choices=ALLOWED_VOCAB_TYPES, help='Vocabulary Type', default='unigram')
    parser.add_argument('--vocab_size', type=int, help="Vocabulary Size", default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arg()
    SOURCE_PATH = args.src
    TARGET_PATH = args.tgt
    VOCAB_SIZE = args.vocab_size
    MODEL_TYPE = args.type
    print(f"TRAIN VOCAB MODEL")
    print(f"params  :                             \n"
          f"--------------------------------------\n"
          f"src     : {SOURCE_PATH}               \n"
          f"tgt     : {TARGET_PATH}               \n"
          f"size    : {VOCAB_SIZE}                \n"
          f"type    : {MODEL_TYPE}                \n"
          f"--------------------------------------\n"
          )
    process(source_path=SOURCE_PATH, target_path=TARGET_PATH, vocab_size=VOCAB_SIZE, model_type=MODEL_TYPE)