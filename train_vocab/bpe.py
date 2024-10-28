import sentencepiece as spm
import argparse

SOURCE_PATH = ""
TARGET_PATH = ""
VOCAB_SIZE = 50000
SRC_ARGS_STRING = ""
TGT_ARGS_STRING = ""

def process():
    source_train_value = SRC_ARGS_STRING
    spm.SentencePieceTrainer.Train(source_train_value)
    print("Done training SentencePiece model for the Source data!")

    target_train_value = TGT_ARGS_STRING
    spm.SentencePieceTrainer.Train(target_train_value)
    print("Done training SentencePiece model for the Target data!")

def join_params():
    global SRC_ARGS_STRING, TGT_ARGS_STRING
    params = {
        '--input': SOURCE_PATH,
        '--model_prefix': 'source',
        '--vocab_size': VOCAB_SIZE,
        '--hard_vocab_limit': 'false',
        '--split_digits': 'true'
    }
    SRC_ARGS_STRING = ' '.join(f"{key}={value}" for key, value in params.items())
    print(f"SRC_ARGS_STRING: {SRC_ARGS_STRING}")
    params_2 = {
        '--input': TARGET_PATH,
        '--model_prefix': 'target',
        '--vocab_size': VOCAB_SIZE,
        '--hard_vocab_limit': 'false',
        '--split_digits': 'true'
    }
    TGT_ARGS_STRING = ' '.join(f"{key}={value}" for key, value in params_2.items())
    print(f"TGT_ARGS_STRING: {TGT_ARGS_STRING}")


def get_arg():
    parser = argparse.ArgumentParser(description='Process Arguments')
    parser.add_argument('--src', type=str, help='Source Path', required=True)
    parser.add_argument('--tgt', type=str, help='Target Path', required=True)
    parser.add_argument("--vocab_size", type=int, help="Vocabulary Size", default=50000)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arg()
    SOURCE_PATH = args.src
    TARGET_PATH = args.tgt
    VOCAB_SIZE = args.vocab_size
    print(f"TRAIN VOCAB MODEL")
    print(f"params  :                             \n"
          f"--------------------------------------\n"
          f"src     : {SOURCE_PATH}               \n"
          f"tgt     : {TARGET_PATH}               \n"
          f"size    : {VOCAB_SIZE}                \n"
          f"--------------------------------------\n"
          )
    join_params()
    process()