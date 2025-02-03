import argparse
import sentencepiece as spm


MODEL = ""
SUBWORDED_DATA = ""
RS_PATH = SUBWORDED_DATA + ".desub"





def process():
    sp = spm.SentencePieceProcessor()
    sp.Load(MODEL)
    with open(SUBWORDED_DATA) as pred, open(RS_PATH, "w+") as pred_decoded:
        for line in pred:
            line = line.strip().split(" ")
            line = sp.DecodePieces(line)
            pred_decoded.write(line + "\n")
    print(f"Desubworded file: {SUBWORDED_DATA} Output: {RS_PATH}")

def get_arg():
    parser = argparse.ArgumentParser(description='Process Arguments')
    parser.add_argument('--model', type=str, help='Source Model Path', required=True)
    parser.add_argument('--data', type=str, help='Target Model Path', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arg()
    MODEL = args.model
    SUBWORDED_DATA = args.data
    print(f"DESUBWORDING DATA...")
    print(f"params   :                             \n"
          f"---------------------------------------\n"
          f"model    : {MODEL}                     \n"
          f"data     : {SUBWORDED_DATA}            \n"
          f"---------------------------------------\n"
          )
    process()