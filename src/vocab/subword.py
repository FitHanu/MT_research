import argparse
import sentencepiece as spm


sp = spm.SentencePieceProcessor()

def subword(model, data, result_path):
    sp.Load(model)
    with open(data, encoding='utf-8') as source, open(result_path, "w+", encoding='utf-8') as sub:
        for line in source:
            line = line.strip()
            line = sp.EncodeAsPieces(line)
            line = " ".join([token for token in line])
            sub.write(line + "\n")
    print(f"Subworded file: {data} Output: {result_path}")

def process(src, tgt, src_data, tgt_data):
    src_rs_path = src_data + ".subw"
    tgt_rs_path = tgt_data + ".subw"
    subword(src, src_data, src_rs_path)
    subword(tgt, tgt_data, tgt_rs_path)

def get_arg():
    parser = argparse.ArgumentParser(description='Process Arguments')
    parser.add_argument('--src', type=str, help='Source Model Path', required=True)
    parser.add_argument('--tgt', type=str, help='Target Model Path', required=True)
    parser.add_argument('--src_data', type=str, help='Source Data Path', required=True)
    parser.add_argument('--tgt_data', type=str, help='Target Data Path', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arg()
    SRC_MODEL = args.src
    TGT_MODEL = args.tgt
    SRC_DATA = args.src_data
    TGT_DATA = args.tgt_data
    print(f"SUBWORDING DATA...")

    print(f"params   :                             \n"
          f"---------------------------------------\n"
          f"src model: {SRC_MODEL}                 \n"
          f"tgt model: {TGT_MODEL}                 \n"
          f"src data : {SRC_DATA}                  \n"
          f"tgt data : {TGT_DATA}                  \n"
          f"---------------------------------------\n"
          )
    process(SRC_MODEL, TGT_MODEL, SRC_DATA, TGT_DATA)