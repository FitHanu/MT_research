# Sentences & Feature Extract from .zip corpus
### python version 3.12
### .venv run recommend

PROCESS DATA:
** STEPS **
1. unzip files
2. check src_lang_id & tgt_lang_id in extracted files
3. collect & frame data with src_lang_id & tgt_lang_id extension in to a dataframe with 2 columns (src, tgt)
4. filter data
5. output filtered data into n*2 files (n is the number of zipped corpus| n/2 is the number of src files, same with tgt)
6. partition 10% of all merged data for development dataset (rule: development dataset should be 5 -> 10% of the total corpus size)
7. output development dataset into 2 files (src, tgt)

** RUN **
python3 ./data_partitioning/partitioner.py vi en 0.1 ./data/corpus/OpenSubtitles/en-vi.txt ./data/corpus/QED/en-vi.txt ./data/corpus/Wikimedia/en-vi.txt


GET TEST DATA FILES:

wget -P ./data/corpus/ https://object.pouta.csc.fi/OPUS-NeuLab-TedTalks/v1/moses/en-vi.txt.zip
wget -P ./data/corpus/ https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-vi.txt.zip

TEST DATA FILES:

+ 2000 lines of filtered wikimedia
+ 2000 lines of filtered QED
+ 2000 lines of filtered NeuLab-TedTalks
+ 2000 lines of filtered TED2020

WORKFLOWS:
+ process data for training
+ create a configuration file for training
+ train the model
+ customize training to accept another input embedding for POS tag
+ train the model with POS tag

python3 tagger.py <source_file> <lang_id> <result_file_name> <batch_size> <cpu_cores_num>

command:
python3 ./pos_tagging/tagger.py ./data/test_data.zh zh ./test_data/test_result/zh_result_tagged.zh 5
python3 ./pos_tagging/tagger.py ./data/test_data.en en ./test_data/test_result/en_result_tagged.en 5

python3 data_processing/process_data.py --src vi --tgt en ./data/OpenSubtitiles/en-vi.txt.zip ./data/QED/en-vi.txt.zip ./data/Wikimedia/en-vi.txt.zip
python3 train_vocab/bpe.py --src ./data/corpus_filtered.vi --tgt ./data/corpus_filtered.en --vocab_size 60000
python3 train_vocab/subword.py --src source.model --tgt target.model --src_data ./data/corpus_filtered.vi --tgt_data ./data/corpus_filtered.en
wc -l ./data/corpus_filtered.vi.subw
wc -l ./data/corpus_filtered.en.subw
head -n 5 ./data/corpus_filtered.vi.subw
head -n 5 ./data/corpus_filtered.en.subw
tail -n 5 ./data/corpus_filtered.vi.subw
tail -n 5 ./data/corpus_filtered.en.subw

python3 ./data_processing/partitioner.py --src ./data/corpus_filtered.vi.subw --tgt ./data/corpus_filtered.en.subw --dev 30000 --test 2500 --result_path ./data/partitioned
