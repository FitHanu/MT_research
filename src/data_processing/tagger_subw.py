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