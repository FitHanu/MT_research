import spacy
import sentencepiece as spm

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load SentencePiece model (you need to train or download one)
sp = spm.SentencePieceProcessor()
sp.load("your_model.model")  # Replace with your trained SentencePiece model

# Sample text
text = "I am learning subword tokenization with spaCy."

# Apply SentencePiece subword tokenization
subwords = sp.encode(text, out_type=str)

# Process with spaCy
doc = nlp(text)

# Print results
print("Subword Tokens:", subwords)
print("spaCy Tokens:", [token.text for token in doc])

