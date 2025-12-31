import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')

english_stop = set(stopwords.words('english'))
hinglish_stop = {
    "hai","ho","tha","thi","hain","ka","ki","ke",
    "aur","par","se","yeh","woh","kar","karo",
    "nahi","bahut","bhi","liye","gaya","raha"
}
all_stopwords = english_stop.union(hinglish_stop)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\u0900-\u097F0-9\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in all_stopwords]
    return " ".join(words)
