import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)
