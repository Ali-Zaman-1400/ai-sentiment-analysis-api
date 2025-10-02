import re
from typing import Iterable, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Lazy download (safe if already present)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_stopwords = set(stopwords.words("english"))
_lem = WordNetLemmatizer()

def clean_text(text: str,
               lowercase: bool = True,
               remove_punct: bool = True,
               remove_stopwords: bool = True,
               lemmatize: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    if lowercase:
        t = t.lower()
    if remove_punct:
        t = re.sub(r"[^a-z0-9\s]", " ", t)
    tokens = t.split()
    if remove_stopwords:
        tokens = [w for w in tokens if w not in _stopwords]
    if lemmatize:
        tokens = [_lem.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def batch_clean_text(texts: Iterable[str], **kwargs) -> List[str]:
    return [clean_text(t, **kwargs) for t in texts]
