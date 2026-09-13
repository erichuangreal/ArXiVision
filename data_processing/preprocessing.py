from datasketch import MinHash, MinHashLSH
# from langdetect import detect
# from bs4 import BeautifulSoup
import re
import shutil
from collections import Counter
from pathlib import Path

def copy_metadata_files(input_directory: Path, output_directory: Path) -> None:
    for metadata_path in input_directory.rglob("metadata.json"):
        relative_path = metadata_path.relative_to(input_directory)
        output_path = output_directory / relative_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_path, output_path)
    print(f"Copied metadata.json files from {input_directory} to {output_directory}.")

# Subset-embedded fonts ship a broken ToUnicode map, so ligature glyphs extract as
# the wrong character. Every extractor reads the same map, so this is repaired here.
LIGATURE_REPAIRS = {"H": "ti", "Y": "tt", "A": "ti", "O": "tti", "M": "tt", "p": "tt"}

# Share of words that must look mis-encoded before a document is repaired.
CORRUPTION_RATE = 0.01

WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z'-]{2,}\b")
SYSTEM_WORDS = Path("/usr/share/dict/words")


def build_vocabulary(texts):
    # A word must appear in two documents to count. Mis-encoding is per-PDF, so a
    # corrupted token never corroborates itself, while real words recur.
    texts = list(texts)
    document_frequency = Counter()
    for text in texts:
        document_frequency.update({w.lower() for w in WORD_PATTERN.findall(text)})

    minimum = 2 if len(texts) > 1 else 1
    vocabulary = {w for w, n in document_frequency.items() if n >= minimum}

    if SYSTEM_WORDS.exists():
        vocabulary |= set(SYSTEM_WORDS.read_text(errors="ignore").lower().split())
    return vocabulary


def mis_encoded_rate(text, vocabulary):
    # An uppercase letter between two lowercase ones, in a word nothing else uses.
    tokens = WORD_PATTERN.findall(text)
    if not tokens:
        return 0.0
    suspicious = sum(
        1 for token in tokens
        if re.search(r"[a-z][A-Z][a-z]", token) and token.lower() not in vocabulary
    )
    return suspicious / len(tokens)


def repair_ligatures(text, vocabulary):
    # Clean documents are skipped entirely, so the ambiguous p->tt rule can never
    # rewrite an abbreviation like "sep." that happens to become a real word.
    if mis_encoded_rate(text, vocabulary) < CORRUPTION_RATE:
        return text, 0

    # Only rewrites a token that is unknown and becomes known, so correct
    # camelCase such as arXiv or ImageNet is never touched.
    repairs = 0

    def repair(match):
        nonlocal repairs
        token = match.group()
        if token.lower() in vocabulary:
            return token
        for bad, good in LIGATURE_REPAIRS.items():
            if bad in token:
                candidate = token.replace(bad, good)
                if candidate.lower() in vocabulary:
                    repairs += 1
                    return candidate
        return token

    return WORD_PATTERN.sub(repair, text), repairs


def minhash_deduplication(texts, threshold=0.7):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_texts = []
    for i, doc in enumerate(texts):
        m = MinHash(num_perm=128)
        for word in set(doc.split()):
            m.update(word.encode('utf8'))
        if not lsh.query(m):
            lsh.insert(f"doc{i}", m)
            unique_texts.append(doc)
    return unique_texts

def clean_html_and_filter_lang(texts, lang='en'):
    filtered = []
    for txt in texts:
        txt = BeautifulSoup(txt, 'html.parser').get_text()
        try:
            if detect(txt.strip()) == lang:
                filtered.append(txt.strip())
        except:
            continue
    return filtered

def strip_pii(text):
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
    text = re.sub(r'\b\d{12,19}\b', '[CREDIT_CARD]', text)
    text = re.sub(r'\b(?:\d{3}-){2}\d{4}\b', '[PHONE]', text)
    return text

def remove_repetitive_ngrams(text, n=3, threshold=3):
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    counts = Counter(ngrams)
    repetitive = [ngram for ngram, count in counts.items() if count >= threshold]

    for phrase in repetitive:
        escaped_phrase = re.escape(phrase)
        text = re.sub(rf'(?:{escaped_phrase}\s*){{{threshold},}}', phrase + ' ', text)

    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def preprocess(text) :
    # chunking text
    chunks = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    original_chunks = len(chunks)
    # minhash deduplication
    step2 = minhash_deduplication(chunks)
    dedup_removed = original_chunks - len(step2)
    
    # PII stripping and repetitive n-gram removal
    pii_replacements = 0
    cleaned_data = []

    for chunk in step2:
        before = chunk
        step3 = strip_pii(chunk)

        if step3 != before:
            pii_replacements += 1

        step4 = remove_repetitive_ngrams(step3)
        cleaned_data.append(step4)
    
    cleaned_text = "\n\n".join(cleaned_data)
    
    print(f"Paragraphs: {original_chunks}")
    print(f"Duplicates removed: {dedup_removed}")
    print(f"PII replacements: {pii_replacements}")
    
    print("Original length:", len(text))
    print("Processed length:", len(cleaned_text))
    
    return cleaned_text