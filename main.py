import re
import requests
import random
import stanza
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import os
from urllib.parse import quote

# -------------------------------
# НАЛАШТУВАННЯ
# -------------------------------
TARGET_TOKENS = 200000
TOP_N = 20
LANGUAGES = ["en","be","uk","bg","fi","nl","cs","fr","pl","da","hr","pt","de","hu","ro","el","it","sk","es","lt","sl","et","lv","sv"]
Q_SEARCH_MAX = 50.0
Q_SEARCH_STEPS = 1000
REQUEST_TIMEOUT = 20
MAX_BOOK_ATTEMPTS = 500
TEXT_SOURCES = ["gutenberg", "wikisource", "internet_archive"]
INTERNET_ARCHIVE_ROWS = 50
INTERNET_ARCHIVE_LANGUAGE_NAMES = {
    "en": "English",
    "be": "Belarusian",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "fi": "Finnish",
    "nl": "Dutch",
    "cs": "Czech",
    "fr": "French",
    "pl": "Polish",
    "da": "Danish",
    "hr": "Croatian",
    "pt": "Portuguese",
    "de": "German",
    "hu": "Hungarian",
    "ro": "Romanian",
    "el": "Greek",
    "it": "Italian",
    "sk": "Slovak",
    "es": "Spanish",
    "lt": "Lithuanian",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "sv": "Swedish",
}
SOURCE_NAMES = {
    "gutenberg": "Project Gutenberg",
    "wikisource": "Wikisource",
    "internet_archive": "Internet Archive",
}
_SOURCE_CACHE = {}

# -------------------------------
# 0. Папка результатів
# -------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = f"results_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(OUTPUT_DIR, "report.txt")

def write_report(text):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# -------------------------------
# 1. Джерела текстів
# -------------------------------
def get_books(lang):
    url = f"https://gutendex.com/books?languages={lang}"
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["results"]

def get_random_book_url(lang):
    cache_key = ("gutenberg_urls", lang)
    if cache_key in _SOURCE_CACHE:
        plain_text_urls = _SOURCE_CACHE[cache_key]
    else:
        books = get_books(lang)
        plain_text_urls = []
        for book in books:
            for k, v in book["formats"].items():
                if "text/plain" in k:
                    plain_text_urls.append(v)
        _SOURCE_CACHE[cache_key] = plain_text_urls

    if not plain_text_urls:
        raise ValueError(f"No text/plain books found for language '{lang}'")

    return random.choice(plain_text_urls)

def download_gutenberg_text(lang):
    url = get_random_book_url(lang)
    return strip_gutenberg(download_text(url)), url

def download_wikisource_text(lang):
    api_url = f"https://{lang}.wikisource.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": 1,
        "prop": "extracts",
        "explaintext": 1,
        "exlimit": 1,
        "redirects": 1,
    }
    r = requests.get(api_url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"No Wikisource pages found for language '{lang}'")

    page = next(iter(pages.values()))
    text = page.get("extract", "")
    if not text.strip():
        raise ValueError(f"Empty Wikisource extract for language '{lang}'")

    title = page.get("title", "")
    page_url = f"https://{lang}.wikisource.org/wiki/{quote(title.replace(' ', '_'))}"
    return text, page_url

def get_internet_archive_identifiers(lang):
    cache_key = ("internet_archive_ids", lang)
    if cache_key in _SOURCE_CACHE:
        return _SOURCE_CACHE[cache_key]

    language_name = INTERNET_ARCHIVE_LANGUAGE_NAMES.get(lang)
    if not language_name:
        raise ValueError(f"No Internet Archive language mapping for '{lang}'")

    params = {
        "q": f'mediatype:texts AND languageSorter:"{language_name}"',
        "fl[]": "identifier",
        "rows": INTERNET_ARCHIVE_ROWS,
        "page": 1,
        "output": "json",
    }
    r = requests.get("https://archive.org/advancedsearch.php", params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    docs = r.json().get("response", {}).get("docs", [])
    identifiers = [doc["identifier"] for doc in docs if doc.get("identifier")]
    _SOURCE_CACHE[cache_key] = identifiers

    if not identifiers:
        raise ValueError(f"No Internet Archive texts found for language '{lang}'")
    return identifiers

def get_internet_archive_text_files(identifier):
    cache_key = ("internet_archive_files", identifier)
    if cache_key in _SOURCE_CACHE:
        return _SOURCE_CACHE[cache_key]

    r = requests.get(f"https://archive.org/metadata/{identifier}", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    files = r.json().get("files", [])
    plain_text_urls = []
    for file_info in files:
        name = file_info.get("name", "")
        lower_name = name.lower()
        if lower_name.endswith("_djvu.txt") or (
            lower_name.endswith(".txt")
            and not lower_name.endswith("_meta.txt")
            and not lower_name.endswith("_files.xml")
        ):
            plain_text_urls.append(f"https://archive.org/download/{identifier}/{quote(name)}")

    _SOURCE_CACHE[cache_key] = plain_text_urls
    if not plain_text_urls:
        raise ValueError(f"No plain text files found for Internet Archive item '{identifier}'")
    return plain_text_urls

def download_internet_archive_text(lang):
    identifiers = get_internet_archive_identifiers(lang)
    identifier = random.choice(identifiers)
    urls = get_internet_archive_text_files(identifier)
    url = random.choice(urls)
    return download_text(url), url

def download_source_text(lang, source):
    if source == "gutenberg":
        return download_gutenberg_text(lang)
    if source == "wikisource":
        return download_wikisource_text(lang)
    if source == "internet_archive":
        return download_internet_archive_text(lang)
    raise ValueError(f"Unknown text source: {source}")

# -------------------------------
# 2. Завантаження
# -------------------------------
def download_text(url):
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

# -------------------------------
# 3. Обрізка Gutenberg
# -------------------------------
def strip_gutenberg(text):
    start = re.search(r"\*\*\* START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* END OF.*?\*\*\*", text)
    if start and end:
        return text[start.end():end.start()]
    return text

# -------------------------------
# 4. Очистка
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"'s\b", "", text)  # remove possessive 's
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t.isalpha()]  # ❗ прибираємо числа
    return tokens

# -------------------------------
# 5. Накопичення токенів
# -------------------------------
def collect_tokens(lang, sources=None):
    sources = sources or TEXT_SOURCES
    tokens = []
    attempts = 0
    while len(tokens) < TARGET_TOKENS and attempts < MAX_BOOK_ATTEMPTS:
        attempts += 1
        try:
            source = sources[(attempts - 1) % len(sources)]
            text, url = download_source_text(lang, source)
            print(f"[{lang}] {SOURCE_NAMES[source]}: {url}")
            new_tokens = preprocess(text)
            if not new_tokens:
                raise ValueError("Downloaded text did not contain alphabetic tokens")
            tokens.extend(new_tokens)
        except Exception:
            continue
    if len(tokens) < TARGET_TOKENS:
        raise RuntimeError(
            f"Could not collect enough tokens for '{lang}'. "
            f"Collected {len(tokens)} out of {TARGET_TOKENS} after {attempts} attempts "
            f"from sources: {', '.join(SOURCE_NAMES[s] for s in sources)}."
        )
    return tokens[:TARGET_TOKENS]

# -------------------------------
# 6. Лематизація
# -------------------------------
def lemmatize(tokens, lang):
    stanza.download(lang, processors="tokenize,lemma", verbose=False)
    try:
        nlp = stanza.Pipeline(lang=lang, processors="tokenize,lemma", use_gpu=True, verbose=False)
    except Exception:
        nlp = stanza.Pipeline(lang=lang, processors="tokenize,lemma", use_gpu=False, verbose=False)
    doc = nlp(" ".join(tokens))
    return [w.lemma for s in doc.sentences for w in s.words if w.lemma.isalpha()]

def check_gutenberg_availability(lang):
    try:
        books = get_books(lang)
    except Exception as e:
        return False, f"Gutenberg unavailable: {e}"

    plain_text_count = 0
    for book in books:
        if any("text/plain" in k for k in book.get("formats", {})):
            plain_text_count += 1

    if plain_text_count == 0:
        return False, "No text/plain books on first Gutendex page"
    return True, f"{plain_text_count} books with text/plain on first Gutendex page"

def check_wikisource_availability(lang):
    try:
        text, _ = download_wikisource_text(lang)
    except Exception as e:
        return False, f"Wikisource unavailable: {e}"

    token_count = len(preprocess(text))
    if token_count == 0:
        return False, "Wikisource returned no alphabetic tokens"
    return True, f"random page available ({token_count} tokens)"

def check_internet_archive_availability(lang):
    try:
        identifiers = get_internet_archive_identifiers(lang)
    except Exception as e:
        return False, f"Internet Archive unavailable: {e}"

    return True, f"{len(identifiers)} text items found"

def check_source_availability(lang, source):
    if source == "gutenberg":
        return check_gutenberg_availability(lang)
    if source == "wikisource":
        return check_wikisource_availability(lang)
    if source == "internet_archive":
        return check_internet_archive_availability(lang)
    raise ValueError(f"Unknown text source: {source}")

def check_stanza_availability(lang):
    try:
        stanza.download(lang, processors="tokenize,lemma", verbose=False)
        stanza.Pipeline(lang=lang, processors="tokenize,lemma", use_gpu=True, verbose=False)
        return True, "Stanza tokenize+lemma available"
    except Exception as e:
        return False, f"Stanza unavailable: {e}"

# -------------------------------
# 7. Zipf + збереження
# -------------------------------
def zipf_analysis(tokens, title, filename_prefix):
    freq = Counter(tokens)
    sorted_freq = sorted(freq.values(), reverse=True)

    ranks = np.arange(1, len(sorted_freq) + 1)
    freqs = np.array(sorted_freq)

    log_r = np.log(ranks)
    log_f = np.log(freqs)

    slope, intercept = np.polyfit(log_r, log_f, 1)
    s = -slope

    fitted_log_f = slope * log_r + intercept
    ss_res = np.sum((log_f - fitted_log_f) ** 2)
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # графік
    plt.figure()
    plt.scatter(log_r, log_f, s=5)
    plt.plot(log_r, fitted_log_f)
    plt.xlabel("log(rank)")
    plt.ylabel("log(freq)")
    plt.title(title)

    plot_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    plt.savefig(plot_path)
    plt.close()

    return s, r2

# -------------------------------
# 7.1. Zipf-Mandelbrot + збереження
# -------------------------------
def zipf_mandelbrot_analysis(tokens, title, filename_prefix):
    freq = Counter(tokens)
    sorted_freq = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freq) + 1, dtype=float)
    freqs = np.array(sorted_freq, dtype=float)
    log_f = np.log(freqs)

    best = None
    for q in np.linspace(0.0, Q_SEARCH_MAX, Q_SEARCH_STEPS):
        x = np.log(ranks + q)
        slope, intercept = np.polyfit(x, log_f, 1)
        fitted = slope * x + intercept
        sse = np.sum((log_f - fitted) ** 2)
        if best is None or sse < best["sse"]:
            best = {
                "q": float(q),
                "s": float(-slope),
                "log_c": float(intercept),
                "fitted": fitted,
                "sse": float(sse),
            }

    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    r2 = 1.0 - best["sse"] / ss_tot if ss_tot > 0 else 1.0
    c = float(np.exp(best["log_c"]))

    log_r = np.log(ranks)

    # графік: крива в координатах log(rank)-log(freq)
    plt.figure()
    plt.scatter(log_r, log_f, s=5, label="data")
    plt.plot(log_r, best["fitted"], label=f"fit q={best['q']:.2f}")
    plt.xlabel("log(rank)")
    plt.ylabel("log(freq)")
    plt.title(title)
    plt.legend()

    plot_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    plt.savefig(plot_path)
    plt.close()

    return best["s"], best["q"], c, r2

# -------------------------------
# 8. ТОП слова
# -------------------------------
def get_top(tokens):
    return Counter(tokens).most_common(TOP_N)

def token_word_stats(tokens):
    return len(tokens), len(set(tokens))

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    write_report(f"Experiment started: {timestamp}")
    write_report(f"Target tokens: {TARGET_TOKENS}\n")
    write_report("Language availability check:")

    process_languages = []
    lemma_supported = {}
    source_supported = {}

    for lang in LANGUAGES:
        source_supported[lang] = []
        for source in TEXT_SOURCES:
            source_ok, source_msg = check_source_availability(lang, source)
            write_report(f"{lang}: {SOURCE_NAMES[source]}={'OK' if source_ok else 'SKIP'} ({source_msg})")
            if source_ok:
                source_supported[lang].append(source)

        stz_ok, stz_msg = check_stanza_availability(lang)
        lemma_supported[lang] = stz_ok
        write_report(f"{lang}: Stanza={'OK' if stz_ok else 'SKIP'} ({stz_msg})")

        if source_supported[lang]:
            process_languages.append(lang)
        else:
            print(f"[SKIP] {lang} (no text sources)")

    if not process_languages:
        raise RuntimeError("No languages available for processing after availability checks.")

    for lang in process_languages:
        print(f"\n=== {lang} ===")
        write_report(f"\n=== LANGUAGE: {lang} ===")
        write_report("Text sources: " + ", ".join(SOURCE_NAMES[s] for s in source_supported[lang]))

        tokens = collect_tokens(lang, source_supported[lang])
        raw_token_count, raw_word_count = token_word_stats(tokens)

        # RAW
        s_raw, r2_raw = zipf_analysis(tokens, f"{lang} raw (Zipf)", f"{lang}_raw_zipf")
        s_m_raw, q_m_raw, c_m_raw, r2_m_raw = zipf_mandelbrot_analysis(
            tokens, f"{lang} raw (Zipf-Mandelbrot)", f"{lang}_raw_zipf_mandelbrot"
        )
        top_raw = get_top(tokens)

        write_report(f"\nRAW Zipf: s ≈ {s_raw:.4f}, R^2 ≈ {r2_raw:.4f}")
        write_report(
            f"RAW Zipf-Mandelbrot: s ≈ {s_m_raw:.4f}, q ≈ {q_m_raw:.4f}, C ≈ {c_m_raw:.4f}, R^2 ≈ {r2_m_raw:.4f}"
        )
        write_report(f"RAW analyzed: tokens = {raw_token_count}, unique words = {raw_word_count}")
        write_report("Top words (raw):")
        for w, c in top_raw:
            write_report(f"{w:15} {c}")

        # LEMMA
        if lemma_supported.get(lang, False):
            try:
                lemmas = lemmatize(tokens, lang)
                lemma_token_count, lemma_word_count = token_word_stats(lemmas)

                s_lemma, r2_lemma = zipf_analysis(lemmas, f"{lang} lemma (Zipf)", f"{lang}_lemma_zipf")
                s_m_lemma, q_m_lemma, c_m_lemma, r2_m_lemma = zipf_mandelbrot_analysis(
                    lemmas, f"{lang} lemma (Zipf-Mandelbrot)", f"{lang}_lemma_zipf_mandelbrot"
                )
                top_lemma = get_top(lemmas)

                write_report(f"\nLEMMA Zipf: s ≈ {s_lemma:.4f}, R^2 ≈ {r2_lemma:.4f}")
                write_report(
                    f"LEMMA Zipf-Mandelbrot: s ≈ {s_m_lemma:.4f}, q ≈ {q_m_lemma:.4f}, C ≈ {c_m_lemma:.4f}, R^2 ≈ {r2_m_lemma:.4f}"
                )
                write_report(f"LEMMA analyzed: tokens = {lemma_token_count}, unique words = {lemma_word_count}")
                write_report("Top words (lemma):")
                for w, c in top_lemma:
                    write_report(f"{w:15} {c}")
            except Exception as e:
                write_report(f"\nLEMMA skipped: Stanza failed at runtime ({e}). Raw-only analysis kept.")
        else:
            write_report("\nLEMMA skipped: Stanza tokenize+lemma unavailable for this language.")

    print(f"\nResults saved in: {OUTPUT_DIR}")
