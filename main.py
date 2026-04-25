import re
import requests
import random
import stanza
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import os

# -------------------------------
# НАЛАШТУВАННЯ
# -------------------------------
TARGET_TOKENS = 80000
TOP_N = 20
LANGUAGES = ["en", "fr", "de"]
Q_SEARCH_MAX = 50.0
Q_SEARCH_STEPS = 251

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
# 1. Gutenberg API
# -------------------------------
def get_books(lang):
    url = f"https://gutendex.com/books?languages={lang}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()["results"]

def get_random_book_url(lang):
    books = get_books(lang)
    while True:
        book = random.choice(books)
        for k, v in book["formats"].items():
            if "text/plain" in k:
                return v

# -------------------------------
# 2. Завантаження
# -------------------------------
def download_text(url):
    r = requests.get(url)
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
def collect_tokens(lang):
    tokens = []
    while len(tokens) < TARGET_TOKENS:
        url = get_random_book_url(lang)
        print(f"[{lang}] {url}")
        try:
            text = download_text(url)
            text = strip_gutenberg(text)
            tokens.extend(preprocess(text))
        except:
            continue
    return tokens[:TARGET_TOKENS]

# -------------------------------
# 6. Лематизація
# -------------------------------
def lemmatize(tokens, lang):
    stanza.download(lang)
    nlp = stanza.Pipeline(lang=lang, processors="tokenize,lemma", use_gpu=True)
    doc = nlp(" ".join(tokens))
    return [w.lemma for s in doc.sentences for w in s.words if w.lemma.isalpha()]

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

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    write_report(f"Experiment started: {timestamp}")
    write_report(f"Target tokens: {TARGET_TOKENS}\n")

    for lang in LANGUAGES:
        print(f"\n=== {lang} ===")
        write_report(f"\n=== LANGUAGE: {lang} ===")

        tokens = collect_tokens(lang)

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
        write_report("Top words (raw):")
        for w, c in top_raw:
            write_report(f"{w:15} {c}")

        # LEMMA
        lemmas = lemmatize(tokens, lang)

        s_lemma, r2_lemma = zipf_analysis(lemmas, f"{lang} lemma (Zipf)", f"{lang}_lemma_zipf")
        s_m_lemma, q_m_lemma, c_m_lemma, r2_m_lemma = zipf_mandelbrot_analysis(
            lemmas, f"{lang} lemma (Zipf-Mandelbrot)", f"{lang}_lemma_zipf_mandelbrot"
        )
        top_lemma = get_top(lemmas)

        write_report(f"\nLEMMA Zipf: s ≈ {s_lemma:.4f}, R^2 ≈ {r2_lemma:.4f}")
        write_report(
            f"LEMMA Zipf-Mandelbrot: s ≈ {s_m_lemma:.4f}, q ≈ {q_m_lemma:.4f}, C ≈ {c_m_lemma:.4f}, R^2 ≈ {r2_m_lemma:.4f}"
        )
        write_report("Top words (lemma):")
        for w, c in top_lemma:
            write_report(f"{w:15} {c}")

    print(f"\nResults saved in: {OUTPUT_DIR}")
