# Zipf Law Multilingual Text Analysis

## Overview

This project implements an automated pipeline for empirical verification of the **Zipf law** on large-scale textual data across multiple languages.

The program:

* downloads random texts from Project Gutenberg
* preprocesses and cleans the data
* optionally performs lemmatization using Stanza
* computes word frequency distributions
* evaluates Zipf’s law via log-log regression
* generates plots and a structured report

The entire pipeline is designed to be **reproducible, language-agnostic, and statistically meaningful**.

---

## Features

### 1. Randomized Text Sampling

* Texts are selected randomly via the Gutenberg catalog API
* Avoids manual selection bias
* Ensures variability between runs

### 2. Multilingual Support

* Supports a configurable list of languages (e.g. English, French, German)
* Each language is processed independently
* Enables cross-linguistic comparison

---

### 3. Data Collection Until Target Size

* The program accumulates tokens from multiple texts
* Stops when a predefined threshold is reached (e.g. 80,000 words)
* Ensures comparable dataset sizes across languages

---

### 4. Text Preprocessing

Includes:

* lowercasing
* removal of punctuation
* removal of numeric tokens
* filtering non-alphabetic tokens

This step ensures that:

* only linguistic units are analyzed
* statistical noise is reduced

---

### 5. Lemmatization (Optional but Enabled)

Performed using Stanza:

* Converts words to base forms (lemmas)
* Reduces morphological variability
* Improves comparability across languages

Example:

```
"running" → "run"
"books" → "book"
```

---

### 6. Zipf Law Analysis

For each dataset:

* words are ranked by frequency
* log(rank) vs log(frequency) is computed
* linear regression is applied

Zipf parameter is estimated as:

$$
s \approx -\text{slope}
$$

---

### 7. Output Generation

Each run creates a timestamped directory:

```
results_YYYY-MM-DD_HH-MM-SS/
```

Containing:

#### Plots

* log-log Zipf plots for:

  * raw tokens
  * lemmatized tokens

#### Report (`report.txt`)

For each language:

* total number of tokens
* estimated Zipf parameter (s)
* list of most frequent words (Top N)

---

## Example Output Structure

```
results_2026-04-19_21-37-12/
│
├── report.txt
├── en_raw.png
├── en_lemma.png
├── fr_raw.png
├── fr_lemma.png
├── de_raw.png
└── de_lemma.png
```

---

## Methodological Notes

* Random sampling approximates a **corpus-based approach**
* Filtering numeric tokens avoids distortion of frequency distributions
* Lemmatization allows analysis at the **lexeme level** rather than surface forms
* Log-log regression is used as a first-order approximation of a power-law

---

## Limitations

* Gutenberg texts are not perfectly balanced across genres
* Random sampling is limited by API pagination
* Lemmatization quality depends on language models
* Zipf’s law is approximate (deviations expected in head and tail)

---

## Possible Extensions

* statistical goodness-of-fit tests (e.g. Kolmogorov–Smirnov)
* comparison with alternative distributions (log-normal, exponential)
* stop-word filtering experiments
* bootstrap estimation of parameter stability
* larger-scale corpus integration (e.g. web corpora)

---

## Requirements

* Python 3.8+
* `stanza`
* `numpy`
* `matplotlib`
* `requests`

---

## Summary

This project provides a complete experimental framework for analyzing word frequency distributions and empirically testing the Zipf law across languages, combining:

* automated data acquisition
* linguistic normalization
* statistical modeling
* reproducible outputs

It is suitable for coursework in probability theory, statistics, or natural language processing.
