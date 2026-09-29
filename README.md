# Vocab Size Test

A browser test that estimates how many words you passively understand in a language (reading and listening, not speaking). It covers Spanish, Portuguese, French and Russian. It is a static site: no server, no accounts, no build step.

## What it measures

Passive vocabulary size, counted in word families (dictionary headwords: "walk", "walks" and "walked" count once). Each language has a frozen list of its 30,000 most frequent word families, ranked by frequency, and split into seven bands from very common to rare.

## How a sitting works

1. **Yes/no phase (about 3 minutes).** 15 real words per band, mixed with 35 made-up words that look plausible. You answer "I know it" or "I don't". The made-up words measure how often you claim words you cannot know, so guessing is corrected for.
2. **Multiple-choice phase (about 6 minutes).** 60 words, four English meanings plus "I don't know". Questions are concentrated on the bands where your knowledge is changing fastest.

## How scoring works

The estimator fits a curve, the chance you know a word as a function of the log of its frequency rank, to both phases at once by maximum likelihood. It corrects for false claims (yes/no) and for lucky guesses (multiple choice). The headline is the estimated number of known words among the top 16,000 word families. A 68% confidence interval comes from a bootstrap. Results show the curve and a per-band table. The estimator is pure JavaScript and unit-tested. Items used in the last 180 days are not reused, so retesting stays honest.

Sittings are saved only in your browser (localStorage).

## Run it

Open `index.html`, or serve the folder with any static server:

    python3 -m http.server 8000

It also works on GitHub Pages. Tests: `node tests/test-estimator.js`.

## Data sources and licenses

- **Code** (index.html, js/, css/, tests/, scripts/): MIT, see LICENSE.
- **Frequency-ranked word lists** (`content/lemmas-*.json`): built with `scripts/build_lemma_list.py` from the top 200,000 word forms of [wordfreq](https://github.com/rspeer/wordfreq) 3.1.1 (`large` list), lemmatized with [simplemma](https://github.com/adbar/simplemma) 1.2.0 (MIT). wordfreq's code is Apache-2.0 and its data is redistributable under CC BY-SA 4.0; it combines Wikipedia, OpenSubtitles, news, books and web frequencies (credits in its README). These lists are only ranked headwords, no counts, and are shared under **CC BY-SA 4.0**.
- **Question banks, made-up words and blocklists** (`content/bank-*.json`, `blocklist-*.json`): generated with Claude (Anthropic) from the lists above, with mechanical checks, and shared under **CC BY-SA 4.0**. The blocklists mark proper nouns, abbreviations and junk that sit in the frequency lists.

Armenian was built for the original app but is not included, because part of its source data has unclear redistribution terms.
