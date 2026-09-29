#!/usr/bin/env python3
"""Build content/lemmas-<lang>.json — the frozen, versioned lemma frequency list.

Usage: python3 scripts/build_lemma_list.py [--lang es|pt]
  (default --lang es, identical invocation/behavior to the original es-only script)

Pipeline (see README.md):
  1. Pull the top 200,000 surface forms for the language from wordfreq's 'large'
     wordlist, with each form's frequency.
  2. Filter forms to plausible plain orthography for that language.
  3. Lemmatize each surviving form with simplemma; drop forms whose lemma doesn't
     also pass the filter (empty, multiword, or disallowed characters).
  4. Aggregate: sum the frequencies of every form that maps to the same lemma.
  5. Rank lemmas by aggregated frequency descending, tie-break alphabetically so the
     result is fully deterministic.
  6. Take the top 30,000 and write content/lemmas-<lang>.json in the exact schema the
     rest of the app depends on (rank of lemmas[i] is i+1).

No network access: wordfreq's word lists ship inside the installed package.
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

try:
    import wordfreq
except ImportError:
    print("ERROR: wordfreq is not importable with this python3. Install it first "
          "(pip install --user wordfreq).", file=sys.stderr)
    raise

try:
    import simplemma
except ImportError:
    print("ERROR: simplemma is not importable with this python3. Install it first "
          "(pip install --user simplemma).", file=sys.stderr)
    raise

FORM_COUNT = 200_000
TOP_N = 30_000
WORDLIST = "large"

# Per-language config table. Everything else (length >= 2 rule, aggregation,
# deterministic tie-break, top 30,000, output schema) is identical across languages;
# only these knobs vary.
LANG_CONFIG = {
    "es": {
        "wordfreq_lang": "es",
        "simplemma_lang": "es",
        "token_re": re.compile(r"^[a-záéíóúüñ]+$"),
        "one_letter_ok": {"y", "a", "o", "e", "u"},
        "list_version": "es-lemmas-v1",
        "source": "wordfreq-large+simplemma",
        # Known simplemma over-collapse (verified empirically, simplemma 1.2.0): its
        # Spanish dictionary maps the feminine definite article / feminine object
        # pronoun "la" (and its plural "las") onto the masculine lemma "el", silently
        # erasing one of the most frequent words in the language from the list. That
        # conflicts with this project's "unit = lemmas (dictionary headwords)" design
        # (the design) -- every Spanish dictionary lists "el" and "la" as distinct
        # headwords -- and with "la" needing to rank in the top 10. This is the one
        # narrow, explicit carve-out from otherwise pure simplemma output; other
        # similar minor simplemma imperfections (e.g. lo/le/les collapsing to "él")
        # are left alone as tolerated, lower-frequency noise.
        "lemma_overrides": {"la": "la", "las": "la"},
    },
    "pt": {
        "wordfreq_lang": "pt",
        "simplemma_lang": "pt",
        "token_re": re.compile(r"^[a-záéíóúâêôãõàçü]+$"),
        # Verified empirically 2026-08-23 against wordfreq.top_n_list('pt', 200000,
        # wordlist='large'): the only one-letter *forms* that appear near the top of
        # the raw pt form list are exactly a(#2) o(#3) e(#5) é(#9) à(#37) -- the
        # definite articles / conjunction "e" (and) / copula "é" (is) / contraction
        # "à" (to the, fem.) -- matching the whitelist below with no adjustment
        # needed. Every other one-letter token that shows up before rank ~2000
        # (d, i, c, r, p, s, x, n, m, b, v, l, t, q, y, f, h, k, g, j, u, plus digits
        # and stray marks like ° ♪ º) is subtitle/OCR noise, not a real Portuguese
        # word, and is correctly dropped by NOT being in this whitelist.
        "one_letter_ok": {"e", "o", "a", "à", "é"},
        "list_version": "pt-lemmas-v1",
        "source": "wordfreq-large+simplemma",
        # Same narrow-override discipline as es's la/las->la fix, for the two
        # simplemma quirks that put a NON-WORD in the pt top 20: the article
        # "uma" was folded into the nonexistent "umar" (a fake headword the
        # yes/no phase could then show as a "real" word), and the preposition
        # "para" — one of the most common words in the language — was swallowed
        # by the verb "parar". "uma"/"umas" keep their own headword (the la
        # precedent) and "para" stays itself. Every other simplemma
        # imperfection is tolerated noise, not chased.
        "lemma_overrides": {"para": "para", "uma": "uma", "umas": "uma"},
    },
    "ru": {
        "wordfreq_lang": "ru",
        "simplemma_lang": "ru",
        "token_re": re.compile(r"^[а-яё]+$"),
        # Verified empirically 2026-08-23 against wordfreq.top_n_list('ru', 200000,
        # wordlist='large'): the only one-letter *forms* near the top of the raw ru
        # form list are the prepositions/conjunction/pronoun в(#1) и(#2) с(#5) я(#7)
        # а(#9) о(#15) к(#16) у(#19) -- all genuine one-letter Russian words, and all
        # within the raw top 19. Every other single Cyrillic letter that appears
        # before rank ~2000 (г, м, п, н, е, т, х, б, д, й, р, ж, л, э, ф, ю, ч, ...)
        # is an abbreviation marker or letter-mention, not a real word, starting at
        # rank 82 ('г') -- a clean break, correctly dropped by NOT being whitelisted.
        "one_letter_ok": {"и", "в", "я", "с", "а", "о", "у", "к"},
        "list_version": "ru-lemmas-v1",
        "source": "wordfreq-large+simplemma",
        # CRITICAL, verified 2026-08-23: wordfreq does NOT support 'hy' (Armenian) --
        # it silently falls back to this same 'ru' data with only a stderr warning
        # ("Using the nearest match, which is 'ru'"). 'hy' is deliberately NOT a key
        # in this table; Armenian is built from a different source by a different
        # worker. See the top_word script-guard in build() for the mechanical check
        # that stops any language from silently reusing another's corpus.
        #
        # simplemma 1.2.0's Russian dictionary has several confirmed bugs where a
        # common word's *own surface form* fails to lemmatize to itself and is
        # instead redirected to an unrelated (and usually much rarer) word --
        # the same failure class as pt's uma->umar, just with more instances because
        # Russian's paradigms are richer. Found by hand-checking the top 100 lemmas
        # plus a systematic scan (own standalone wordfreq frequency vs. aggregated
        # frequency) through rank 500 for aggregate frequency > 0.0003. Each fix
        # below overrides only the specific broken surface form(s); every other,
        # correctly-lemmatizing form of the same word (e.g. "дня", "дни", "дела",
        # "такая", "моя") is left to simplemma as before -- only the exact form
        # that misfires is redirected to its real, already-existing headword:
        #   день/дней -> деть (a rare verb "to put") instead of день ("day")
        #   дело -> деть also, instead of дело ("matter/thing")
        #   буду -> дуть ("to blow") instead of быть ("to be", future 1st singular)
        #   будет/будешь/будем/будете/будут -> бздеть (vulgar/slang) instead of быть
        #     (future tense of "to be" -- these are extremely common forms)
        #   были -> быль (a rare noun "a true story") instead of быть ("were")
        #   нет -> житься (a rare reflexive verb) instead of нет ("no" -- one of the
        #     most common words in the language, own form maps to the wrong target)
        #   после -> посол ("ambassador") instead of после ("after")
        #   так/такой/таки -> така (a rare dialectal form) instead of themselves
        #     ("so/thus", "such", the particle "still/indeed" -- three distinct,
        #     very common headwords all mis-collapsed into one non-standard form)
        #   мой -> мыть ("to wash") instead of мой ("my" -- the possessive pronoun;
        #     every other case form, моя/моё/мои/..., correctly lemmatizes to мой)
        #   потом -> пот ("sweat") instead of потом ("afterward/then")
        #   почти -> почтить ("to honor") instead of почти ("almost")
        #   три -> тереть ("to rub") instead of три ("three")
        #   тут -> тута (a rare dialectal variant) instead of тут ("here")
        # Other simplemma imperfections of the normal, EXPECTED kind (an irregular
        # verb/adjective infinitive having lower standalone frequency than its
        # common conjugated forms, e.g. мочь<-может, хотеть<-хочу, должный<-должны,
        # нужный<-нужно, являться<-является -- all correct Russian morphology) are
        # left alone as tolerated, lower-frequency noise, same discipline as es/pt.
        "lemma_overrides": {
            "день": "день", "дней": "день", "дело": "дело",
            "буду": "быть", "будешь": "быть", "будет": "быть", "будем": "быть",
            "будете": "быть", "будут": "быть", "были": "быть",
            "нет": "нет", "после": "после",
            "так": "так", "такой": "такой", "таки": "таки",
            "мой": "мой", "потом": "потом", "почти": "почти",
            "три": "три", "тут": "тут",
        },
    },
    "fr": {
        "wordfreq_lang": "fr",
        "simplemma_lang": "fr",
        "token_re": re.compile(r"^[a-zàâäéèêëîïôöùûüÿçœæ]+$"),
        # Verified empirically 2026-08-23 against wordfreq.top_n_list('fr', 200000,
        # wordlist='large'): the real one-letter fr *words* near the top of the raw
        # form list are à(#6, preposition "to/at"), a(#14, "has", 3rd sing. of
        # avoir), y(#43, pronoun "there/of it"), and ô(#7139, rare literary vocative
        # interjection "oh", explicitly checked and confirmed real despite its low
        # frequency). Every other single-letter token near the top of the raw list
        # -- l(#5), d(#11), c(#23), j(#33), n(#34), s(#38), t(#62), m(#63) -- is a
        # bare ELISION STEM left over from l'/d'/c'/j'/n'/s'/t'/m' (the apostrophe
        # isn't in the allowed character set, so the fragment survives stage 1 as
        # a lone letter); these are correctly dropped by the length rule since the
        # full elided forms (le, de, ce, je, ne, se, te, me) already rank high on
        # their own. Remaining single letters (p, i, h, e, x, r, b, o, v, g, f, u,
        # k, w, z, q, é, ç, â, è, ...) are OCR/abbreviation noise, not real words.
        "one_letter_ok": {"a", "à", "y", "ô"},
        "list_version": "fr-lemmas-v1",
        "source": "wordfreq-large+simplemma",
        # Same narrow-override discipline as es's la/las->la fix and ru's above:
        # simplemma 1.2.0's French dictionary has the identical la/le over-collapse
        # bug as Spanish -- le/la/les ALL lemmatize to "le", which would silently
        # erase "la" (the feminine definite article/pronoun, one of the most
        # frequent words in French) from the list entirely (confirmed: with no
        # override, "la" does not appear anywhere in the top 60). "les" is left
        # folded into "le" as tolerated noise (mirrors es tolerating "los"->"el"),
        # matching the project's established minimal-intervention precedent.
        # Two further confirmed own-form bugs of the same class as ru's, found by
        # hand-checking the top 100 plus the same systematic frequency-ratio scan:
        #   cela -> celer (a near-unused archaic verb "to conceal") instead of cela
        #     ("this/that" -- a very common demonstrative pronoun)
        #   étais -> étai (a real but obscure carpentry noun "prop, strut") instead
        #     of être ("was/were", 1st/2nd singular imperfect of "to be" -- every
        #     other imperfect form, était/étions/étiez/étaient, correctly maps to
        #     être already, only this one form misfires)
        "lemma_overrides": {"la": "la", "cela": "cela", "étais": "être"},
    },
}

REPO_ROOT = Path(__file__).resolve().parent.parent
BAND_EDGES = [500, 1000, 2000, 4000, 8000, 16000, 30000]


def is_valid_token(tok: str, token_re: re.Pattern, one_letter_ok: set[str]) -> bool:
    """Mechanical filter shared by forms and lemmas per the data contract in README.md."""
    if not token_re.match(tok):
        return False
    if len(tok) >= 2:
        return True
    return tok in one_letter_ok


def build(lang: str, out_path: Path | None = None) -> None:
    cfg = LANG_CONFIG[lang]
    wf_lang = cfg["wordfreq_lang"]
    sm_lang = cfg["simplemma_lang"]
    token_re = cfg["token_re"]
    one_letter_ok = cfg["one_letter_ok"]
    lemma_overrides = cfg["lemma_overrides"]

    if out_path is None:
        out_path = REPO_ROOT / "content" / f"lemmas-{lang}.json"

    # Hard guard against a silent wrong-corpus fallback (verified 2026-08-23: wordfreq
    # does not support 'hy' and silently substitutes 'ru' data with only a stderr
    # warning -- no exception, no obviously-wrong return value). If wordfreq's own top
    # word for wf_lang doesn't even match this language's script/alphabet regex, the
    # requested language almost certainly isn't really backed by wordfreq at all.
    guard_word = wordfreq.top_n_list(wf_lang, 10, wordlist=WORDLIST)[0]
    if not token_re.match(guard_word):
        print(f"ERROR: wordfreq.top_n_list({wf_lang!r}, 10, wordlist={WORDLIST!r})[0] = "
              f"{guard_word!r}, which does not match the {lang} script/alphabet regex "
              f"{token_re.pattern!r}. This is the signature of wordfreq silently falling "
              f"back to a different language's corpus (confirmed for 'hy' -> 'ru') rather "
              f"than actually supporting {wf_lang!r}. Refusing to build -- do not add a "
              f"language to LANG_CONFIG without first confirming wordfreq truly supports "
              f"it (wf_lang in wordfreq.available_languages(wordlist='large')).",
              file=sys.stderr)
        sys.exit(1)

    print(f"Pulling top {FORM_COUNT:,} {lang} forms from wordfreq ({WORDLIST})...")
    forms = wordfreq.top_n_list(wf_lang, FORM_COUNT, wordlist=WORDLIST)
    print(f"  got {len(forms):,} forms")

    # Stage 1: mechanical form filter.
    surviving_forms = [f for f in forms if is_valid_token(f, token_re, one_letter_ok)]
    print(f"Stage 1 (form regex/length filter): {len(surviving_forms):,} / {len(forms):,} survive")

    # Stage 2: lemmatize, filter lemma, aggregate frequency per lemma.
    lemma_freq: dict[str, float] = {}
    lemmatized_ok = 0
    dropped_bad_lemma = 0
    dropped_empty_or_multiword = 0

    for form in surviving_forms:
        freq = wordfreq.word_frequency(form, wf_lang, wordlist=WORDLIST)
        if freq <= 0:
            continue
        if form in lemma_overrides:
            lemma = lemma_overrides[form]
        else:
            try:
                lemma = simplemma.lemmatize(form, lang=sm_lang)
            except Exception:
                dropped_bad_lemma += 1
                continue

        if lemma is None:
            dropped_empty_or_multiword += 1
            continue

        lemma = lemma.lower().strip()

        if not lemma or " " in lemma or "-" in lemma:
            dropped_empty_or_multiword += 1
            continue

        if not is_valid_token(lemma, token_re, one_letter_ok):
            dropped_bad_lemma += 1
            continue

        lemma_freq[lemma] = lemma_freq.get(lemma, 0.0) + freq
        lemmatized_ok += 1

    print(f"Stage 2 (lemmatize + lemma filter): {lemmatized_ok:,} forms mapped to a valid lemma")
    print(f"  dropped (empty/multiword lemma): {dropped_empty_or_multiword:,}")
    print(f"  dropped (lemma fails regex/length): {dropped_bad_lemma:,}")
    print(f"Distinct lemmas before cut: {len(lemma_freq):,}")

    # Stage 3: rank descending by frequency, tie-break alphabetically (deterministic).
    ranked = sorted(lemma_freq.items(), key=lambda kv: (-kv[1], kv[0]))
    top_lemmas = [lemma for lemma, _freq in ranked[:TOP_N]]

    if len(top_lemmas) < TOP_N:
        print(f"WARNING: only {len(top_lemmas):,} distinct lemmas survived, "
              f"fewer than the requested {TOP_N:,}.", file=sys.stderr)

    out = {
        "language": lang,
        "listVersion": cfg["list_version"],
        "source": cfg["source"],
        "built": datetime.date.today().isoformat(),
        "maxRank": len(top_lemmas),
        "bandEdges": BAND_EDGES,
        "lemmas": top_lemmas,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"\nWrote {out_path} ({size_kb:.1f} KB), {len(top_lemmas):,} lemmas")

    # QA report.
    print("\n=== QA REPORT ===")
    print(f"Forms pulled from wordfreq:            {len(forms):,}")
    print(f"Forms surviving regex/length filter:   {len(surviving_forms):,}")
    print(f"Forms surviving lemmatize+lemma filter:{lemmatized_ok:,}")
    print(f"Distinct lemmas before top-{TOP_N:,} cut: {len(lemma_freq):,}")
    print(f"Lemmas written (final list size):      {len(top_lemmas):,}")

    print(f"\nTop 20 lemmas:")
    for i, lemma in enumerate(top_lemmas[:20], start=1):
        print(f"  {i:>3}. {lemma}")

    print("\nSample ranks:")
    for rank in (100, 1000, 5000, 10000, 20000, 30000):
        if rank <= len(top_lemmas):
            print(f"  rank {rank:>6,}: {top_lemmas[rank - 1]}")
        else:
            print(f"  rank {rank:>6,}: <list only has {len(top_lemmas):,} entries>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(LANG_CONFIG.keys()), default="es",
                         help="language to build (default: es)")
    parser.add_argument("--out", type=Path, default=None,
                         help="override output path (default: content/lemmas-<lang>.json)")
    args = parser.parse_args()
    build(args.lang, out_path=args.out)


if __name__ == "__main__":
    main()
