# qsim - algorithm-only question similarity chatbot

No LLM, no encoder, no training. A **frozen spaCy `en_core_web_md` pipeline** (tagger, lemmatiser, parser, entity recogniser,
static word vectors) plus **deterministic keyword / slot matching**. The same text always gives the same answer.

## Run (Windows PowerShell)
```powershell
py -m pip install numpy spacy
py -m spacy download en_core_web_md
py -m pip install nltk                      # optional, WordNet synonyms
py -m nltk.downloader wordnet               # optional
py -m qsim ui                               # chatbot page at http://localhost:8000, similarity report in the terminal
py -m qsim chat                             # terminal chat
py -m qsim compare "question 1" "question 2"
```

## How a pair is judged
1. Spell-fix against the KB vocabulary, then spaCy: keywords (lemmas of nouns/verbs/adjectives, phrasal verbs merged), numbers, names, question type, negation, time reference (last/this/next).
2. Align keywords: same stem, synonym (KB `synonyms` groups, WordNet shared lemma/derived form), or static-vector cosine >= 0.70. IDF weights come from the KB.
3. Score = harmonic mean of the weighted share of each question's keywords that found a partner.
4. Hard conflicts cap the score and block SAME: different number / name / question type / time reference, negation, a number in only one question,
   KB-learned alternatives (swim vs bike vs run: learned from KB questions that differ by one word), a clearly different single noun.
5. SAME >= 0.55 with no hard conflict; RELATED >= 0.30; else UNRELATED. Thresholds: `qsim/config.py`.

## The knowledge base (`data/knowledge_base.json`)
`{"qa": [{"question", "answer", "paraphrases": [...]}], "synonyms": [[word, ...]], "names": ["Goa", "Lake Placid"]}`. `names` lists places/events so they are recognised even when typed in lower case. `paraphrases` and `synonyms` are the **recall lever**:
algorithms alone cannot know that "cut off" = "time limit" in your domain, so such knowledge is data you can extend without touching code.
The shipped answers are SAMPLES.

## Known limits
Role swaps ("athletes" vs "spectators"), word order ("relay a team" vs "team relay"), and loose paraphrases not covered by a paraphrase/synonym entry.

## Use in your own UI
```python
from qsim.bot import QABot
bot = QABot()
turn = bot.ask(question)    # turn.answer, turn.source = session | knowledge_base | none
```
