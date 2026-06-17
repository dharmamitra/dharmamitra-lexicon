# dharmamitra-lexicon

Word and phrase level translation spans between Sanskrit, Tibetan and Chinese Buddhist
canonical texts, mined from aligned parallel text. Each record is one attested occurrence
of one source term together with its translation span in the target, with character offsets.

## Layout

```
<language-pair>/<src_text>_<tgt_text>.jsonl
```

Language pairs are sa-bo (Sanskrit to Tibetan), sa-zh (Sanskrit to Chinese) and zh-bo
(Chinese to Tibetan). Files are grouped by the source text and target text they come from,
where the text id is the segment id up to the colon (for example SA_T06_bsa065:36 gives the
text SA_T06_bsa065). Any file larger than 90 MB is split into parts named
`<src_text>_<tgt_text>-01.jsonl`, `-02.jsonl` and so on; files that fit in one part have no
numeric suffix.

## Record fields

One JSON object per line.

```
type         token (single word) or phrase (multi word, compound or collocation)
head         the headword (lemma for tokens, surface form for phrases)
lemma        source lemma (Sanskrit pairs)
sanskrit     source surface form (Sanskrit pairs)
zh / bo      the source or target word, depending on the pair
indices      source token indices the span covers
start, end   character offsets of the span inside the target sentence
sentence_id  combined source and target segment id
src_segnr    source segment numbers
tgt_segnr    target segment numbers
source       provenance of the alignment (model, gemini, or cache)
target       the full target sentence
src_sentence the full source sentence
```

## How it was created

1. Parallel sentence pairs are mined from aligned canonical text, then filtered by length
   ratio and lexical overlap so only plausible translations remain.
2. The source side is enriched. Sanskrit is lemmatized and segmented. Chinese is word
   segmented with a finetuned segmentation model.
3. Span projection. A finetuned Gemma model reads the segmented source and the target
   sentence and inserts numbered tags into the target, one span per source token, marking
   the stretch of target text that translates that token. Every predicted span is required
   to be a literal substring of the target sentence, which removes hallucinated output.
4. For sa-bo and sa-zh, low confidence sentences (verse, low coverage, low model confidence)
   were re-aligned with gemini-3.1-flash-lite. For zh-bo the finetuned model alone was used,
   since a blind LLM judge found the redo step gave no measurable improvement over it.
5. Multi word terms are reconstructed by stitching adjacent source tokens (compounds and
   their contiguous sub compounds for Sanskrit, real adjacency in the Chinese for Chinese
   source), and the resulting phrase spans are recorded alongside the single word spans.
6. For every occurrence the character offsets of the span inside the target sentence are
   stored so the match can be highlighted in the original text.

Statistical aligners (eflomal) and neural embedding aligners (awesome-align) were evaluated
against this method on the same data and scored well below it in a blind LLM arena, so the
span projection output is what is released here.
