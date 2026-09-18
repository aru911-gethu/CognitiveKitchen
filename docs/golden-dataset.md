# Building a truth set

How `data/golden_dataset.json` was built, and how to build one for your own
corpus. This is the most reusable thing in the repo and the only part that cannot
be generated.

## Why it is required at all

The golden dataset is the answer key. It is read only at scoring time, never by
the pipeline that produces answers, because a benchmark the system can see is a
benchmark the system will fit.

Not every metric needs it. The split is between **questions** and **labels**:

```
documents only        chunk purity, self-sufficiency, latency, cost
+ a question list     faithfulness, relevancy, cookability, abstention,
                      diversity, constraint compliance
+ relevance labels    recall, hit@k, MAP, k@90, set recall
```

**You can skip this document entirely** and still measure most of the pipeline,
including the constraint metric. Put one question per line in
`data/questions.txt` and the evaluators use it; the label-dependent metrics report
`None` rather than a misleading zero.

Labels buy you recall, and recall cannot be obtained any other way. An LLM judge
can rate whether a retrieved document is relevant, but it can never tell you about
a document that was *not* retrieved, because it only sees what you show it. That
is bookkeeping, not reasoning, and no model quality fixes it.

## Shape

```
schema_version
built_at
source
provenance     method, human_adjudicated_lines, mechanical_text_fixes,
               corrections_to_previous_build
validation     verified_at, checks, known_caveats
selection      how these recipes were chosen out of the corpus
recipes[]      50 fully structured records
evaluation     built_at, corpus_size, notes, queries[]
corpus         corpus_characters, corpus_sha256, separator, spans[], unattributable
substitutions  synonyms and substitution candidates
moods          structural claims, e.g. cooling = yogurt / cucumber / coconut milk
```

Each recipe:

```
id, title, course, region, source_pages[], servings_text,
total_time_text, total_time_minutes, timed_steps_minutes_sum,
effective_time_minutes, ingredients[], method[], tags[], notes,
confusable_with[]
```

`confusable_with` matters more than it looks. Two chutneys that differ only by one
ingredient are the hardest case for retrieval, and naming the pairs up front is
what lets a near-miss be scored as a near-miss rather than a hit.

## Queries

229 queries across seven families. Each one is
`query_id`, `family`, `question`, `relevant_ids[]`, `rationale`.

```
by_dish          52   "How do I make Coconut Chutney - North Indian (fresh)?"
by_ingredient    58   "What can I cook with cumin?"
exclusion        40   "Something with no nuts"
constraint       14   "Under 30 minutes with fewer than 6 ingredients"
course           26   "A bread dish"
substitution     24   "What can I use instead of ghee?"
ambiguous        15   questions with no single right answer
```

The `rationale` field is not decoration. It records *why* a recipe is a valid
answer, which is what makes a disagreement reviewable later instead of a matter of
memory.

**The families are the point.** A single mixed pool of questions produces one
average that hides everything. Splitting by family is what exposed that
`constraint` scores 0.000 for every retriever — all 14 of its questions compare
numbers, and nothing in the pipeline does. A blended average would have buried
that as a slightly lower score.

## Method

1. **Pick a subset you can actually verify.** 50 recipes out of 184. Not a
   sample for statistical reasons — a sample small enough to check by hand.

2. **Rebuild each record from the source, do not copy the pipeline's output.**
   If the truth set is derived from the thing being tested, it agrees with it by
   construction. Every field here was re-read from the PDF.

3. **Recompute derived fields.** `effective_time_minutes` and
   `timed_steps_minutes_sum` are calculated, not transcribed, so a stated time
   that contradicts the steps is visible.

4. **Pin the corpus.** `corpus_sha256` and `corpus_characters` fix exactly which
   text the labels refer to. Re-ingest with a different parser and the hash tells
   you the labels may no longer apply.

5. **Record what a human decided.** `human_adjudicated_lines` and
   `mechanical_text_fixes` separate judgement calls from typo repairs.

6. **Write the checks down.** `validation.checks` records what was verified: no
   missing ingredients, nothing fabricated, every derived field recomputed.
   `known_caveats` records what was not.

7. **Never correct silently.** Which brings us to the part that matters most.

## Assume your truth set is wrong

It was. Twice.

Queries `q_0119` and `q_0129` listed fish and prawn recipes (`ck_041`, `ck_042`)
as valid answers to *"no meat"*. Someone avoiding meat does not want prawns.

Both are corrected, and the correction is recorded in
`provenance.corrections_to_previous_build` rather than quietly fixed. That record
is the reason the rest of the numbers can be trusted: it shows the truth set is
treated as a thing that can be wrong and gets audited, not as scripture.

**This is the argument against generated labels.** Had those labels come from a
model and been trusted, the error would have silently inflated the compliance
score of whichever retriever happened to return those two recipes — and there
would have been no way to notice. Generated labels are a fine starting point.
Ratification is not the optional half.

## For your own corpus

```
1  pick 30-60 documents you can personally verify
2  rebuild each record from the source, not from your pipeline
3  write questions in families that reflect how your users actually ask,
   and include at least one family that is about exclusion
4  record the rationale for every label
5  pin a corpus hash
6  keep a corrections block from day one
```

Step 3 is where most of the value is. If no family in your set asks "find me
something *without* X", you will not discover the failure this project exists to
demonstrate — because none of the standard metrics look for it.