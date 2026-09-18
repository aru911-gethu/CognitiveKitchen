# Product

What this is for, who needs it, and where it stops working.

## The job

*I have documents. I need an assistant over them that I can defend. There are
roughly 300 architecture combinations and no basis on which to choose.*

Most RAG tooling helps with the building. This scores every choice, tells you
which to use and why, and hands you the running assistant plus a list of what it
still cannot do.

```
input     your documents.  a question list is optional but unlocks more metrics.
output    a configured, running assistant
          + the evidence behind every choice
          + a named list of what is still broken
```

## The problem worth solving

RAG systems are evaluated on quality: did retrieval find relevant text, did the
answer stay faithful to it. Nothing in the standard metric set asks whether the
system respected something the user ruled out.

That is not a small gap. Asked *"I am avoiding nuts, what can I make?"*, the
highest-scoring retriever in this lab returned Nut Milk, Cashew Nut Chutney and
Almond Honey Milk. Four standard metrics rated that configuration well. It
satisfied the stated constraint 40% of the time.

The mechanism is known in the literature as semantic collapse: similarity has no
direction for *without*, so "avoiding nuts" sits closest to the recipes that are
mostly nuts. The better the ranker, the more confidently wrong. See the Related
Work section of the README.

**What is not addressed in the literature is the tooling.** Four open RAG
evaluation tools were checked. None of them measure it:

```
ChunkLab               P@k, R@k, MRR, MAP, NDCG@k
enterprise-rag-bench   faithfulness, relevance, groundedness, context precision
crag-bench             accuracy
google/rag-playground  Vertex Rapid Evals + human votes
this project           + constraint_respected
```

## Who needs this most

Anywhere an exclusion carries liability:

| domain | the exclusion | cost of getting it wrong |
|---|---|---|
| food service, meal planning | allergens | hospitalisation |
| clinical decision support | contraindications, interactions | patient harm |
| legal, contracts | carve-outs, excluded jurisdictions | malpractice |
| financial compliance | restricted instrument lists | regulatory penalty |
| insurance | policy exclusions | wrongful denial |

Cooking is the demo because allergens are the most immediately legible form of a
liability-bearing exclusion. The mechanism is identical for a contract clause
that carves something out.

## North star metric

**Stated-exclusion compliance rate** — of the occasions a user told the system
what they cannot have, how often did the answer respect it?

This is deliberately not a system metric. hit@5 and faithfulness describe the
machine's health. This describes the user's experience of being ignored or
endangered, and it is the number the graph route moves from 40% to 100%.

## Adoption: what it costs to try

The honest barrier is the truth set. Three levels, and only one of them is a real
prerequisite:

| level | you provide | you get | state |
|---|---|---|---|
| **Structural** | documents only | chunk purity, self-sufficiency, latency, cost | works |
| **Unlabelled** | + a plain question list | + faithfulness, relevancy, cookability, abstention, **constraint compliance** | works |
| **Ratified** | + relevance judgements | + recall, hit@k, MAP, k@90, set recall | works |

The important line is the middle one. **The differentiating metric does not need
labels** — verified by hiding the golden dataset and running on three plain
questions, which produced `constraint_respected 0.9` with `hit_at_k None`.

Recall is why labels cannot be dropped entirely, and why an LLM judge cannot
replace them: to know a retriever missed something you have to know what existed,
and a judge only ever sees what you showed it. That is a bookkeeping problem, not
a reasoning problem, and no model quality fixes it.

Generating candidate labels from the corpus and having a human ratify them is the
obvious next step and is **not built**. See the scaling ceiling below for why the
ratification half is not optional.

## Competitive position

Two markets, and the position differs in each.

**Cooking RAG assistants** — crowded and shallow. ChefAssistAI, recipe_rag_agent,
recipe-generator, AllerGenie. None measure whether a dietary constraint was
respected, which is the thing that matters most in the domain.

**RAG evaluation tooling** — crowded and more serious. ChunkLab has 10 chunkers
and more tests than this project. enterprise-rag-bench has PII detection and cost
attribution. crag-bench runs on a public benchmark, so its results are comparable
across projects and these are not.

What survives comparison with all four:

```
1  a metric for exclusion compliance          none of the four have it
2  sequential stage locking with attribution  they compare whole pipelines, or
                                              offer per-stage playgrounds with
                                              no carry-forward
3  it ends with a running assistant           all four end at a report
4  findings, not just a harness               two of the four publish no numbers
```

Point 3 is the clearest product logic. Every comparable tool produces an
evaluation; this one produces the thing you were evaluating.

## Where it stops working

**Curation does not scale, and it is load-bearing.** The 100% compliance figure
rests on `rag/vocab/curated.py` — 478 hand-written lines mapping 1,776 raw
ingredient strings onto 158 canonical names. A corpus with 50,000 entities cannot
be curated by hand.

This is not dodged, because the reason it is hand-written is measurable: a model
that files besan as gluten, or treats coriander seeds and coriander leaves as one
ingredient, produces a vocabulary that is wrong in precisely the cases a cook
notices. Machine-proposed, human-ratified curation is the plausible path. The open
question is what accuracy the proposal step needs before ratification stops being
cheaper than doing it by hand. **That experiment has not been run.**

**One corpus.** Everything here is measured on cookbooks. The claim that the same
exclusion problem appears in contracts and policy documents is reasoning from the
mechanism, not evidence. A second corpus is the only thing that would earn the
word "generalises", and it is the highest-value unbuilt item.

**Eight files in the evaluation core import cooking by name.** Listed in
KNOWN_ISSUES.md. The lab is not yet domain-agnostic in the way the positioning
implies.

**Numeric constraints are unsupported.** The `constraint` query family scores
0.000 for every retriever because all 14 of its questions compare numbers and
nothing in the pipeline does. Deferred deliberately, with a sketch of the fix.

## What would change the verdict

```
a second corpus                     turns a mechanism argument into evidence
generated + ratified labels         removes the adoption barrier
numeric comparison in the graph     closes the 0.000 family
a scaling experiment on curation    answers the one question an interviewer
                                    will reach in five minutes
```