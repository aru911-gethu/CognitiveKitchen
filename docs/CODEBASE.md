# Codebase walkthrough

Every Python file: what it does, how it does it, where it sits, and why it was
built that way. Written to be read in order, but each section stands alone.

Format for each file:

    WHAT   one line
    WHERE  who calls it
    HOW    the actual code
    WHY    the reasoning, and what breaks without it

---

# Contents

    LAYER 1  Startup and configuration      config, models, jobs, api
    LAYER 2  Getting documents in           ingest, loaders
    LAYER 3  Foundations                    types, registry, corpus
    LAYER 4  The six stages                 chunking, embedding, retrieval,
                                            query, strategies, generate
    LAYER 5  The knowledge graph            vocab, graph
    LAYER 6  Measurement                    eval
    LAYER 7  Runtime                        pipeline, memory, telemetry
    LAYER 8  The interface                  ui

---

# LAYER 1 - Startup and configuration

## `config.py`

**WHAT** Reads `.env`, converts text into typed values, creates the folders the
app needs, and exposes one shared `settings` object.

**WHERE** Imported by almost every file as `from ..config import settings`.

**HOW**

```python
ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    api_port: int = 8010
    data_dir: Path = ROOT / "data"
    neo4j_uri: str | None = None
    langsmith_tracing: bool = False

    @property
    def api_base(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

settings = Settings()
settings.ensure_dirs()
```

**WHY, line by line**

`Path(__file__).resolve().parents[2]` finds the project root *from the code's own
location*, not from the current working directory. The alternative, `Path(".")`,
depends on where you were standing when you ran the command - launch from your
home folder and every data path is wrong.

    __file__            src/cognitive_kitchen/config.py
    .resolve()          absolute, symlinks followed
    .parents[0]         src/cognitive_kitchen
    .parents[1]         src
    .parents[2]         the project root

The annotation is the conversion. Everything in a `.env` file is text - there is
no such thing as a number in a `.env`. `api_port: int` is what turns the five
characters `8`,`0`,`1`,`0` into the integer 8010. Without it you get a string,
and the failure surfaces later inside uvicorn where the cause is invisible.

`bool` is the one people get wrong by hand:

```python
bool(os.environ.get("LANGSMITH_TRACING", "false"))   # always True!
```

Any non-empty string is truthy in Python, so the hand-written version can never
be False. Pydantic knows `"false"`, `"0"`, `"no"`, `"off"` all mean False, and
raises on anything it cannot interpret.

`extra="ignore"` tolerates variables in `.env` that this class does not define.
The alternative, `extra="forbid"`, refuses to start if `.env` has one stray line
- and `.env` is a file humans hand-edit and paste into.

`str | None = None` means genuinely optional. Distinguishing `None` from `""`
matters: `None` says never configured, `""` says configured as blank.

`@property` for `api_base` rather than a stored field, because it is built from
two other settings. Store it and it goes stale when the port changes. Derive it
and it cannot disagree with its inputs.

`settings = Settings()` at module level is the design decision with the widest
consequences. Python caches modules in `sys.modules`, so this runs exactly once
no matter how many files import it. Every importer gets the same object - which
is why the Lab can do `setattr(settings, "embedding_model", ...)` and change the
model for the whole process with no restart.

`ensure_dirs()` runs on import, so `data/uploads/` exists before anything tries
to write there:

```python
d.mkdir(parents=True, exist_ok=True)
```

    parents=True    create data/ if it is missing, not just data/uploads/
    exist_ok=True   do not raise on the second run

Together: idempotent, safe to call any number of times.

## `models.py`

**WHAT** Pydantic schemas describing ingestion output, so PDF and web produce one
identical shape.

**WHERE** `ingest/*` builds these; `api.py` returns them as JSON; `loaders/*`
reads them back.

**HOW**

```python
class SourceType(str, Enum):
    pdf = "pdf"
    url = "url"

class RawRecipe(BaseModel):
    recipe_id: str
    title: str
    source_type: SourceType
    pages: list[int] = Field(default_factory=list)
    url: str | None = None
    ingredient_lines: list[str] = Field(default_factory=list)
    detected_by: str = "headings"
```

**WHY**

`BaseModel` validates at construction. Bad data dies where it entered:

```python
RawRecipe(..., pages=["12", "13"])     # -> [12, 13]  converted
RawRecipe(..., pages="page twelve")    # -> ValidationError, right here
```

Without this the bad value travels and fails 200 lines away, where the error
message describes a symptom rather than the cause.

`class SourceType(str, Enum)` inherits from **both**. That means it behaves like
the plain string `"pdf"` in comparisons, f-strings and JSON, while Pydantic still
rejects anything that is not one of the two members. `source_type="excel"` fails
immediately instead of becoming a mystery three stages later.

`Field(default_factory=list)` rather than `= []` avoids Python's mutable-default
trap. A default is evaluated once, when the class is defined, so a literal `[]`
is **shared by every instance ever created**:

```python
a = RawRecipe(...); b = RawRecipe(...)
a.pages.append(12)
b.pages                # [12]   <- b got a's page
```

`default_factory` calls `list()` fresh per instance.

`detected_by` is worth noting as a design habit: it records *how* a recipe was
found - `json-ld`, `headings`, `prose-boundary`, `heuristic-text`. That is an
audit trail for parse quality, which lets you ask "are the bad recipes all from
one detection method" instead of guessing.

## `jobs.py`

**WHAT** Tracks a running ingestion and carries live progress frames from a
worker thread to the HTTP stream.

**WHERE** `api.py` creates a job then starts a thread; the SSE endpoint drains
the queue.

**HOW**

```python
SENTINEL = object()

@dataclass
class Job:
    job_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    history: list[ProgressEvent] = field(default_factory=list)
    loop: asyncio.AbstractEventLoop | None = None
    _seq: int = 0

    def emit(self, event: ProgressEvent) -> None:
        self._seq += 1
        event.seq = self._seq
        self.history.append(event)
        self.state = event.state
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
```

**WHY**

This file exists because of one hard constraint: **ingestion is blocking work and
FastAPI is async.** Reading a 222-page PDF takes minutes. Do it on the event loop
and the whole API freezes, including the endpoint that reports progress.

So ingestion runs on a worker thread. But a thread cannot touch an asyncio queue
directly - `queue.put_nowait` mutates state the event loop owns, and doing that
from another thread corrupts it in ways that show up as intermittent hangs.

`self.loop.call_soon_threadsafe(...)` is the correct bridge. It does not do the
work; it *schedules* the work onto the loop, which then runs it safely on its own
thread. This single line is the difference between a reliable stream and a
mystery deadlock.

`SENTINEL = object()` creates a value guaranteed unique - `object()` is never
equal to anything but itself. Pushed at the end of the stream so the reader knows
"finished" rather than waiting forever. A string like `"DONE"` would be ambiguous
because a legitimate frame could contain it.

`_seq` increments per frame. A UI that reconnects mid-run can tell which frames
it already has, so a dropped websocket does not double-count pages.

`history` keeps every frame, so a subscriber joining late can be replayed rather
than seeing a run already in progress with no context.

```python
try:
    job.loop = asyncio.get_running_loop()
except RuntimeError:
    job.loop = None
```

`get_running_loop()` raises `RuntimeError` when called outside async context -
for example from a test or a script. Catching it means the registry still works
with no loop at all, just without live streaming. The feature degrades instead of
the import failing.

## `api.py`

**WHAT** FastAPI app: upload a PDF, crawl URLs, stream progress, list datasets.

**WHERE** Started by `uv run ck-api`. The Streamlit UI talks to it over HTTP.

**HOW**

```python
@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)): ...

@app.get("/ingest/stream/{job_id}")      # Server-Sent Events
```

**WHY the UI and API are separate programs**

Streamlit re-runs its entire script on every interaction. If ingestion lived in
the UI process, a single click during a long read would restart the script and
either duplicate the work or lose it. Splitting them means the API owns the
long-running job and the UI is free to redraw as often as it likes.

Two guards live here, and both are deliberate:

    SSRF          non-HTTP schemes refused; hosts resolving to loopback,
                  private, link-local or reserved addresses rejected BEFORE
                  any fetch
    robots.txt    checked per host and honoured; disallowed URLs skipped
                  with a warning rather than fetched

The API has **no authentication**, which is why it binds to localhost. That is a
documented limitation rather than an oversight - it accepts file uploads and
fetches user-supplied URLs, so exposing it without auth would be a real
vulnerability.
---

# LAYER 2 - Getting documents in

## `ingest/pdf_ingest.py`

**WHAT** Reads a PDF page by page and pulls out recipes.

**WHY page by page** Not because it is faster - it is not - but because it lets a
frame be emitted per page. The alternative parses the whole book then reports, so
the user watches a spinner for two minutes with no idea whether it is working.

Each recipe records `detected_by`, so parse quality is inspectable afterwards.

## `ingest/web_ingest.py`

**WHAT** Playwright crawler. Given recipe pages, extracts them. Given a *category*
page, finds the recipe links and follows them.

**WHY it is the largest ingest file (333 lines)** The hard problem is not reading
a recipe page, it is deciding which links on an index page are recipes rather than
navigation, pagination, tags and adverts. Links are scored and ranked so a limited
page budget goes to real recipes.

```python
# A recipe page yields a recipe and stops; an index page yields none
# and is expanded. No toggle needed - the page itself decides.
```

That comment records a design choice worth copying: rather than asking the user
"is this a category page?", try to extract a recipe and use the *result* to decide.
The data answers the question.

## `loaders/json_loader.py` and `loaders/pdf_loader.py`

**WHAT** Turn stored ingestion output into LangChain `Document` objects.

**HOW**

```python
Document(page_content=rendered_recipe_text,
         metadata={"recipe_id": r["recipe_id"], "title": r["title"],
                   "pages": r["pages"]})
```

**WHY LangChain Documents** Not because LangChain is needed - most of the pipeline
is hand-written - but because `Document` is the shape its text splitters expect,
and `recursive.py` uses `RecursiveCharacterTextSplitter`. Using the standard
container means one less adapter.

`load_latest_ingested(source_type="pdf")` picks the newest run of a given type,
which is what makes the Lab's corpus dropdown work.

---

# LAYER 3 - Foundations

## `rag/types.py`

**WHAT** The data contracts every stage passes to the next.

**HOW**

```python
@dataclass(frozen=True)
class RecipeSpan:
    recipe_id: str
    title: str
    start: int          # inclusive
    end: int            # exclusive

@dataclass(frozen=True)
class Passage:
    passage_id: str
    text: str
    start: int
    end: int
    meta: dict[str, Any] = field(default_factory=dict)
```

**WHY the offsets are the whole design**

`start` and `end` are character positions in `Corpus.text`. Recipes are recorded
in the same coordinate space. Therefore "did this chunk cut a recipe in half" is
arithmetic:

```python
def overlap(a_start, a_end, b_start, b_end) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))
```

`min(ends) - max(starts)` is the width of the intersection. It goes negative when
the ranges do not touch, so `max(0, ...)` floors it - "shares -400 characters" is
nonsense.

Run that against all 184 spans and you get the number of recipes a chunk touched.
One means clean. Two means it welded two dishes together. **No AI, no judge, no
human labelling - subtraction.**

`frozen=True` makes these immutable. Assigning `span.start = 5` raises. This
matters because spans are the ruler every Stage 2 metric is measured against - if
a chunker could edit them, a metric could be made to agree with itself.

`spans: tuple[...]` not `list`, because a frozen dataclass holding a list is only
shallowly immutable - you could still `.append()` to it.

`meta: dict[str, Any]` is the extension point, and it is load-bearing:

```python
meta["recipe_ids"]   # which recipes this chunk touches
meta["source"]       # "pdf-6a2d0fb32710"
meta["strategy"]     # "graph_only" when the graph faked a passage
```

`meta["source"]` exists because of a real bug. Recipe ids restart at `r_0001` in
every ingestion run, so a PDF's `r_0002` and a website's `r_0002` are different
recipes sharing an id. Without the source prefix, a glossary page inherited a web
recipe's constraint eligibility and took the top three slots. Compliance measured
68% instead of 100%.

```python
def merge_intervals(intervals): ...
```

Interval union, and the docstring says why: *"so coverage cannot exceed 1.0 by
double counting."* If you measure how much of a recipe your chunks cover by
summing overlaps, two overlapping chunks count the shared region twice and you get
coverage above 100%. Merge first and that is impossible.

This is the same bug class as `diversity_at_k` exceeding 1.0, which happened
because a chunk spanning two recipes was counted once per recipe. Both were caught
by a metric producing an impossible number - which is an argument for metrics with
known bounds.

`Answer` is the only non-frozen type, because strategies build it up in steps:

```python
@dataclass
class Answer:
    question: str
    text: str
    contexts: list[Passage] = field(default_factory=list)
    refused: bool = False
```

`refused` is a field rather than a check for `text == "NOT_IN_CONTEXT"`, because
refusal is a first-class outcome. `honest_abstention` needs to tell "correctly
refused an unanswerable question" from "failed on an answerable one". An earlier
version used `bool(contexts)` as a proxy for answerability and **punished correct
refusals** - the fix needed refusal to be explicit.

## `rag/registry.py`

**WHAT** A plugin registry. Strategies declare themselves; nothing keeps a list.

**HOW**

```python
_REG: dict[str, dict[str, Callable]] = defaultdict(dict)
_DISCOVERED: set[str] = set()

def register(kind: str, name: str):
    def deco(factory):
        _REG[kind][name] = factory
        return factory
    return deco

def discover(*packages: str) -> None:
    for pkg in packages:
        if pkg in _DISCOVERED:
            continue
        module = importlib.import_module(pkg)
        for _, modname, _ in pkgutil.iter_modules(module.__path__):
            if modname.startswith("_"):
                continue
            importlib.import_module(f"{pkg}.{modname}")
        _DISCOVERED.add(pkg)
```

**WHY, and how decorators actually work here**

`register` is a **decorator factory** - a function returning a decorator. Three
layers:

```python
@register("chunker", "recipe")          # 1. call register(...) -> returns deco
def make_recipe(max_chars: int = 0):    # 2. deco(make_recipe) runs
    return RecipeChunker(max_chars)     # 3. the name now maps to this function
```

The decorator stores the function in `_REG` and returns it unchanged, so
`make_recipe` still works if called directly. Registration is a side effect of
*defining* it.

`defaultdict(dict)` means `_REG["chunker"]` auto-creates an empty dict on first
access. Without it every register call would need `if kind not in _REG`.

`discover` is the piece that makes it work. A decorator only runs when its module
is **imported**, and nothing imports `rag/chunking/recursive.py` explicitly. So
`discover` walks the package with `pkgutil.iter_modules` and imports every module,
which fires every decorator.

    if modname.startswith("_")    skip _base.py and _semantic.py
                                  they hold shared helpers, not strategies
    _DISCOVERED set               import once per process, not per call

```python
def signature(kind: str, name: str) -> dict[str, Any]:
    sig = inspect.signature(_REG[kind][name])
    return {p.name: p.default for p in sig.parameters.values()
            if p.default is not inspect.Parameter.empty}
```

This is introspection, and it fixed a real bug. The Lab needs to render controls
for a strategy, which means knowing what parameters it accepts. The earlier
version guessed:

```python
try:
    return build("chunker", name, size=size, overlap=overlap)
except TypeError:
    return build("chunker", name)        # silently used defaults
```

The factories take `chunk_size`, not `size`. So the guess **always** failed, the
fallback **always** ran, and the size and overlap controls in Stage 2 did nothing
at all - silently. Every early Stage 2 run used 800 rather than the 600 shown on
screen.

Asking the registry what a factory accepts is both correct and self-maintaining as
strategies are added.

`inspect.Parameter.empty` is the sentinel meaning "no default given", which is why
the comprehension filters on `is not`. Using `!= None` would be wrong, since
`None` is a legitimate default.

## `rag/corpus.py`

**WHAT** Glues documents into one string and records where each recipe sits.

**HOW**

```python
SEPARATOR = "\n\n"

def build_corpus(docs: list[Document], source: str = "") -> Corpus:
    chunks, spans, cursor = [], [], 0
    for doc in docs:
        text = doc.page_content
        if not text.strip():
            continue
        spans.append(RecipeSpan(
            recipe_id=doc.metadata.get("recipe_id", f"d_{len(spans):04d}"),
            title=doc.metadata.get("title", "")[:180],
            start=cursor,
            end=cursor + len(text),
            pages=list(doc.metadata.get("pages") or []),
        ))
        chunks.append(text)
        cursor += len(text) + len(SEPARATOR)
    return Corpus(text=SEPARATOR.join(chunks), spans=tuple(spans), source=source)
```

**WHY the cursor arithmetic is the important part**

`cursor` tracks the position each document *will* occupy once everything is
joined. The critical line:

```python
cursor += len(text) + len(SEPARATOR)
```

It adds the separator's length too. Miss that and every span after the first is
offset by two characters per document - by recipe 184 the bookmarks point 366
characters wrong, and every purity number is quietly garbage.

This is why the text is built and the offsets recorded **in one pass**. Doing it
in two - join first, then search for each recipe - would mean finding text by
searching, which breaks the moment two recipes share an opening line.

Verified on the real corpus:

    corpus.text        148,963 characters, one string
    corpus.spans       184 entries
    spans[17]          r_0018 'BOILED RICE' start=14462 end=14611

```python
corpus.text[14462:14611]
# 'BOILED RICE\nIngredients:\n- Rice 500 gm\n- Salt to taste\nMethod:\n...'
```

The bookmark is exact.

Other details:

`if not text.strip(): continue` skips empty documents, otherwise you get a span of
zero length that later divides by zero in a mean-length calculation.

`title=...[:180]` truncates, because a mis-parsed title can be an entire page and
it is only used for display.

`f"d_{len(spans):04d}"` is the fallback id when metadata has none. `:04d` means
zero-padded to 4 digits, so ids sort lexically the same way they sort numerically -
`d_0002` before `d_0010`, which plain `d_2`/`d_10` would not.

```python
def summarise(corpus: Corpus) -> dict:
    lengths = [s.length for s in corpus.spans] or [0]
```

`or [0]` guards an empty corpus. Without it `sum(lengths)/len(lengths)` raises
`ZeroDivisionError` on a corpus with no recipes - which is exactly the state you
are in before ingesting anything, so the Lab header would crash on first load.

---

# LAYER 4 - The six stages

## `rag/chunking/` - eight ways to cut the text

### `_base.py` - the shared helper

```python
def make_passages(corpus, bounds, name, params) -> list[Passage]:
    out = []
    for index, (start, end) in enumerate(bounds):
        touched = [s.recipe_id for s in corpus.spans
                   if overlap(start, end, s.start, s.end) > 0]
        out.append(Passage(
            passage_id=f"{name}-{index:05d}",
            text=corpus.text[start:end],
            start=start, end=end,
            meta={"strategy": name, "params": params,
                  "recipe_ids": touched, "source": corpus.source}))
    return out
```

**WHY it exists** Every chunker produces character ranges. Turning ranges into
`Passage` objects - slicing the text, computing which recipes were touched,
stamping the metadata - is identical work. Doing it once means `meta["recipe_ids"]`
is computed the same way for all eight, so the purity metric cannot be gamed by a
chunker that reports its own overlap.

### `naive.py` - the control

```python
class NaiveChunker:
    def split(self, corpus):
        size, overlap_n = self.params["chunk_size"], self.params["overlap"]
        bounds, cursor = [], 0
        while cursor < len(corpus.text):
            bounds.append((cursor, min(cursor + size, len(corpus.text))))
            cursor += size - overlap_n
        return make_passages(corpus, bounds, self.name, self.params)
```

**WHY** It exists to be beaten. Cuts every N characters with no regard for words,
sentences or recipes. Any cleverer chunker has to prove it is worth its cost
against this. Without a control, "semantic chunking scored 0.83" is a number with
no meaning.

### `recursive.py` - LangChain, mapped back to offsets

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def locate(corpus_text, piece, from_index):
    found = corpus_text.find(piece, from_index)
    return found
```

**WHY the `locate` step** LangChain's splitter returns *strings*, not offsets. But
every metric here needs offsets. So each returned piece is re-located in the
original text, searching forward from where the last one ended. Searching forward
rather than from zero matters - a duplicated line elsewhere in the book would
otherwise match the wrong position and corrupt the span.

`SEPARATORS` in that order means: try paragraph breaks, then lines, then sentence
ends, then spaces, then give up and cut mid-word. It only escalates when a piece is
still over the limit, so it respects the writer's structure where it can.

### `_semantic.py` - the shared embedding-based logic

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .recursive import SEPARATORS, locate
```

**WHY it pre-splits first** The docstring: cutting on raw character count first
means units never start mid-word, and the semantic pass then decides which pieces
belong together. Embedding a half-word produces a meaningless vector.

Four chunkers share this: `semantic_adjacent`, `semantic_centroid`, and the two
`recursive_semantic_*` variants. The difference is only how they decide to cut:

    adjacent    compare each unit to the NEXT one, cut where similarity dips
    centroid    keep a running average of the group, cut when a unit drifts

**Why both lose to `recipe`** A recipe's ingredient list and its method read as
different topics. Similarity dips *inside* a dish, so these split recipes in half
at exactly the wrong place.

### `recipe_aware.py` - the winner, and it does no inference

```python
class RecipeChunker:
    def __init__(self, max_chars: int = 0) -> None:
        # 0 means never split. A long recipe stays whole, because a recipe is
        # the unit a cook asks for -- half of one is not a smaller answer, it
        # is a wrong one.
        self.params = {"max_chars": max_chars}

    def split(self, corpus: Corpus) -> list[Passage]:
        limit = self.params["max_chars"]
        bounds = []
        for span in corpus.spans:
            if not limit or span.length <= limit:
                bounds.append((span.start, span.end))
                continue
            cursor = span.start
            while cursor < span.end:
                stop = min(cursor + limit, span.end)
                if stop < span.end:
                    newline = corpus.text.rfind("\n", cursor, stop)
                    if newline > cursor:
                        stop = newline
                bounds.append((cursor, stop))
                cursor = stop
        return make_passages(corpus, bounds, self.name, self.params)
```

**WHY it wins** The other six *infer* where one thing ends. Ingestion already
recorded it. `corpus.spans` is the answer, and re-deriving it by character count
throws information away and then spends compute approximating it.

`max_chars=0` means never split, and that default is a product decision, not a
technical one: a recipe is the unit a cook asks for.

`rfind("\n", cursor, stop)` searches **backwards** for a newline within the window,
so an oversized recipe is cut at a line break rather than mid-sentence. `rfind`
not `find`, because you want the last newline before the limit, giving the largest
legal piece.

The measured difference, on the same text:

    recipe      184 passages, chunk 17 -> recipe_ids ['r_0018'], 1 recipe
    recursive   275 passages, chunk 17 -> recipe_ids ['r_0017','r_0018']

That second one starts inside Lemon Pickle and ends inside Boiled Rice. One chunk,
two unrelated dishes. That is the mechanism behind the answer that returned butter
quantities from five recipes.

### `RecipeSectionChunker` - the interesting trade

```python
SECTION = re.compile(
    r"^(?:ingredients?|method|directions?|steps?|preparation)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE)
```

```python
if title and not text.lower().startswith(title.lower()[:24]):
    text = f"{title}\n{text}"
```

**WHY the title is repeated into every piece** This is the point of the chunker. A
bare ingredient list is useless without knowing which dish it belongs to - the
exact failure it exists to prevent. Prefixing costs a little duplication and buys
two things: every chunk can answer on its own, and the title's words become
searchable from the method section, so "how do I make lemon rice" can match the
steps rather than only the heading.

The regex anchors matter:

    ^ ... $        with MULTILINE, matches a whole LINE
                   so "Ingredients:" as a heading matches,
                   but "ingredients:" inside a sentence does not
    \s*:?\s*       optional colon and surrounding whitespace
    IGNORECASE     the PDF is inconsistent about capitalisation

```python
if not cuts or span.length < self.params["min_chars"] * 2:
    pieces = [(0, len(body))]
```

A short recipe is left whole. Splitting a 200-character recipe produces fragments
too small to answer anything - so the guard is `min_chars * 2`, meaning "only split
if both halves could plausibly stand alone".

```python
edges = sorted({0, *cuts, len(body)})
pieces = [(a, b) for a, b in zip(edges, edges[1:]) if b - a > 0]
```

`{0, *cuts, len(body)}` is a **set**, which deduplicates - if a heading happens to
be at position 0, you do not get a zero-width piece. `zip(edges, edges[1:])` is the
standard pairwise idiom: `[0,50,120]` becomes `[(0,50),(50,120)]`.

## `rag/embedding/sentence_transformer.py`

**WHAT** Turns text into vectors, cached to disk.

**HOW**

```python
class TextEmbedder:
    def __init__(self, model_name=None, batch_size=16, cache=True):
        self.model_name = model_name or settings.embedding_model

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _cache_file(self, texts: list[str]) -> Path:
        digest = hashlib.sha256(
            (self.model_name + "\x00" + "\x00".join(texts)).encode("utf-8")
        ).hexdigest()
        return settings.data_dir / "eval" / f"emb-{digest[:16]}.npy"
```

**WHY the model name is in the cache key** Without it, switching from MiniLM to
Qwen would serve MiniLM's vectors from cache and label the results as Qwen. The
numbers would be wrong and nothing would complain. Including the model name makes
that impossible.

`"\x00".join(texts)` uses a null byte as separator rather than a comma, because a
null byte cannot appear in the text. With a comma, the two lists `["a,b"]` and
`["a","b"]` would hash identically.

`model_name or settings.embedding_model` is the fallback pattern used throughout -
explicit argument wins, otherwise config. That is what makes the Lab's model
selector work without threading a parameter through twenty call sites.

**Lazy loading.** The docstring says it: the model loads on first `encode`, not on
import, *so naive and recursive chunkers stay fast*. Those two need no vectors at
all - loading a 500 MB model to run them would be pure waste.

## `rag/retrieval/` - nine ways to find chunks

### `_base.py` - `MultiQueryMixin`

```python
class MultiQueryMixin:
    def search_multi(self, queries: list[str], k: int) -> list[Scored]:
        seen, merged = set(), []
        for q in queries:
            for hit in self.search(q, k):
                if hit.passage.passage_id not in seen:
                    seen.add(hit.passage.passage_id)
                    merged.append(hit)
        return merged[:k]
```

**WHY a mixin** `decompose` and `hyde` turn one question into several. Every
retriever needs to handle a list, and the merge logic is identical. A mixin adds
the method to any class that inherits it, without a base class dictating
construction.

`seen` deduplicates, because the same chunk will legitimately match several
sub-questions and returning it twice wastes a slot in the top-k.

### `dense.py` - FAISS cosine similarity

```python
import faiss

def index(self, passages):
    vectors = self.embedder.encode([p.text for p in passages])
    self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    self._index = faiss.IndexFlatIP(self._vectors.shape[1])
    self._index.add(self._vectors)
```

**WHY inner product and not cosine** The docstring: *"Vectors are L2-normalised by
the embedder, so inner product is cosine."* Once vectors are unit length, the dot
product **is** the cosine. `IndexFlatIP` is cheaper than a cosine index because it
skips the normalisation it does not need to repeat.

`np.ascontiguousarray(..., dtype=np.float32)` is required by FAISS - it is C++ and
needs a contiguous float32 block. Pass a non-contiguous view or float64 and it
either errors or silently misreads memory.

`IndexFlatIP` is brute force, comparing against every vector. For a few hundred
chunks that is microseconds and exactly right; approximate indexes like IVF only
pay off at hundreds of thousands.

### `rrf.py` - fusion on rank, not score

```python
def search(self, query: str, k: int) -> list[Scored]:
    pool = max(self.params["pool"], k)
    c = self.params["c"]
    fused: dict[str, float] = {}
    for retriever in (self.dense, self.sparse):
        for hit in retriever.search(query, pool):
            fused[hit.passage.passage_id] = (
                fused.get(hit.passage.passage_id, 0.0) + 1.0 / (c + hit.rank))
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [Scored(passage=self._by_id[pid], score=score, rank=i)
            for i, (pid, score) in enumerate(ranked, start=1)]
```

**WHY this is better than weighted hybrid, and it is the key insight**

    RRF(d) = sum over retrievers of  1 / (c + rank(d))

It uses **only rank position, never the score.** That is the whole appeal.

BM25 scores are unbounded positive numbers whose scale depends on corpus
statistics. Cosine similarity is bounded -1 to 1. Adding them with weights means
picking an alpha that makes them comparable - and that alpha is corpus-specific,
undiscoverable without tuning, and silently wrong when the corpus changes.

Rank has no units. Position 1 from BM25 and position 1 from dense mean the same
thing. **No calibration required.**

`c = 60` dampens the top. Without it, rank 1 scores 1.0 and rank 2 scores 0.5 - a
single retriever's first pick dominates. With c=60, rank 1 gives 1/61 and rank 2
gives 1/62, nearly equal, so *agreement between retrievers* matters more than one
retriever's confidence. 60 is the value from the original paper.

`key=lambda kv: -kv[1]` sorts descending by negating, avoiding `reverse=True`.

`self._by_id` exists because the fusion works on ids, not objects, so the passage
has to be recovered at the end. Built once during `index`.

### `graph_hybrid.py` - the gate that fixes the nut problem

```python
def search(self, query: str, k: int) -> list[Scored]:
    try:
        constraints = self.extractor().extract(query)
    except Exception:
        constraints = None

    if constraints is None or not constraints.restricting:
        self.trace.append({"query": query, "filtered": False,
                           "reason": "no restriction in question"})
        return self.base.search(query, k)

    allowed = self._allowed_recipe_ids(constraints)
    if not allowed:
        return []

    pool = max(self.params["pool"], len(self._passages))
    candidates = self.base.search(query, pool)
    kept = [hit for hit in candidates
            if allowed & set(hit.passage.meta.get("recipe_ids") or [])]
    return [Scored(passage=hit.passage, score=hit.score, rank=i)
            for i, hit in enumerate(kept[:k], start=1)]
```

**WHY rank everything then filter** This is the most important line in the file:

```python
pool = max(self.params["pool"], len(self._passages))
```

It asks the base retriever for **every passage in the corpus**, then intersects.
The tempting alternative - take the top 50 and filter those - is the same mistake
in a different place: a correct answer ranked 200th by similarity never reaches the
filter, so the graph gets blamed for a miss it did not cause. The corpus is a few
hundred chunks, so asking for all of them costs almost nothing and makes this a
genuine pre-filter rather than a post-filter wearing the name.

**Why it delegates when there is no restriction** "How do I make dosa" has nothing
to gate on. Delegating costs zero extra and keeps the Stage 3 comparison fair
rather than flattering - if it always paid for an LLM call, its latency column
would be misleading.

`allowed & set(...)` is set intersection. Truthy when non-empty, so it doubles as
the membership test.

```python
def _allowed_recipe_ids(self, constraints) -> set[str]:
    keys = traverse.filter_recipes(...)
    if self._source_prefix:
        keys = [k for k in keys
                if k.split(":", 1)[0].startswith(self._source_prefix)]
    return {k.split(":", 1)[-1] for k in keys}
```

**Filter to this corpus first, then drop the prefix.** The docstring explains the
bug this fixes: ids restart per run, so stripping the prefix first merges the PDF's
`r_0002` with the website's `r_0002`. `split(":", 1)` with maxsplit=1 matters
because a source name could itself contain a colon.

### `graph_only.py` - making the graph comparable

```python
def search(self, query: str, k: int) -> list[Scored]:
    constraints = self.extractor().extract(query)
    keys = traverse.filter_recipes(...)
    cards = traverse.recipe_cards(keys)
    chosen = self._ordered(keys, cards, constraints)[:k]
    out = []
    for rank, key in enumerate(chosen, start=1):
        card = cards.get(key) or {}
        passage = Passage(
            passage_id=f"graph-{key}",
            text=_render(card),
            start=0, end=len(_render(card)),
            meta={"strategy": "graph_only",
                  "recipe_ids": [card.get("recipe_id")],
                  "source": key.split(":", 1)[0]})
        out.append(Scored(passage=passage, score=1.0 / rank, rank=rank))
    return out
```

**WHY this file exists at all** Stage 5 scores the graph as a *set* - precision and
recall. But it never generates an answer, so faithfulness, relevancy and
cookability have nothing to read. The graph and retrieval lived in incompatible
scoring universes, and "should I use the graph or retrieval?" had no answer.

This adapter makes the graph impersonate a retriever. It emits `Passage` with the
same `meta` keys, so `stage6_generation.evaluate` runs unchanged and the graph
lands in the same table as `recursive -> rrf -> stuff_strict`.

`score=1.0 / rank` gives 1.0, 0.5, 0.33 - monotonically decreasing so anything
expecting sorted scores works, while being honest that it is positional, not a
similarity.

```python
def index(self, passages: list[Passage]) -> None:
    sources = {p.meta.get("source") or "" for p in passages or []}
    if len(sources) == 1:
        self._source = sources.pop() or self._source
```

`index` is a **no-op that only sniffs the corpus source.** It takes no chunks and
builds nothing. It exists purely so evaluators can treat this like any other
retriever - duck typing, honoured deliberately.

```python
def key_for(key: str) -> tuple:
    ingredients = set(cards.get(key, {}).get("ingredients") or ())
    overlap = len(ingredients & wanted)
    return (-overlap, len(ingredients), key)
```

**Tuple sorting** compares element by element. So this sorts by most overlap with
what was asked for (negated, to get descending), then fewest total ingredients,
then key for stability. The docstring is explicit that this is *a heuristic, not a
relevance score* - and that the absence of ranking is what separates this route
from retrieval.

### `mmr.py`, `hybrid.py`, `cross_encoder.py`, `sparse_bm25.py`, `sparse_tfidf.py`

    mmr             re-selects for diversity: pick the highest-scoring item that
                    is least similar to what you already picked
    hybrid          weighted blend of dense + sparse; needs the alpha that RRF
                    avoids needing
    cross_encoder   re-scores the top N by feeding query and chunk through a
                    model TOGETHER, rather than comparing separate vectors.
                    Much more accurate, ~25s on CPU
    sparse_bm25     keyword scoring with term saturation
    sparse_tfidf    older keyword scoring, kept as a comparison point

`mmr.py` carries a fixed bug worth naming: it was hardwired to `dense` as its base,
so choosing "rrf + mmr" in the Lab silently measured "dense + mmr". It now takes a
`base` parameter.

## `rag/query/` - rewriting the question

    passthrough.py   returns [query]. the control.
    decompose.py     splits a compound question into parts
    hyde.py          generates a fake ideal answer, searches with THAT

**WHY `hyde` is kept despite losing** ~400 seconds per query on CPU for no
measurable gain. That is a negative result worth keeping, because the technique is
widely recommended and the cost is invisible until you are the one without a GPU.
Deleting it would delete the evidence.

## `rag/strategies/generation.py` - four ways to write the answer

    stuff_strict    all chunks in, strict "only use this context" instruction
    reordered       same chunks, strongest evidence at BOTH ENDS of the prompt
    structured      asks for a structured answer
    map_reduce      summarise each chunk, then combine

**WHY `reordered` wins on identical chunks** Models attend to the start and end of
a context window more than the middle - the lost-in-the-middle effect. Putting the
strongest evidence at both ends rather than in retrieval order recovers accuracy
that was being thrown away. Same chunks, same model, +4 points faithfulness, free.

**Why this matters methodologically:** no improvement to retrieval would have found
it, because the retrieval was identical. Measured end to end, you would conclude
one pipeline was mysteriously better and never know which knob did it.

## `rag/generate/`

### `qwen_chat.py` - local, CPU, free

```python
def _ensure(self):
    if self._model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        # no device_map: it pulls in accelerate and this is CPU-only anyway
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float32)
        self._model.to("cpu")
        self._model.eval()
```

`.eval()` switches off dropout and batch-norm updating. Skip it and you get
non-deterministic output from a model you asked for temperature 0.

`float32` not `float16` because CPUs have no fast half-precision path - fp16 on CPU
is slower, not faster.

The comment about `device_map` is a real dependency decision: it would pull in
`accelerate` for functionality that is meaningless without a GPU.

### `openai_chat.py` - the judge and the constraint extractor

```python
self.ledger.record(span, elapsed, tokens_in=n_in, tokens_out=n_out,
                   model=self.model_name)
```

Every call is recorded, which is how the Stage 3 table can honestly show what the
graph route costs against retrievers that are free.

---

# LAYER 5 - The knowledge graph

## `rag/vocab/normalise.py` - strip a line down to an ingredient

```python
def normalise(line: str) -> str | None:
    s = strip_prep(strip_quantities(line))
```

**HOW** Two passes. `strip_quantities` removes numbers, fractions and units -
`"2 tbsp"`, `"1/2 cup"`, `"500 gm"`. `strip_prep` removes preparation words -
`"finely chopped"`, `"to taste"`, `"as required"`.

    "2 tbsp finely chopped onion"   ->   "onion"
    "Rice 500 gm"                   ->   "rice"
    "Salt to taste"                 ->   "salt"

**WHY rules and not a model** These transformations are mechanical and
deterministic. A model would be slower, cost money, and occasionally hallucinate an
ingredient that was never written. Rules are auditable - when one is wrong you can
see exactly which pattern fired.

## `rag/vocab/curated.py` - 478 hand-written lines, the only non-generated input

**WHAT** Maps 1,776 raw ingredient strings onto 158 canonical names.

**HOW** Three tiers, described in its own docstring:

```python
# Anything absent from all three keeps its Tier 1 normalised form as its own
# canonical name. That is the safe default: it may be redundant, never wrong.
```

**WHY hand-written, and this is the load-bearing decision in the project**

A model that files besan as gluten, or treats coriander seeds and coriander leaves
as one ingredient, produces a vocabulary that is wrong in precisely the cases a
cook notices. And wrong **invisibly** - the graph will confidently report
compliance against a wrong category, and no metric here can see it.

Two attempts are recorded in the file as comments:

    cloves guard    REVERTED. protecting cloves as a spice fragmented clean
                    `garlic` into six variants, because the corpus uses cloves
                    as a UNIT of garlic
    gram guard      KEPT. gram flour is not wheat flour

**The honest limit:** this does not scale. 158 ingredients is demonstrated; 50,000
is not. `KNOWN_ISSUES.md` names the experiment that would resolve it.

## `rag/vocab/ingredients.py` - the lookup

```python
@lru_cache(maxsize=1)
def load_map() -> dict[str, dict[str, Any]]: ...

def resolve(raw: str) -> list[str]: ...

def canonical(raw: str) -> str | None:
    names = resolve(raw)
    return names[0] if names else None
```

`@lru_cache(maxsize=1)` reads the map file once per process. `maxsize=1` because
there is only ever one map - it is being used as a memoiser for a zero-argument
function, which is the idiomatic way to do lazy module-level loading.

`resolve` returns a **list**, because one raw line can legitimately name several
ingredients - "salt and pepper" is two. `canonical` is the convenience wrapper for
when you want one answer.

**The subtlety that caused a bug in `pantry.py`:** because unknown terms become
their own canonical name, `canonical()` almost never returns `None`. So testing
"did the vocabulary recognise this?" with `if canonical(x)` is always true. The
correct test is whether the resolved name exists as a node in the graph.

## `rag/graph/client.py` - the driver

```python
def run(cypher: str, **params) -> list[dict]:
    with _driver().session(database=settings.neo4j_database) as s:
        return [record.data() for record in s.run(cypher, **params)]
```

**WHY the list comprehension** Neo4j records are only valid while the session is
open. Returning the cursor would give you an object that fails the moment the
`with` block exits. Materialising into plain dicts inside the block means callers
get data they can keep.

`**params` passes named parameters to Cypher rather than formatting them into the
string - the same injection defence as SQL parameterisation.

## `rag/graph/schema.py` - constraints and indexes

**WHAT** Creates uniqueness constraints and indexes before any data is loaded.

**WHY first** A uniqueness constraint on `Recipe.key` makes a duplicate insert fail
loudly rather than creating a second node. Adding it after loading means the load
has to be correct by luck. Indexes before loading also make the load faster,
because the `MERGE` statements can look up existing nodes.

## `rag/graph/build.py` - the `ck-graph` command

```python
key = f"{source}:{recipe_id}"      # "pdf-6a2d0fb32710:r_0018"
```

**WHY namespaced keys** Recipe ids restart at `r_0001` in every ingestion run. Two
runs means two `r_0002` values for different recipes. The source prefix makes the
key globally unique. This is the same collision that `graph_hybrid` guards against
at its gate - fixed in both places because the id space is genuinely ambiguous.

**WHY it wipes and rebuilds** Repeatability. An incremental build accumulates state
you cannot reason about; a wipe means the graph is always exactly a function of
`data/ingested/`. `--dry-run` shapes and reports without touching the database, so
you can check what a rebuild *would* do.

Measured output:

    Recipe 195 (177 real)   Ingredient 156   Category 6   Step 1358
    USES 1658   NEXT 1164   PERFORMS 869   NEEDS 242

`is_recipe = true` distinguishes the 177 real recipes from 18 fragments the parser
produced - glossary pages, headings. Every traversal filters on it.

## `rag/graph/traverse.py` - the three things search cannot do

### `filter_recipes` - membership

```python
def filter_recipes(include=(), exclude=(), exclude_categories=(), course=""):
    keys: set[str] | None = None

    if course:
        keys = set(of_course(course))

    wanted = _names(include)
    if include and not wanted:
        # The question named an ingredient and the vocabulary could not resolve
        # it. Falling through to "everything" would answer "what can I cook with
        # freshly?" with all 177 recipes, which reads as confidence rather than
        # failure. An empty set is the honest reply.
        return []
    if wanted:
        found = set(using_any(wanted))
        keys = found if keys is None else (keys & found)
    ...
    return sorted(keys) if keys is not None else all_recipes()
```

**WHY `keys: set | None` rather than starting with all recipes**

`None` means "no condition has been applied yet". That is different from "a
condition was applied and matched nothing". Start with the full set and you cannot
tell those apart - an empty result would be indistinguishable from no filtering.

The `include and not wanted` guard is a fixed bug. Asked "what can I cook with
freshly?" the vocabulary cannot resolve "freshly", `wanted` is empty, and the old
code fell through to returning **every recipe** - 168 results presented as an
answer. Returning `[]` is honest failure.

`course` is in here because of a measured precision failure, recorded in
`course.py`:

```python
# Stage 5 needs this. Asked "a bread dish with no dairy", the graph answered the
# dairy half and ignored the bread half, so it returned 86 recipes where 1 was
# wanted -- precision 0.01. Absence it models; course it did not.
```

### `pantry_gap` - what you are missing

```python
rows = client.run(f"""
    MATCH (r:Recipe)-[:USES]->(i:Ingredient)
    WHERE {REAL}
    WITH r, collect(DISTINCT i.name) AS needed
    WITH r, needed,
         [n IN needed WHERE NOT n IN $owned] AS missing
    WHERE size(missing) <= $max_missing
    RETURN r.key AS key, r.title AS title, needed, missing,
           size(missing) AS n_missing, size(needed) AS n_needed
    ORDER BY n_missing ASC, n_needed DESC, key
    LIMIT $limit""", owned=owned, max_missing=max_missing, limit=limit)
```

**WHY this is the flagship example** The word "missing" appears in no document.
There is no passage to be similar to. The answer is set subtraction over the graph:

    missing = [n IN needed WHERE NOT n IN $owned]

That is a Cypher list comprehension - same shape as Python's. No search can produce
this, at any quality.

`ORDER BY n_missing ASC, n_needed DESC` reads as: fewest missing first, then prefer
*more complex* recipes among ties. Counterintuitive, but if you are missing zero
things from both, the more elaborate dish is the more interesting suggestion.

### `substitutes` - the cleverest query in the project

```python
#     context(x)  = every other ingredient sharing a recipe with x
#     similarity  = jaccard(context(a), context(b))
#     penalty     = how often a and b appear together, which argues against
#                   substitution, since a recipe using both treats them as
#                   different things
#
#     score = similarity * (1 - together / recipes(a))
```

**WHY plain co-occurrence is the wrong measure, stated in the docstring**

> the ingredients most often found next to ghee are water, salt and turmeric,
> because those are in almost every recipe. Appearing WITH something is the
> opposite of substituting FOR it.

So it uses the distributional idea: two ingredients are alike if they **keep the
same company**, even when they never meet.

The penalty term is the insight. If a recipe uses both ghee and butter, that recipe
is treating them as *different things*. So frequent co-occurrence argues against
substitutability, and it is multiplied in as `(1 - together/total)`.

```cypher
WHERE shared_ctx >= $min_shared
```

Requires at least 3 shared context ingredients before scoring, so two obscure
ingredients that each appear once do not get a spurious jaccard of 1.0.

```cypher
ORDER BY same_category DESC, score DESC
```

Same-category candidates first, so a dairy swap stays inside dairy even when the
text never suggested it.

## `rag/graph/constraints.py` - question to graph conditions

```python
PROMPT = """Read the cooking question and return only JSON, no prose.
{{"include": [...], "exclude": [...], "exclude_categories": [...],
 "substitute_for": "...", "course": "...", "dish": "..." }}
...
```

**WHY an LLM and not a keyword table**, from the docstring:

> "no lexicon matching" was the whole point: "I'm avoiding dairy", "nothing with
> milk in it" and "lactose free" are the same condition and share no words, so any
> pattern table would need endless upkeep.

**The prompt is the code here.** Several rules in it are fixed bugs:

```
"no meat", "avoiding meat"   -> ["meat","fish"]
```
Someone avoiding meat does not want prawns. This is the same error that was found
in the golden dataset - the model was making it too.

```
Asking for a REPLACEMENT is not the same as ruling something out.
"What can I use instead of ghee?" -> substitute_for, exclude MUST stay empty.
```
The cook still wants the recipe that uses ghee. Treating it as an exclusion
returns dishes without ghee, which is the opposite of what was asked.

```
course is INDEPENDENT of include: filling it never excuses leaving include empty.
"a rice dish with paneer"  -> course rice, include [paneer]
```
The model kept putting the ingredient in `course` and leaving `include` empty.

**Caching**

```python
digest = hashlib.sha256(question.encode()).hexdigest()
```

Answers cache to disk by question. Evaluation replays the same 229 questions
repeatedly, and there is no reason to pay twice. This is also what makes the Stage
3 latency column honest - a cached run shows the graph's real traversal cost
without the LLM call it already paid for.

## `rag/graph/pantry.py` - the answer audit

```python
def resolve_pantry(lines, model="gpt4o-mini", use_llm=True):
    try:
        valid = set(known_ingredients())
    except Exception:
        # No graph. Fall back to vocabulary-only so the page still works
        ...
    leftovers = []
    for raw in raws:
        name = vocab.canonical(raw)
        if name and name in valid:
            out.mapped[raw] = name
        else:
            leftovers.append(raw)
```

**WHY the gate is graph membership, not vocabulary success** This was a bug I
introduced and caught while building it. `curated.py` deliberately lets an
unrecognised term become its own canonical name - safe for building a vocabulary,
useless here. "black gram" resolves to "black gram" and matches nothing, so
`if canonical(x)` is true and the LLM leg never fires. Asking whether the name is a
real `Ingredient` node is what routes genuine synonyms to the model.

Measured on real input:

    urad daal   -> urad dal    rules
    curd        -> yogurt      rules
    black gram  -> urad dal    rules
    kadai       -> unknown     correct, it is a pan

The rules did more than expected, so the model is a genuine fallback rather than
the main path.

```python
def can_i_make(key: str, pantry: list[str], max_swaps: int = 6) -> dict:
    needed = sorted({n for n in (card.get("ingredients") or []) if n})
    have = set(pantry)
    missing = [n for n in needed if n not in have]
    swaps, buy = {}, []
    for item in missing[:max_swaps]:
        covered = [c["name"] for c in traverse.substitutes(item, limit=8)
                   if c["name"] in have]
        if covered:
            swaps[item] = covered
        else:
            buy.append(item)
```

**WHY it identifies the recipe by cited source, never by parsing the answer**
Generated prose is not a data structure. Re-deriving an ingredient list from it
would be a second extractor to maintain and a second thing to be wrong.
`Passage.meta` already carries `source` and `recipe_ids`, so the graph key is just
those two joined.

`max_swaps` caps the substitute lookups, because each is a complex Cypher query and
a recipe missing 15 ingredients does not need 15 round trips to tell you to go
shopping.

---

# LAYER 6 - Measurement

## `rag/eval/golden.py` - the only file that opens the answer key

```python
@lru_cache(maxsize=1)
def load_golden() -> tuple[GoldenRecipe, ...]:
    if not golden_path().exists():
        return ()
    ...

def questions_for_eval() -> tuple[dict, ...]:
    return load_questions() or load_user_questions()
```

**WHY nothing on the answering path may import this** A benchmark the system can
see is a benchmark the system will fit. The separation is enforced by convention
rather than by types, which is one of the honest weaknesses.

**WHY absence returns empty rather than raising** Requiring a hand-labelled truth
set before anything can be measured is the largest barrier to using this on your own
documents - and only *some* metrics need labels. The split is questions versus
labels:

    no questions        nothing is measurable
    questions only      purity, self-sufficiency, latency, cost, diversity,
                        faithfulness, relevancy, cookability, abstention,
                        AND constraint compliance
    questions + labels  adds recall, hit@k, MAP, k@90, set recall

```python
out.append({"query_id": f"u_{index:04d}", "question": text,
            "family": "user", "relevant_ids": []})
```

`relevant_ids` is empty **by construction**. Evaluators read that as "do not score
the label-dependent metrics" rather than "no recipe is relevant", which would report
a false zero.

**Why recall specifically cannot come from an LLM judge:** to know a retriever
missed something you have to know what existed. A judge only ever sees what you
showed it. That is bookkeeping, not reasoning, and no model quality fixes it.

## `rag/eval/stage3_ranking.py` - the biggest evaluator, and the one that matters

```python
def _r(value, digits: int):
    return None if value is None else round(value, digits)

def _mean(values, digits: int):
    present = [v for v in values if v is not None]
    return round(statistics.fmean(present), digits) if present else None
```

**WHY these two helpers exist** `round(None, 3)` raises `TypeError`, and
`statistics.fmean` cannot average a list containing `None`. Once unlabelled runs
became possible, every aggregation needed to skip missing values rather than crash.
These were added after exactly that failure.

```python
if relevant:
    h = hit_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    m = mean_average_precision(ranked, relevant, k)
else:
    h = r = m = None
d = diversity_at_k(hits, k)
```

`None` renders blank in the table. A `0` would claim the retriever found nothing -
a different and false statement from "we were never told what correct looks like".

```python
cr = None
if extractor is not None:
    try:
        constraints = extractor.extract(question["question"])
        cr = constraint_respected(hits, constraints, ings)
        if cr is not None and cr < 1.0:
            for rid, offending in violations(hits, constraints, ings):
                breaches.append({...})
    except Exception:
        cr = None
```

**WHY breaches are recorded, not just counted** A compliance score of 0.40 tells
you there is a problem. The breach list tells you it returned Nut Milk for "avoiding
nuts", which is what makes the finding communicable. The number is the alarm; the
breach list is the diagnosis.

`per_family` splits results by question family, and **that split is what found the
zero**:

    constraint family   0.000 for every retriever

All 14 of its questions compare numbers - under 30 minutes, fewer than six
ingredients - and nothing in the pipeline compares numbers. A single blended average
would have buried that as a slightly lower score.

## `rag/eval/stage2_chunking.py`

**WHAT** Scores chunkers on purity, recall, k@90, self-sufficiency.

**WHY no model is involved** Every metric here is arithmetic over offsets:

    purity            chunks touching exactly one recipe / all chunks
    recall            can the right chunk be found at all, best case
    k@90              how many chunks to reach 90% recall
    self-sufficiency   can a chunk answer alone, or does it need a neighbour

`recall 0.745` for `recursive` is the damning number: a quarter of the time the
correct answer **cannot be retrieved by any search**, because no single chunk
contains it. The information was destroyed at cutting time, before search was
involved.

## `rag/eval/judge.py` - LLM as judge

```python
GEval(name="Cookable",
      rubric=[Rubric(score_range=(0, 2), expected_outcome="..."), ...])
```

**A fixed bug worth recording.** `GEval(rubric="some string")` fails with
`'str' object has no attribute 'score_range'` - it needs `Rubric` objects. And
because all three metrics were built in **one dict**, that single failure took out
faithfulness and relevancy too. They are constructed per-metric now, so one broken
judge does not blank the column next to it.

```python
if refused:
    return None        # Cookable scored refusals 0.0 before this
```

A refusal is not an uncookable answer. Scoring it zero punished the strategy that
correctly declined.

## `rag/eval/atoms.py` and `tokens.py`

Text normalisation so "500 gm rice" and "rice" match when checking whether an
answer invented an ingredient. Shared so the invention check and the golden loader
tokenise identically - if they differed, the check would flag correct answers.

---

# LAYER 7 - Runtime

## `rag/pipeline.py` - the seam between measuring and using

**WHAT** The configuration the Lab writes and the Kitchen runs.

**HOW**

```python
@dataclass
class Pipeline:
    chunker: str = "recursive"
    source: str = "rrf"
    use_mmr: bool = False
    use_rerank: bool = False
    use_graph: bool = False
    k: int = 5
    query_transform: str = "passthrough"
    strategy: str = "stuff_strict"
    max_new_tokens: int = 350   # 120 truncated recipes mid-step
```

**WHY on disk rather than in session state**, from the docstring:

> Keeping it on disk rather than in session state means the chat survives a
> restart and the configuration can be inspected, diffed and committed.

That last word matters. `data/eval/pipeline.json` is a committed artifact, so the
choice is reviewable in a diff rather than living in someone's browser tab.

`max_new_tokens: int = 350` carries a bug in its comment. At 120 the model ran out
of tokens mid-instruction, so answers ended halfway through step 3 and looked like
a model failure rather than a budget failure.

### `retriever_spec` - composing four controls into one call

```python
def retriever_spec(self) -> tuple[str, dict[str, Any]]:
    inner, params = self.source, {}
    if self.source in ("hybrid", "rrf"):
        params["sparse"] = self.sparse
    if self.use_mmr:
        inner, params = "mmr", {"base": inner}
    if self.use_rerank:
        inner, params = "cross_encoder", {"first_stage": inner}
    if self.use_graph:
        inner, params = "graph_hybrid", {"base": inner}
    return inner, params
```

**HOW this works** Each `if` wraps the previous result. `inner` starts as the base
retriever name and becomes the *argument* to the next wrapper. So
`rrf + rerank + graph` builds:

    graph_hybrid(base=cross_encoder(first_stage=rrf(sparse=bm25)))

**WHY the order is fixed**, from the docstring:

> the graph decides membership, then a source ranks, then diversity or reranking
> reshuffles. Reranking wraps the source, and the graph wraps whatever that
> produced, so the graph is outermost - a gate applied after reranking is still a
> gate.

The graph must be outermost. If reranking ran *after* the gate, it could not
reintroduce a banned recipe - but it would be scoring items the gate already
approved, which is wasted work. Outermost means the gate has the final word.

This is why Stage 3 is presented as four independent toggles rather than one list of
nine names. The nine combinations are *generated* from the toggles, so adding a new
base retriever gives you six new configurations for free.

### `load` - tolerating an older file

```python
known = {f for f in Pipeline.__dataclass_fields__}
return Pipeline(**{k: v for k, v in data.items() if k in known})
```

**WHY filter the keys** `Pipeline(**data)` raises `TypeError` on an unexpected key.
If a field is renamed or removed, every previously saved `pipeline.json` becomes
unloadable and the Kitchen breaks. Filtering to known fields means old files still
load, with removed fields ignored and new fields taking their defaults.

`__dataclass_fields__` is the introspection dict every dataclass gets
automatically.

```python
except json.JSONDecodeError:
    return None
```

A truncated or hand-edited file returns `None`, which the Kitchen renders as "no
pipeline saved yet, go to the Lab" rather than a traceback.

### `build_runtime` - config into live objects

```python
def build_runtime(pipeline: Pipeline):
    from .corpus import build_corpus
    from .loaders import load_latest_ingested
    from .registry import build, discover

    discover("cognitive_kitchen.rag.chunking", ..., "cognitive_kitchen.rag.strategies")
    corpus = build_corpus(load_latest_ingested(source_type=pipeline.corpus_source))
    embedder = build("embedder", "st")
    passages = build("chunker", pipeline.chunker, **(pipeline.chunker_params or {})).split(corpus)

    name, params = pipeline.retriever_spec()
    if name in ("dense", "hybrid", "rrf", "mmr", "cross_encoder", "graph_hybrid"):
        params["embedder"] = embedder
    retriever = build("retriever", name, **params)
    retriever.index(passages)
    ...
    return {"corpus": corpus, "passages": passages, "retriever": retriever,
            "transform": transform, "generator": generator,
            "strategy": strategy, "embedder": embedder}
```

**WHY the imports are inside the function** Importing `corpus` and `registry` at
module top level would pull in the whole retrieval stack whenever anything reads a
pipeline config - including the Kitchen just checking whether one exists. Local
imports defer that until the runtime is actually being built.

**WHY the embedder is only passed to some retrievers** `bm25` and `tfidf` take no
embedder and would raise `TypeError` on an unexpected keyword. The membership test
is explicit rather than a try/except, because silently swallowing a `TypeError`
is exactly how the `chunk_size` bug hid.

**WHY it returns a dict rather than an object** The Kitchen needs several pieces -
retriever, transform, strategy, generator - and a dict is the smallest thing that
carries them. It is cached by `@st.cache_resource`, so it is built once per
configuration.

## `rag/memory.py` - conversation, and the part that matters

```python
def resolve(question: str, conversation) -> tuple[str, bool]:
    ...
```

**WHAT** Persists conversations to `data/chat/` and rewrites follow-ups.

**WHY the rewriting is the whole point**, from the docstring:

> a follow-up is rewritten against the recent turns before retrieval. That second
> part is the one that matters -- "what about without dairy?" has no subject, so a
> retriever handed it alone returns nothing useful. Without resolution, memory
> would be a transcript and nothing more.

    turn 1   "a rice dish with paneer"
    turn 2   "what about without dairy?"
             -> rewritten: "a rice dish without dairy"

Returning `(query, was_rewritten)` lets the UI show `retrieved as: ...`, so the
user can see when their question was reinterpreted. Silent rewriting would be worse
than none - you could not tell whether a bad answer came from bad retrieval or bad
rewriting.

## `rag/telemetry.py` - where the time and money went

```python
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

def price_of(model: str, tokens_in: int, tokens_out: int) -> float:
    key = next((k for k in PRICES if k in model), None)
    if key is None:
        return 0.0
    rate_in, rate_out = PRICES[key]
    return round(tokens_in / 1e6 * rate_in + tokens_out / 1e6 * rate_out, 6)
```

**`if k in model` is a substring test, not equality.** So `"openai:gpt-4o-mini"`
matches the `"gpt-4o-mini"` key. That is deliberate - generators prefix their names.

**Note the ordering hazard:** `"gpt-4o"` is a substring of `"gpt-4o-mini"`. Because
`next` takes the first match and dicts preserve insertion order, `gpt-4o-mini` must
come first. It does. Swap the two lines and mini calls are priced 16x too high.

Unknown models return 0.0, which is correct here - local models genuinely cost
nothing.

```python
@contextmanager
def timed(self, name: str, **meta) -> Iterator[Span]:
    start = time.perf_counter()
    try:
        yield self.span(name)
    finally:
        self.record(name, time.perf_counter() - start, **meta)
```

**`@contextmanager` explained** It turns a generator into something usable with
`with`. Everything before `yield` is setup, everything after is teardown:

```python
with ledger.timed("retrieve"):
    hits = retriever.search(query, k)     # timed automatically
```

`finally` means the timing is recorded **even if the body raises**. Without it, a
failed retrieval would vanish from the breakdown, and the totals would silently not
add up.

`time.perf_counter()` not `time.time()`. `perf_counter` is monotonic and
high-resolution; `time.time()` can jump backwards if the system clock adjusts,
producing negative durations.

```python
def span(self, name: str) -> Span:
    return self.spans.setdefault(name, Span(name))
```

`setdefault` means calling `timed("retrieve")` in a loop accumulates into one span
rather than overwriting, so `calls` counts correctly and `per_call_s` is meaningful.

```python
def traced(name: str) -> Callable:
    def decorator(fn):
        if not configure_langsmith():
            return fn                     # no-op when not configured
        try:
            from langsmith import traceable
        except Exception:
            return fn                     # no-op when not installed
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return traceable(name=name)(fn)(*args, **kwargs)
        return wrapper
    return decorator
```

**WHY it degrades to nothing** Two escape hatches: not configured, and not
installed. Either returns the original function unchanged, so the pipeline has **no
hard dependency on LangSmith**. Observability that breaks the app when the
observability service is absent is worse than no observability.

`@functools.wraps(fn)` copies the original's name, docstring and signature onto the
wrapper. Without it, every traced function would report as `wrapper` in tracebacks
and `help()`.

---

# LAYER 8 - The interface

## The one Streamlit fact everything else follows from

**Streamlit re-runs the entire script, top to bottom, on every interaction.** Not
"updates the changed widget" - re-executes the file from line 1.

This does not work:

```python
count = 0
if st.button("Add one"):
    count += 1
st.write(count)          # always 1
```

The script restarted, so `count = 0` ran again.

## `ui/chrome.py` - shared page setup

```python
def page(title: str, icon: str = "C", layout: str = "wide") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout=layout)
    st.markdown((HERE / "theme.css").read_text(encoding="utf-8"),
                unsafe_allow_html=True)
```

**WHY this exists** `theme.css` was loaded by the console and by nothing else.
Streamlit runs each page as its own script, so **a stylesheet does not carry
across** - every page has to ask. The Lab and Kitchen rendered as unstyled
Streamlit while only the landing page looked designed.

`set_page_config` must be the **first** Streamlit call on a page, which is why both
steps are wrapped in one function that pages call before anything else.

```python
def stepper(steps: list[tuple[str, str]]) -> None:
    out = ['<div class="stepper">']
    seen_pending = False
    for label, value in steps:
        if value:
            state, shown = "done", value
        elif not seen_pending:
            state, shown = "current", "choose"
            seen_pending = True
        else:
            state, shown = "todo", "-"
```

`seen_pending` marks only the **first** unlocked stage as current; later ones are
dimmed. That is what makes it read as a chain with a position rather than a row of
equal boxes.

## `ui/theme.css` - the palette rule

```css
/* Any block that sets a background also sets its own text colour, and any
   block that does NOT set a background inherits from Streamlit's theme. */
```

**WHY this rule is written into the file** Every section heading was invisible in
dark mode. The cause was hardcoded near-black text on elements with no background
of their own - which only works if you assume a light theme. Cards use translucent
`rgba()` backgrounds so they read on either theme, and accents are mid-tone purples
that clear contrast on both.

The rule is a comment in the stylesheet specifically so the bug cannot be
reintroduced by someone adding a class later.

## `ui/warmup.py` - load the slow things in advance

```python
def start(warm_generator: bool = False) -> None:
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_run, args=(warm_generator,),
                     name="ck-warmup", daemon=True).start()
```

**WHY a background thread** The first visit to the Lab or Kitchen otherwise pays
for the embedding model and the first Neo4j round trip at once, which reads as the
app being slow when it is only cold.

`daemon=True` means the thread will not keep the process alive on exit. Without it,
closing the app would hang until the model finished loading.

`with _lock` around the `_started` check makes it safe against two browser tabs
starting simultaneously. Checking and setting a flag is not atomic - both threads
could pass the check before either sets it.

`global _started` is required because Python treats assignment inside a function as
creating a local by default.

**WHY the answering model is excluded by default** It is the one item that costs
real memory - roughly 6 GB resident - and it is already lazy, loading on the first
*answer* rather than on page open. So warming it improves the first question, not
the first click. On a 16 GB machine also running the Lab, that trade is a choice
rather than a default.

```python
except Exception:
    _mark("embeddings", f"failed: {type(exc).__name__}")
```

Every step swallows its own failure and records it. A blocked VPN delays nothing and
breaks nothing.

## `ui/models.py` - switching models at runtime

```python
def apply(role: str, model_id: str) -> bool:
    model_id = (model_id or "").strip()
    if not model_id or model_id == current(role):
        return False
    setattr(settings, role, model_id)
    return True
```

**WHY `setattr` works** `settings` is a module-level singleton, so mutating it
changes the value for the whole process instantly. Every factory already does
`model_name or settings.embedding_model`, so the plumbing was already there - only
the control was missing.

**Returning `bool`** so the caller knows whether to clear the cache. No change
means no cache invalidation, which avoids throwing away a loaded model for nothing.

```python
def verify(role: str, model_id: str) -> tuple[bool, str]:
    try:
        if role == "embedding_model":
            model = SentenceTransformer(model_id, device="cpu")
            return True, f"loaded, {int(model.get_sentence_embedding_dimension())} dimensions"
        ...
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return False, first[:160]
```

**WHY Verify is separate from Use** A model id is only a string until something
tries to load it, and discovering a typo forty seconds into a nine-retriever sweep
is the worst possible time. Verify pays that cost once, up front, and **reports
instead of raising** - hence returning a tuple rather than throwing.

`str(exc).splitlines()[0]` takes the first line, because HuggingFace errors are
often twenty lines of suggested alternatives.

## `ui/explain.py` - the documentation is data

```python
CHUNKERS: dict[str, tuple[str, str]] = {
    "recipe": (
        "One chunk per recipe - uses the boundaries ingestion already found",
        "The six above all INFER where one thing ends and the next begins. "
        "Ingestion already recorded it..."),
}
```

**WHY a dict rather than strings in the page**, from the docstring:

> Kept as data in one place rather than scattered through the page, so a stage can
> render "what am I choosing between" and "how was this number produced" without
> the reader having to open the source to find out.

The Lab loops over it:

```python
def show_options(title: str, table: dict, keys) -> None:
    with st.expander(f"What each {title} does"):
        for key in keys:
            if key in table:
                headline, detail = table[key]
                st.markdown(f"**`{key}`** - {headline}  \n{detail}")
```

Adding a strategy means adding one dict entry. **Nothing in the Lab is a black
box**, and the explanation cannot drift out of sync with the dropdown because both
read the same dict.

```python
def models_in_play(stage: str) -> list[tuple[str, str]]:
    """Which models a stage actually runs, read from .env."""
```

Reads live settings rather than hardcoding names, so the stage header is correct
after a model switch.

## `ui/progress.py` - live commentary

```python
note = Commentary("Comparing retrievers", total=len(combos))
note.say(f"{len(passages)} chunks from {SS['locked']['chunker']}")
note.step(label, f"composed as {name}")
note.result(f"hit@{k} {metrics['hit_at_k']} ... {elapsed:.1f}s")
note.warn(f"{name}: {type(exc).__name__}: {exc}")
note.finish(f"{len(rows)} configurations compared")
```

**WHY narration and not a progress bar** A bar says how far along you are. This says
*what is happening*, which for a sixty-second sweep is the difference between
waiting and watching. `warn` also means a single failing retriever reports itself
and the sweep continues.

## `ui/pages/2_RAG_Lab.py` - the stage comparison

```python
SS = st.session_state
SS.setdefault("locked", {})
SS.setdefault("results", {})
```

**`setdefault`, not `=`.** This is the single most important line in the file. On
every rerun, `SS["locked"] = {}` would wipe your locked pipeline. `setdefault`
creates the key only if absent, so choices survive navigation.

```python
@st.cache_resource(show_spinner=False)
def corpus_for(source: str): ...

@st.cache_resource(show_spinner=False)
def embedder(): ...

@st.cache_data(show_spinner=False)
def chunk(source: str, chunker: str, size: int, overlap: int): ...
```

**Two cache flavours, and the distinction matters**

    @st.cache_resource   LIVE THINGS. models, connections. returns the SAME object
    @st.cache_data       VALUES. returns a COPY, so callers cannot corrupt the cache

The embedder must be `cache_resource` - you want one shared model, not forty
copies. Chunks are `cache_data` because they are just a list of values.

```python
@st.cache_data(ttl=120, show_spinner=False)
def _graph_badge() -> str: ...
```

**WHY the TTL** The header used to call `client.health()` on every render - a Neo4j
round trip before the page appeared. Since Streamlit reruns on any navigation,
opening the Lab meant waiting on the network. A node count stale by up to two
minutes is worth far more than a page that stalls every time you come back.

```python
with st.expander("STAGE 3 · Retrieval", expanded=not is_locked("retrieval")):
```

`expanded=not is_locked(...)` auto-opens whichever stage you have not done and
collapses the finished ones, so returning to the page puts you where you left off.

```python
slot = st.empty()
def paint_partial() -> None:
    slot.dataframe(pd.DataFrame([...]))

paint_partial()
for index, (base, mmr, rerank, graph) in enumerate(combos):
    status[index] = "running"
    paint_partial()
```

**WHY status is tracked per configuration** rather than inferred from how many rows
have been appended: a configuration that raises appends nothing, which would shift
every later row up by one and mislabel them all.

```python
def graph_offline(exc: Exception) -> None:
    st.warning("The graph is not reachable, so this panel cannot answer right now.")
    with st.expander("Details"):
        st.caption("Neo4j speaks Bolt on port 7687. A VPN blocking that port "
                   "produces exactly this...")
```

**WHY a named cause and not a traceback** This is the most common failure on the
page and it has nothing to do with the query. Verified: port 7687 times out at 8
seconds while port 443 on the same host opens in 0.11s. The diagnosis belongs where
the error appears, not buried in a file.

## `ui/pages/3_Kitchen.py` - the chat

```python
@st.cache_resource(show_spinner="Loading the pipeline...")
def runtime(signature: str):
    return P.build_runtime(P.load())

parts = runtime(config.label() + str(config.k) + str(config.max_new_tokens))
```

**WHY a `signature` argument that is never used inside** Streamlit caches on the
*arguments*. The function ignores it, but passing a string derived from the config
means a different pipeline gets a different cache entry. Change the locked strategy
and you get a rebuilt runtime; change nothing and you get the cached one. **The
argument is the cache key.**

```python
st.session_state["last_sources"] = unique
```

Stored because Streamlit reruns on the button press, so the answer's cited recipes
have to outlive the turn that produced them.

```python
head = next((ln.strip() for ln in passage.text.splitlines() if ln.strip()), ids[0])
```

The label comes from the first non-empty line of the passage, not from the graph -
reading a title should not need a network round trip on every answer.

```python
question = st.chat_input("Ask about a recipe, or what you can cook")
question = question or st.session_state.pop("pending_question", None)
```

**WHY `.pop`** A starter button writes into `pending_question` and reruns. `pop`
reads it and removes it in one step, so the question is not re-asked on the next
rerun. Using `.get` would loop forever.

```python
def meta_chips(meta: str) -> None:
    for part in [x.strip() for x in meta.split("\u00b7") if x.strip()]:
        kind = "ok" if part.startswith("\u2713") else (
            "warn" if part.startswith("\u26a0") else "")
```

The footer is **stored as plain text** and the styling derived on render, so saved
conversations pick up a new look with no migration.

## `ui/app.py` - the console

```python
warmup.start(st.session_state.get("warm_generator", False))

@st.cache_data(ttl=300, show_spinner=False)
def local_stats() -> dict: ...      # disk only, safe to block on

@st.cache_data(ttl=300, show_spinner=False)
def graph_nodes() -> int: ...       # network, called LAST

pills_slot = st.empty()
...
pills_slot.markdown(...)            # final statement in the script
```

**WHY the split by cost** The page used to block on a Neo4j round trip before
rendering anything. Disk counts are fast and safe to block on; the graph query is
the **last statement in the script** and writes into a reserved slot. The slot holds
its place in the layout, so the pills still appear near the top while the page
paints without waiting.

`show_spinner=False` on both, because "Running local_stats()" is not information a
user wants.

```python
if up is not None:
    fingerprint = f"{up.name}:{len(up.getvalue())}"
    if st.session_state.get("ingested_upload") != fingerprint:
        st.session_state["ingested_upload"] = fingerprint
        ... POST and ingest ...
```

**WHY the fingerprint guard** Single-step upload means ingesting on file-select. But
Streamlit reruns on *any* interaction, so an unguarded `if up is not None` would
re-POST the file every time anything else on the page was clicked. Fingerprinting by
name and size means one file ingests once, while a genuinely different file still
triggers a fresh run.

---

# The architecture in six sentences

    1  types.py defines Passage with character offsets into one corpus string,
       so chunk quality is arithmetic rather than a lookup.

    2  registry.py lets strategies declare themselves, so 8 chunkers and 9
       retrievers exist with no list of them anywhere.

    3  corpus.py builds the text and the offsets in ONE pass, because doing it
       in two would mean finding recipes by searching.

    4  Because the graph also emits Passage, it can impersonate a retriever and
       be judged on the same metrics - which is what made the head-to-head
       comparison possible at all.

    5  eval/golden.py is the only file that may read the answer key, and it now
       treats absence as a state rather than an error.

    6  pipeline.json is the seam: the Lab writes it, the Kitchen reads it, so
       evaluation and deployment are the same artifact.

# The recurring design habit

Almost every non-obvious line in this codebase is a fixed bug with its reason
recorded next to it:

    meta["source"]              a PDF recipe inherited a web recipe's eligibility
    registry.signature()        size/overlap controls silently did nothing
    merge_intervals             coverage exceeded 1.0 by double counting
    filter_recipes returns []   "freshly" returned all 168 recipes
    course in filter_recipes    "bread with no dairy" returned 86 recipes
    Rubric objects per metric    one bad rubric blanked all three judges
    refused as a field           correct refusals were scored 0.0
    graph outermost in spec      a gate after reranking is still a gate
    rfind on newline             oversized recipes cut mid-sentence
    fingerprint on upload        re-ingested on every unrelated click
    color: inherit in CSS        headings invisible in dark mode
    setdefault not =             locked pipeline wiped on every rerun
    cache key includes model     stale vectors labelled as a new model

The comments are not documentation of what the code does. They are the record of
what went wrong, so it cannot go wrong again silently.
