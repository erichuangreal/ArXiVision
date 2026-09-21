import { useRef, useState } from 'react';
import { Masthead } from '../components/Masthead';
import { Icon } from '../components/icons';
import './HowItWorks.css';

// Six real pipeline stages, in the order the code actually runs them
// (ingest.py's _run_pipeline). Every claim here is accurate to the shipped
// system, not a simplified explainer - see DESIGN.md/PRODUCT.md for why
// that accuracy is the point of this page.
const STAGES = [
  {
    id: 'search',
    label: 'Search',
    lede: 'Tries a live arXiv search first; a single rate-limited response falls back to the offline index instantly.',
    stat: '429 → instant local fallback, no retry wait',
    status: "Searching arXiv for 'your topic'...",
    problem:
      "arXiv's live search API (export.arxiv.org/api/query) rate-limits with a 429 - expected occasionally, not a bug. The old approach retried with backoff on every attempt, which could mean minutes of waiting before an expedition even started.",
    fix:
      "Fixed by trying live search exactly once, with zero retries: on a 429 or any other request failure, it falls back immediately to an offline SQLite full-text index built from arXiv's own published metadata snapshot (via Kaggle). A rate limit now costs one fast failed request, not a multi-minute retry loop.",
    body: [
      "Every expedition starts by querying arXiv's live API directly - arXiv's own index is always current, while the local snapshot is only as fresh as the last time it was rebuilt. That one attempt uses no retries, so a failure resolves immediately rather than blocking on backoff. Either path - live or local - then runs an exact-phrase match first, with a broader match requiring every word to appear (not necessarily together) as the fallback if that finds nothing.",
    ],
  },
  {
    id: 'download',
    label: 'Download',
    lede: "Fetches PDFs straight from arxiv.org - the one link in the pipeline that never broke.",
    stat: 'arxiv.org/pdf — never failed in testing',
    status: 'Downloading 3/10: Effects of Varying LLM Access...',
    body: [
      'PDFs are fetched directly from arxiv.org/pdf/{id} - different infrastructure from the search API, and it never failed once in testing. Every concurrent user’s downloads share one paced lock, so simultaneous expeditions queue instead of colliding; a failed download (a withdrawn paper, a dead link) is recorded and skipped without taking down the rest.',
    ],
  },
  {
    id: 'extract',
    label: 'Extract & Chunk',
    lede: 'Splits each paper into overlapping chunks so no fact is ever cut in half.',
    stat: '500 tok chunks · 50 tok overlap',
    status: 'Extracting 7/10: 003_paper.pdf (ok)',
    body: [
      'Each PDF’s text is extracted page by page with PyMuPDF. Near-duplicate content (repeated boilerplate, running headers) is filtered out with MinHash-based deduplication before anything gets chunked.',
      'Text is split into 500-token chunks with 50-token overlap, using a token-aware splitter. The overlap means a sentence that falls on a chunk boundary still appears whole in at least one chunk, so a fact never gets silently cut in half.',
    ],
  },
  {
    id: 'embed',
    label: 'Embed & Index',
    lede: 'Embeds every chunk and builds a parallel keyword index for the hybrid search a query needs.',
    stat: '~4,000 chunks / topic',
    status: 'Creating embeddings for 638 chunk(s)...',
    body: [
      'Every chunk is embedded with OpenAI’s text-embedding-3-small into a per-user Chroma vector store - re-running on an unchanged corpus skips re-embedding entirely, since each chunk is content-addressed. A second, independent BM25 keyword index is built over the same chunks here too - the other half of what makes retrieval hybrid one stage from now.',
    ],
  },
  {
    id: 'retrieve',
    label: 'Hybrid Retrieve',
    lede: 'Fuses keyword and embedding search so a terse fact beats its own wordier restatement.',
    stat: 'hit@1 40% → 60%   ·   hit@4 60% → 100%',
    status: 'Retriever set up: hybrid embeddings + BM25 fusion',
    problem:
      'A paper often states a fact tersely in one place (the abstract) and restates it at length elsewhere (the results section). Embedding search alone consistently ranked the longer, wordier restatement above the terse original - it has more matching vocabulary, so it "looks" more similar, even when the terse original is the actual answer.',
    fix:
      'Fixed by running keyword (BM25) search and embedding search side by side, then combining both rankings with Reciprocal Rank Fusion. Measured, not assumed: this moved hit@1 from 40% to 60% and hit@4 from 60% to 100% on the same real corpus, before and after.',
    body: [
      'BM25 scores chunks by exact word overlap; embeddings score by semantic similarity. A chunk ranking well in either list - or both - rises to the top of the combined ranking, so BM25 catches the terse original a query shares words with, while embeddings catch conceptually related chunks that don’t share exact wording. Neither alone is enough; together they cover each other’s blind spot.',
    ],
  },
  {
    id: 'answer',
    label: 'Answer & Evaluate',
    lede: 'Answers with citations, then every answer is measured across four categories on your own verification ledger.',
    stat: 'retrieval · grounding · abstention · correctness',
    status: 'Drafting verification questions...',
    body: [
      'The top 4 fused chunks are "stuffed" into a prompt with the question, and the model answers citing the specific page each claim comes from.',
      'Every answer is then measured, not just trusted, across the same four categories shown on your Evaluation page: retrieval (was the right evidence in the top 4 chunks?), grounding (do the citations point at retrieved text that actually supports the claim?), abstention (does it decline when it lacks evidence, instead of guessing?), and correctness (does an independent judge model agree the answer states the same fact as a reference). A stronger model authors one test question per newly-ingested paper to build this ledger, per user, per corpus.',
    ],
  },
];

// One small hand-built diagram per stage - real mechanics, not decoration.
// No charting library: everything here is flex/SVG built from the same
// ink-only, no-shadow, near-square vocabulary as the rest of the system.

function FlowNode({ children }) {
  return <div className="flow__node mono">{children}</div>;
}

function FlowArrow({ vertical }) {
  return <Icon.ArrowRight size={16} className={`flow__arrow${vertical ? ' flow__arrow--vertical' : ''}`} />;
}

function SearchDiagram() {
  return (
    <div className="flow" aria-hidden="true">
      <FlowNode>your topic</FlowNode>
      <FlowArrow />
      <FlowNode>live arXiv search (1 try)</FlowNode>
      <div className="flow__branch">
        <div className="flow__branch-row">
          <FlowArrow />
          <FlowNode>ok → candidate papers</FlowNode>
        </div>
        <div className="flow__branch-row">
          <FlowArrow />
          <FlowNode>429 / fails → local index</FlowNode>
          <FlowArrow />
          <FlowNode>candidate papers</FlowNode>
        </div>
      </div>
    </div>
  );
}

function DownloadDiagram() {
  const slots = 5;
  return (
    <div className="pace-strip" aria-hidden="true">
      <div className="pace-strip__track">
        {Array.from({ length: slots }).map((_, i) => (
          <div key={i} className="pace-strip__tick" style={{ left: `${(i / (slots - 1)) * 100}%` }} />
        ))}
      </div>
      <div className="pace-strip__intervals mono">
        {Array.from({ length: slots - 1 }).map((_, i) => (
          <span key={i} style={{ left: `${((i + 0.5) / (slots - 1)) * 100}%` }}>
            ≥3s
          </span>
        ))}
      </div>
      <p className="pace-strip__caption mono">
        every request, from every user, waits at least 3 seconds behind one shared lock
      </p>
    </div>
  );
}

function ExtractDiagram() {
  const chunks = 5;
  const width = 24; // percent width per chunk - overlapping washes, not gaps
  const step = 19; // 5-token overlap is ~5% of each 24%-wide chunk visually
  return (
    <div className="chunk-strip" aria-hidden="true">
      {Array.from({ length: chunks }).map((_, i) => (
        <div key={i} className="chunk-strip__chunk mono" style={{ left: `${i * step}%`, width: `${width}%` }}>
          chunk {i + 1}
        </div>
      ))}
      <p className="chunk-strip__caption mono">darker bands = the 50-token overlap shared by neighboring chunks</p>
    </div>
  );
}

function EmbedDiagram() {
  const rows = [
    { label: 'pages extracted', value: 2000, display: '~2,000+' },
    { label: 'chunks indexed', value: 4000, display: '~4,000+' },
  ];
  const max = Math.max(...rows.map((r) => r.value));
  return (
    <>
      <div className="flow" aria-hidden="true">
        <FlowNode>chunk</FlowNode>
        <div className="flow__branch">
          <div className="flow__branch-row">
            <FlowArrow />
            <FlowNode>embedding vector store (semantic)</FlowNode>
          </div>
          <div className="flow__branch-row">
            <FlowArrow />
            <FlowNode>BM25 keyword index (exact-word)</FlowNode>
          </div>
        </div>
      </div>
      <div className="hbar-chart" aria-hidden="true">
        {rows.map((r) => (
          <div className="hbar-chart__row" key={r.label}>
            <span className="hbar-chart__label mono">{r.label}</span>
            <div className="hbar-chart__track">
              <div className="hbar-chart__fill" style={{ width: `${(r.value / max) * 100}%` }} />
            </div>
            <span className="hbar-chart__value mono">{r.display}</span>
          </div>
        ))}
        <p className="hbar-chart__caption mono">measured on one real expedition, single topic</p>
      </div>
    </>
  );
}

const RETRIEVE_DATA = [
  { label: 'hit@1', before: 40, after: 60 },
  { label: 'hit@4', before: 60, after: 100 },
];

function RetrieveTable() {
  return (
    <table className="metric-table mono" aria-hidden="true">
      <thead>
        <tr>
          <th />
          <th>embeddings only</th>
          <th className="metric-table__hero-col">hybrid (BM25 + embeddings)</th>
        </tr>
      </thead>
      <tbody>
        {RETRIEVE_DATA.map((d) => (
          <tr key={d.label}>
            <td>{d.label}</td>
            <td>{d.before}%</td>
            <td className="metric-table__hero-col metric-table__hero-value">{d.after}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function AnswerDiagram() {
  const checks = ['retrieval', 'grounding', 'abstention', 'correctness'];
  return (
    <div className="flow flow--converge" aria-hidden="true">
      <div className="flow__checks">
        {checks.map((c) => (
          <div key={c} className="flow__node flow__node--check mono">
            <Icon.Check size={14} />
            {c}
          </div>
        ))}
      </div>
      <FlowArrow vertical />
      <FlowNode>your verification ledger</FlowNode>
    </div>
  );
}

const DIAGRAMS = {
  search: SearchDiagram,
  download: DownloadDiagram,
  extract: ExtractDiagram,
  embed: EmbedDiagram,
  retrieve: RetrieveTable,
  answer: AnswerDiagram,
};

export function HowItWorks() {
  const [active, setActive] = useState(0);
  const trackRef = useRef(null);
  const draggingRef = useRef(false);

  function setFromClientX(clientX) {
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    const index = Math.round(ratio * (STAGES.length - 1));
    setActive(index);
  }

  function handlePointerDown(event) {
    draggingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    setFromClientX(event.clientX);
  }

  function handlePointerMove(event) {
    if (!draggingRef.current) return;
    setFromClientX(event.clientX);
  }

  function handlePointerUp(event) {
    draggingRef.current = false;
    event.currentTarget.releasePointerCapture(event.pointerId);
  }

  const stage = STAGES[active];
  const fillPercent = (active / (STAGES.length - 1)) * 100;
  const Diagram = DIAGRAMS[stage.id];

  return (
    <div className="how-it-works">
      <Masthead />

      <div className="how-it-works__intro">
        <h1 className="how-it-works__title">How this works</h1>
        <p className="how-it-works__lede">
          Six real stages, in the order the code actually runs them - click through, or drag the marker, to trace
          one paper from a topic to a cited answer.
        </p>
      </div>

      <div className="trace">
        <div
          className="trace__track"
          ref={trackRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        >
          <div className="trace__fill" style={{ width: `${fillPercent}%` }} />
          <div className="trace__handle" style={{ left: `${fillPercent}%` }} aria-hidden="true" />
          {STAGES.map((s, i) => (
            <button
              key={s.id}
              type="button"
              className={`trace__stop ${i <= active ? 'trace__stop--filled' : ''}`}
              style={{ left: `${(i / (STAGES.length - 1)) * 100}%` }}
              onClick={() => setActive(i)}
              aria-label={s.label}
              aria-current={i === active}
              aria-controls="how-it-works-detail"
            />
          ))}
        </div>
        <div className="trace__labels mono">
          {STAGES.map((s, i) => (
            <span
              key={s.id}
              className={i === active ? 'trace__label--active' : ''}
              style={{ left: `${(i / (STAGES.length - 1)) * 100}%` }}
            >
              {s.label}
            </span>
          ))}
        </div>
        <p className="trace__hint mono">Click a stage to see how it works, or drag the marker along the rail.</p>
      </div>

      <section className="ledger-panel how-it-works__detail" id="how-it-works-detail" aria-live="polite">
        <p className="how-it-works__stage-heading mono">
          Stage {active + 1} / {STAGES.length} — {stage.label}
        </p>
        <p className="how-it-works__lede">{stage.lede}</p>
        <p className="how-it-works__status mono">{stage.status}</p>
        <p className="how-it-works__stat mono">{stage.stat}</p>
        {Diagram && (
          <div className="how-it-works__diagram">
            <Diagram />
          </div>
        )}
        {stage.problem && (
          <p className="how-it-works__note how-it-works__note--caution">
            <Icon.Warning size={14} />
            {stage.problem}
          </p>
        )}
        {stage.body.map((paragraph, i) => (
          <p key={i} className="how-it-works__paragraph">
            {paragraph}
          </p>
        ))}
        {stage.fix && (
          <p className="how-it-works__note how-it-works__note--verified">
            <Icon.Check size={14} />
            {stage.fix}
          </p>
        )}
      </section>
    </div>
  );
}
