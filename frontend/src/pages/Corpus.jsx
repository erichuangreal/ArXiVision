import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api, ApiError, describeError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Masthead } from '../components/Masthead';
import { Icon } from '../components/icons';
import './Corpus.css';

function groupByTopic(papers) {
  const groups = new Map();
  for (const paper of papers) {
    const topic = paper.topic || 'uncategorized';
    if (!groups.has(topic)) groups.set(topic, []);
    groups.get(topic).push(paper);
  }
  return groups;
}

export function Corpus() {
  const { apiKey } = useAuth();
  const navigate = useNavigate();

  const [papers, setPapers] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(() => new Set());

  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState(null);
  const [searchError, setSearchError] = useState(null);

  useEffect(() => {
    api
      .listPapers(apiKey)
      .then((data) => setPapers(data.papers))
      .catch((err) => {
        if (err instanceof ApiError && err.status === 400) setPapers([]);
        else setError('The ledger could not be reached.');
      });
  }, [apiKey]);

  const groups = useMemo(() => groupByTopic(papers || []), [papers]);

  const selectedPapers = useMemo(
    () => (papers || []).filter((p) => selected.has(p.paper_id)),
    [papers, selected]
  );
  const selectedTopic = selectedPapers[0]?.topic ?? null;

  function toggle(paper) {
    setSelected((prev) => {
      // A comparison can only span one expedition's specimens - switching
      // topics starts a fresh selection rather than silently mixing them.
      if (selectedTopic && selectedTopic !== paper.topic) {
        return new Set([paper.paper_id]);
      }
      const next = new Set(prev);
      if (next.has(paper.paper_id)) next.delete(paper.paper_id);
      else next.add(paper.paper_id);
      return next;
    });
  }

  function handleSearch(event) {
    event.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setSearchError(null);
    api
      .search(apiKey, query.trim())
      .then((data) => setResults(data.results))
      .catch((err) => setSearchError(describeError(err, 'The search could not be completed.')))
      .finally(() => setSearching(false));
  }

  function handleCompare() {
    navigate('/collections/new', { state: { papers: selectedPapers } });
  }

  return (
    <div className="corpus">
      <Masthead />
      <p className="corpus__intro">
        A specimen is a paper in your corpus, grouped by the expedition that brought it in. A comparison can
        only span specimens from the same expedition.
      </p>

      <form className="corpus__search" onSubmit={handleSearch}>
        <input
          className="corpus__search-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search retrieved passages across the whole corpus…"
        />
        <button type="submit" className="corpus__search-submit" disabled={searching}>
          {searching ? <Icon.Loading className="spin" size={16} /> : <Icon.ArrowRight size={16} />}
        </button>
      </form>

      {searchError && <p className="notice">{searchError}</p>}

      {results && (
        <section className="corpus__results">
          <h2 className="section-title">Passages matching “{query}”</h2>
          <ol className="ruled-list">
            {results.map((r, i) => (
              <li key={i} className="corpus__result">
                <p className="corpus__result-text">{r.text}</p>
                <p className="corpus__result-meta mono">
                  {r.metadata.title} — p.{r.metadata.page_number}
                </p>
              </li>
            ))}
            {results.length === 0 && <p className="notice">No passages matched.</p>}
          </ol>
        </section>
      )}

      <main className="corpus__body">
        {error && <p className="notice">{error}</p>}
        {!error && papers === null && <p className="notice">Reading the ledger…</p>}
        {papers && papers.length === 0 && (
          <p className="notice">
            No specimens catalogued yet.{' '}
            <Link to="/" className="corpus__cabinet-link">
              Begin an expedition from the cabinet
            </Link>{' '}
            to bring some in.
          </p>
        )}

        {[...groups.entries()].map(([topic, group]) => (
          <details key={topic} className="corpus__group" open>
            <summary className="section-title corpus__group-title">
              {topic} <span className="corpus__group-count mono">({group.length})</span>
            </summary>
            <ol className="ruled-list corpus__list">
              {group.map((p) => (
                <li key={p.paper_id} className="corpus__row">
                  <label className="corpus__row-label">
                    <input
                      type="checkbox"
                      checked={selected.has(p.paper_id)}
                      onChange={() => toggle(p)}
                    />
                    <span className="corpus__row-title">{p.title}</span>
                  </label>
                  <span className="corpus__row-meta mono">
                    {p.authors?.join(', ')} · {p.arxiv_id}
                  </span>
                </li>
              ))}
            </ol>
          </details>
        ))}
      </main>

      {selected.size > 0 && (
        <div className="corpus__action-bar">
          <span className="mono">
            {selected.size} specimen{selected.size === 1 ? '' : 's'} selected from “{selectedTopic}”
          </span>
          <button type="button" className="corpus__compare" onClick={handleCompare}>
            Compare selected
            <Icon.ArrowRight size={16} />
          </button>
        </div>
      )}
    </div>
  );
}
