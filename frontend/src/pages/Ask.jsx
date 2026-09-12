import { useEffect, useMemo, useState } from 'react';
import { api, describeError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Masthead } from '../components/Masthead';
import { Icon } from '../components/icons';
import './Ask.css';

export function Ask() {
  const { apiKey } = useAuth();
  const [papers, setPapers] = useState(null);
  const [topic, setTopic] = useState('');
  const [query, setQuery] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [showEvidence, setShowEvidence] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    api
      .listPapers(apiKey)
      .then((data) => setPapers(data.papers))
      .catch(() => setPapers([]));
  }, [apiKey]);

  const topics = useMemo(() => {
    const seen = new Set();
    for (const p of papers || []) seen.add(p.topic || 'uncategorized');
    return [...seen];
  }, [papers]);

  function handleSubmit(event) {
    event.preventDefault();
    if (!query.trim() || !topic) return;
    setBusy(true);
    setError(null);
    setShowEvidence(false);
    api
      .ask(apiKey, query.trim(), topic)
      .then((data) => {
        setAnswer(data);
        setHistory((prev) => [{ query: data.query, topic, abstained: data.abstained }, ...prev].slice(0, 8));
      })
      .catch((err) => setError(describeError(err, 'Could not reach the server.')))
      .finally(() => setBusy(false));
  }

  return (
    <div className="ask">
      <Masthead />
      <p className="ask__intro">
        A specimen is a paper in your corpus. Questions are scoped to one expedition at a time - the corpus
        never mixes topics when answering.
      </p>

      <form className="ask__form" onSubmit={handleSubmit}>
        <select
          className="ask__topic mono"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        >
          <option value="" disabled>
            {papers === null ? 'loading topics…' : 'choose a topic…'}
          </option>
          {topics.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <input
          className="ask__input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about this topic…"
        />
        <button type="submit" className="ask__submit" disabled={busy || !topic}>
          {busy ? <Icon.Loading className="spin" size={16} /> : <Icon.ArrowRight size={16} />}
          {busy ? 'Asking…' : 'Ask'}
        </button>
      </form>

      {papers && papers.length === 0 && (
        <p className="notice ask__error">No specimens catalogued yet - begin an expedition first.</p>
      )}

      {error && <p className="notice ask__error">{error}</p>}

      {answer && (
        <div className="ledger-panel ask__answer">
          <p className="ask__query">
            {answer.query} <span className="mono ask__query-topic">— {topic}</span>
          </p>

          {answer.abstained && (
            <p className="ask__abstained mono">
              <Icon.Warning size={14} /> declined — not enough evidence in this topic's specimens
            </p>
          )}

          <p className="ask__answer-text">{answer.answer}</p>

          {answer.sources?.length > 0 && (
            <div className="ask__sources">
              <h2 className="section-title">Sources</h2>
              <ul className="ruled-list">
                {answer.sources.map((s, i) => (
                  <li key={i} className="ask__source">
                    <span>{s.title}</span>
                    <span className="mono ask__source-page">p.{s.page_number}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button
            type="button"
            className="ask__evidence-toggle"
            onClick={() => setShowEvidence((v) => !v)}
          >
            {showEvidence ? 'Hide' : 'Inspect'} evidence
          </button>

          {showEvidence && (
            <dl className="ask__evidence">
              <div className="compare-field">
                <dt className="mono">Citations</dt>
                <dd>
                  {answer.evidence.citations.valid ? 'valid' : 'invalid'}
                  {answer.evidence.citations.invalid_ids?.length > 0 &&
                    ` — unrecognized: ${answer.evidence.citations.invalid_ids.join(', ')}`}
                </dd>
              </div>
              <div className="compare-field">
                <dt className="mono">Claims with citations</dt>
                <dd>{answer.evidence.claims_with_citations}</dd>
              </div>
              <div className="compare-field">
                <dt className="mono">Claims assessed for support</dt>
                <dd>{answer.evidence.claims_assessed_for_support}</dd>
              </div>
              <div className="compare-field">
                <dt className="mono">Not applicable</dt>
                <dd>{answer.evidence.claims_not_applicable} (no checkable fact)</dd>
              </div>
              {answer.evidence.ungrounded_numbers?.length > 0 && (
                <div className="compare-field">
                  <dt className="mono">Ungrounded numbers</dt>
                  <dd>{answer.evidence.ungrounded_numbers.join(', ')}</dd>
                </div>
              )}
              {answer.evidence.ungrounded_names?.length > 0 && (
                <div className="compare-field">
                  <dt className="mono">Ungrounded names</dt>
                  <dd>{answer.evidence.ungrounded_names.join(', ')}</dd>
                </div>
              )}
              <p className="ask__evidence-note">{answer.evidence.note}</p>
            </dl>
          )}
        </div>
      )}

      {history.length > 0 && (
        <div className="ask__history">
          <h2 className="section-title">Asked this session</h2>
          <ul className="ruled-list">
            {history.map((h, i) => (
              <li key={i} className="ask__history-row">
                <span>
                  {h.query} <span className="mono ask__history-topic">({h.topic})</span>
                </span>
                {h.abstained && <span className="mono ask__history-flag">declined</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
