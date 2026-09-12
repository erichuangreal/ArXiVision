import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { api, ApiError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Masthead } from '../components/Masthead';
import { Icon } from '../components/icons';
import './CreateCollection.css';

export function CreateCollection() {
  const { apiKey } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const papers = location.state?.papers || [];

  const [question, setQuestion] = useState('');
  const [reasons, setReasons] = useState(() => Object.fromEntries(papers.map((p) => [p.paper_id, ''])));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  if (papers.length === 0) {
    return (
      <div className="create-collection">
        <Masthead />
        <div className="create-collection__body">
          <p className="notice">
            No specimens were carried over.{' '}
            <Link to="/corpus" className="create-collection__link">
              Select some from the corpus
            </Link>{' '}
            first.
          </p>
        </div>
      </div>
    );
  }

  function handleSubmit(event) {
    event.preventDefault();
    if (!question.trim()) {
      setError('State the research question this comparison is for.');
      return;
    }
    setBusy(true);
    setError(null);
    api
      .createCollection(
        apiKey,
        question.trim(),
        papers.map((p) => ({ paper_id: p.paper_id, why_included: reasons[p.paper_id]?.trim() || 'Selected from the corpus.' }))
      )
      .then((collection) => navigate(`/collections/${collection.collection_id}`))
      .catch((err) => setError(err instanceof ApiError ? err.detail : 'Could not reach the server.'))
      .finally(() => setBusy(false));
  }

  return (
    <div className="create-collection">
      <Masthead />
      <div className="create-collection__body">
        <div className="ledger-panel create-collection__panel">
          <h1 className="create-collection__title">Pose a research question</h1>
          <p className="create-collection__lede">
            A comparison opens once a question is posed of the specimens you selected.
          </p>

          <form onSubmit={handleSubmit}>
            <label className="create-collection__field">
              <span>Research question</span>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. Under what conditions does this method fail?"
                rows={2}
                autoFocus
              />
            </label>

            <h2 className="section-title">Specimens in this comparison</h2>
            <ol className="ruled-list create-collection__papers">
              {papers.map((p) => (
                <li key={p.paper_id} className="create-collection__paper">
                  <p className="create-collection__paper-title">{p.title}</p>
                  <label className="create-collection__field">
                    <span>Why this one belongs here</span>
                    <input
                      value={reasons[p.paper_id] || ''}
                      onChange={(e) => setReasons((prev) => ({ ...prev, [p.paper_id]: e.target.value }))}
                      placeholder="e.g. directly addresses the failure mode in question"
                    />
                  </label>
                </li>
              ))}
            </ol>

            {error && <p className="create-collection__error">{error}</p>}

            <div className="create-collection__actions">
              <button type="submit" className="create-collection__submit" disabled={busy}>
                {busy ? <Icon.Loading className="spin" size={16} /> : <Icon.Stamp size={16} />}
                {busy ? 'Filing…' : 'Open this comparison'}
              </button>
              <Link to="/corpus" className="create-collection__cancel">
                Back to the corpus
              </Link>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
