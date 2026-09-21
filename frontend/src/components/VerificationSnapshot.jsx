import { Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Icon } from './icons';
import './VerificationSnapshot.css';

// Same four metrics settled on during the resume-accuracy pass: the ones
// that are real, verified, and hold up to "so what does that actually mean"
// - not just whichever numbers happened to be highest.
const FEATURED = [
  { section: 'retrieval', key: 'hit@4', label: 'hit@4', description: 'expected evidence in top 4' },
  { section: 'grounding', key: 'citation_validity', label: 'citations valid', description: 'point at sources actually retrieved' },
  { section: 'abstention', key: 'abstention_rate', label: 'abstained correctly', description: 'declined unanswerable questions' },
  { section: 'correctness', key: 'answer_correctness', label: 'answer correctness', description: 'judged to match the reference' },
];

function pct(metric) {
  if (metric === undefined || metric === null) return '—';
  return `${Math.round(metric * 100)}%`;
}

export function VerificationSnapshot() {
  const { apiKey } = useAuth();
  const [summary, setSummary] = useState(null);
  const [notReady, setNotReady] = useState(false);

  useEffect(() => {
    let alive = true;
    api
      .evaluation(apiKey)
      .then((data) => alive && setSummary(data.summary))
      .catch(() => alive && setNotReady(true));
    return () => {
      alive = false;
    };
  }, [apiKey]);

  return (
    <section className="verify-snapshot" aria-label="Verification snapshot">
      <h2 className="dashboard__section-title">Verification snapshot</h2>

      {notReady && (
        <p className="dashboard__notice">
          No verification ledger yet. Run an expedition first — it generates one automatically.
        </p>
      )}

      {!notReady && !summary && <p className="dashboard__notice">Reading the ledger…</p>}

      {summary && (
        <dl className="verify-snapshot__row">
          {FEATURED.map(({ section, key, label, description }) => (
            <div className="verify-snapshot__stat" key={key}>
              <dt className="mono">{label}</dt>
              <dd className="verify-snapshot__value mono">{pct(summary[section]?.[key]?.value)}</dd>
              <dd className="verify-snapshot__description">{description}</dd>
            </div>
          ))}
        </dl>
      )}

      <Link to="/evaluation" className="verify-snapshot__link">
        <Icon.Stamp size={14} />
        full verification report
        <Icon.ArrowRight size={14} />
      </Link>
    </section>
  );
}
