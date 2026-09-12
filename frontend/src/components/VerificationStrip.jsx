import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Icon } from './icons';
import './VerificationStrip.css';

function pct(metric) {
  if (!metric) return '—';
  return `${Math.round(metric.value * 100)}%`;
}

export function VerificationStrip() {
  const { apiKey } = useAuth();
  const [summary, setSummary] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    api
      .evaluation(apiKey)
      .then((data) => alive && setSummary(data.summary))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [apiKey]);

  if (failed) return null; // no verification ledger yet (no expedition has completed); say nothing rather than fake a number

  return (
    <div className="verify-strip">
      <Icon.Stamp size={16} className="verify-strip__mark" />
      <span className="verify-strip__label">verified against your own corpus</span>
      {summary ? (
        <span className="verify-strip__figures mono">
          retrieval {pct(summary.retrieval?.['hit@1'])} first-try
          <span className="verify-strip__sep">·</span>
          claims supported {pct(summary.grounding?.claim_support)}
          <span className="verify-strip__sep">·</span>
          abstained correctly {pct(summary.abstention?.abstention_rate)}
        </span>
      ) : (
        <span className="verify-strip__figures mono">reading the ledger…</span>
      )}
      <Link to="/evaluation" className="verify-strip__link">
        full report
        <Icon.ArrowRight size={14} />
      </Link>
    </div>
  );
}
