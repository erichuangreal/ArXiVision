import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, ApiError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Masthead } from '../components/Masthead';
import { Icon } from '../components/icons';
import './Evaluation.css';

const SECTION_TITLE = {
  retrieval: 'Retrieval',
  grounding: 'Grounding',
  correctness: 'Correctness',
  abstention: 'Abstention',
};

function formatValue(name, metric) {
  if (name === 'claims_not_applicable') return metric.value;
  if (typeof metric.value === 'number' && metric.value <= 1) return `${(metric.value * 100).toFixed(1)}%`;
  return metric.value;
}

// Only color a metric when it's actually notable, in the direction that
// metric is defined to be good or bad in (e.g. a LOW false_refusal_rate is
// the good outcome) - never a blanket "verified" green regardless of value.
function metricTone(metric) {
  if (typeof metric.value !== 'number' || metric.higher_is_better === null || metric.higher_is_better === undefined) {
    return null;
  }
  const score = metric.higher_is_better ? metric.value : 1 - metric.value;
  if (score >= 0.9) return 'verified';
  if (score < 0.5) return 'caution';
  return null;
}

export function Evaluation() {
  const { apiKey } = useAuth();
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    api
      .evaluation(apiKey)
      .then(setReport)
      .catch((err) =>
        setError(
          err instanceof ApiError && err.status === 404
            ? 'No verification ledger yet. Run an expedition first, it generates one automatically.'
            : 'Could not reach the server.'
        )
      );
  }, [apiKey]);

  return (
    <div className="evaluation">
      <Masthead />
      <div className="evaluation__intro">
        <Link to="/" className="evaluation__back">
          <Icon.ArrowLeft size={14} />
          Cabinet
        </Link>
        <h1 className="evaluation__title">Verification Ledger</h1>
        <p className="evaluation__lede">
          Test questions authored from your own ingested papers, then scored the same way every answer is
          checked elsewhere in this app.
        </p>
      </div>

      {error && <p className="evaluation__notice">{error}</p>}

      {report && (
        <>
          {Object.entries(report.summary).map(([section, metrics]) => (
            <section key={section} className="evaluation__section">
              <h2 className="evaluation__section-title">{SECTION_TITLE[section] || section}</h2>
              <dl className="evaluation__metrics">
                {Object.entries(metrics).map(([name, metric]) => {
                  const tone = metricTone(metric);
                  return (
                    <div key={name} className="evaluation__metric-row">
                      <dt className="mono">{name}</dt>
                      <dd
                        className={`mono evaluation__metric-value${tone ? ` evaluation__metric-value--${tone}` : ''}`}
                      >
                        {formatValue(name, metric)}
                      </dd>
                      <dd className="evaluation__metric-label">{metric.label}</dd>
                    </div>
                  );
                })}
              </dl>
            </section>
          ))}

          <section className="evaluation__section">
            <h2 className="evaluation__section-title">Test questions</h2>
            <ol className="evaluation__questions">
              {report.questions.map((q, i) => (
                <li key={i} className="evaluation__question">
                  <button
                    type="button"
                    className="evaluation__question-toggle"
                    onClick={() => setExpanded(expanded === i ? null : i)}
                  >
                    <span>
                      {q.topic && <span className="mono evaluation__question-topic">[{q.topic}]</span>}
                      {q.query}
                    </span>
                    <span className="mono evaluation__question-flags">
                      {q.answer_correct === true && 'correct'}
                      {q.answer_correct === false && 'incorrect'}
                      {q.answer_correct === undefined && 'ungraded'} · {q.citations_valid ? 'cited' : 'uncited'} · {q.claims_supported}
                    </span>
                  </button>
                  {expanded === i && (
                    <p className="evaluation__answer">
                      {q.answer}
                      {q.correctness_reason && (
                        <>
                          <br />
                          <span className="mono evaluation__question-topic">{q.correctness_reason}</span>
                        </>
                      )}
                    </p>
                  )}
                </li>
              ))}
              {report.unanswerable_questions.map((q, i) => (
                <li key={`u${i}`} className="evaluation__question">
                  <button
                    type="button"
                    className="evaluation__question-toggle"
                    onClick={() => setExpanded(expanded === `u${i}` ? null : `u${i}`)}
                  >
                    <span>
                      <span className="mono evaluation__question-topic">[out of corpus]</span>
                      {q.query}
                    </span>
                    <span className="mono evaluation__question-flags">
                      {q.abstained ? 'declined correctly' : 'answered'}
                    </span>
                  </button>
                  {expanded === `u${i}` && <p className="evaluation__answer">{q.answer}</p>}
                </li>
              ))}
            </ol>
          </section>
        </>
      )}
    </div>
  );
}
