import { useEffect, useRef, useState } from 'react';
import { Icon } from './icons';
import './ExpeditionPanel.css';

const STAGE_LABEL = {
  queued: 'Queued',
  searching_arxiv: 'Surveying arXiv',
  downloading: 'Collecting specimens',
  extracting: 'Pressing (extracting text)',
  indexing: 'Cataloguing (embedding)',
  evaluating: 'Verifying',
  ready: 'Filed',
};

function glyphFor(stage, status) {
  if (status === 'failed') return <Icon.Warning size={14} className="expedition-log__glyph expedition-log__glyph--fail" />;
  if (stage === 'ready') return <Icon.Check size={14} className="expedition-log__glyph expedition-log__glyph--done" />;
  return <Icon.Loading size={14} className="expedition-log__glyph expedition-log__glyph--running spin" />;
}

function timestamp(epochSeconds) {
  if (!epochSeconds) return '';
  return new Date(epochSeconds * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function ExpeditionForm({ onSubmit, onCancel, busy, error, hasQueuedJob }) {
  const [topic, setTopic] = useState('');
  const [numPapers, setNumPapers] = useState(10);
  const [validation, setValidation] = useState(null);

  function handleSubmit(event) {
    event.preventDefault();
    if (!topic.trim()) {
      setValidation('An expedition needs a subject.');
      return;
    }
    if (numPapers < 1 || numPapers > 100) {
      setValidation('Between 1 and 100 specimens per expedition.');
      return;
    }
    setValidation(null);
    onSubmit(topic.trim(), numPapers);
  }

  return (
    <form className="expedition-form" onSubmit={handleSubmit}>
      <p className="expedition-form__lede">
        {hasQueuedJob
          ? 'An expedition is already underway. This one will be queued to begin once it returns.'
          : 'Name the subject, and how many specimens to bring back.'}
      </p>

      <label className="expedition-form__field">
        <span>Subject of inquiry</span>
        <input
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. mechanistic interpretability"
          autoFocus
        />
      </label>

      <label className="expedition-form__field expedition-form__field--narrow">
        <span>Specimens to collect</span>
        <input
          type="number"
          min={1}
          max={100}
          value={numPapers}
          onChange={(e) => setNumPapers(Number(e.target.value))}
        />
      </label>

      {(validation || error) && <p className="expedition-form__error">{validation || error}</p>}

      <p className="expedition-form__expectation">
        Takes longer for more specimens - downloads are paced a few seconds apart, politely, so arXiv never sees a
        burst of requests. Capped at 20 expeditions a day per account.
      </p>

      <div className="expedition-form__actions">
        <button type="submit" className="expedition-form__submit" disabled={busy}>
          {busy ? <Icon.Loading className="spin" size={16} /> : <Icon.Compass size={16} />}
          {hasQueuedJob ? 'Queue expedition' : 'Launch expedition'}
        </button>
        <button type="button" className="expedition-form__cancel" onClick={onCancel}>
          Return to the cabinet
        </button>
      </div>
    </form>
  );
}

export function ExpeditionLog({ job, onReturn, onRetry }) {
  const [lines, setLines] = useState([]);
  const lastMessage = useRef(null);
  const logEndRef = useRef(null);

  useEffect(() => {
    if (!job || job.message === lastMessage.current) return;
    lastMessage.current = job.message;
    setLines((prev) => [
      ...prev,
      { stage: job.stage, status: job.status, message: job.message, at: job.updated_at },
    ]);
  }, [job]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [lines]);

  if (!job) return null;

  const progressLabel =
    job.total > 0 && job.stage !== 'ready' && job.stage !== 'queued'
      ? `${job.current} / ${job.total}`
      : null;

  return (
    <div className="expedition-log">
      <div className="expedition-log__header">
        <h2 className="expedition-log__topic">{job.topic}</h2>
        <span className="expedition-log__stage mono">
          {job.status === 'failed' ? 'Failed while: ' : ''}
          {STAGE_LABEL[job.stage] || job.stage}
          {progressLabel && <span className="expedition-log__progress"> · {progressLabel}</span>}
        </span>
      </div>

      <ol className="expedition-log__lines">
        {lines.map((line, i) => (
          <li key={i} className="expedition-log__line">
            {glyphFor(line.stage, line.status)}
            <span className="expedition-log__message">{line.message}</span>
            <span className="expedition-log__time mono">{timestamp(line.at)}</span>
          </li>
        ))}
        <li ref={logEndRef} />
      </ol>

      {job.status === 'ready' && (
        <button type="button" className="expedition-log__done" onClick={onReturn}>
          <Icon.Archive size={16} />
          View the cabinet
        </button>
      )}

      {job.status === 'failed' && (
        <div className="expedition-log__failed">
          <p>{job.error || 'The expedition was interrupted.'}</p>
          <div className="expedition-log__actions">
            <button type="button" className="expedition-form__submit" onClick={onRetry}>
              Try again
            </button>
            <button type="button" className="expedition-form__cancel" onClick={onReturn}>
              Return to the cabinet
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
