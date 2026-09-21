import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, ApiError, describeError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Masthead } from '../components/Masthead';
import { VerificationSnapshot } from '../components/VerificationSnapshot';
import { ExpeditionForm, ExpeditionLog } from '../components/ExpeditionPanel';
import { Icon } from '../components/icons';
import './Dashboard.css';

const POLL_MS = 2500;

function jobStorageKey(userId) {
  return `specimen-ledger.job.${userId}`;
}

export function Dashboard() {
  const { userId, apiKey } = useAuth();

  const [mode, setMode] = useState('cabinet'); // 'cabinet' | 'form' | 'log'
  const [job, setJob] = useState(null);
  const [queued, setQueued] = useState(null); // { topic, numPapers } waiting for the active job to finish
  const [startError, setStartError] = useState(null);
  const [starting, setStarting] = useState(false);

  const [collections, setCollections] = useState(null);
  const [papers, setPapers] = useState(null);
  const [corpusError, setCorpusError] = useState(false);

  const pollRef = useRef(null);

  const loadCabinet = useCallback(() => {
    api
      .listPapers(apiKey)
      .then((data) => setPapers(data.papers))
      .catch((err) => {
        if (err instanceof ApiError && err.status === 400) setPapers([]); // no corpus yet
        else setCorpusError(true);
      });
    api
      .listCollections(apiKey)
      .then((data) => setCollections(data.collections))
      .catch(() => setCorpusError(true));
  }, [apiKey]);

  // Resume an in-flight (or last) expedition after a refresh - the backend
  // persists job state across restarts, so the page should too.
  useEffect(() => {
    loadCabinet();
    const savedJobId = localStorage.getItem(jobStorageKey(userId));
    if (savedJobId) {
      api
        .getIngestJob(apiKey, savedJobId)
        .then((data) => {
          setJob(data);
          // Only pull the user into the full log view for a job that still
          // needs attention; a finished one just populates the Field Log
          // panel on the cabinet.
          if (data.status === 'running') setMode('log');
        })
        .catch(() => localStorage.removeItem(jobStorageKey(userId)));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (mode !== 'log' || !job || job.status !== 'running') {
      clearInterval(pollRef.current);
      return;
    }
    pollRef.current = setInterval(() => {
      api
        .getIngestJob(apiKey, job.job_id)
        .then(setJob)
        .catch(() => clearInterval(pollRef.current));
    }, POLL_MS);
    return () => clearInterval(pollRef.current);
  }, [mode, job, apiKey]);

  // A queued expedition launches itself the moment the active one resolves.
  useEffect(() => {
    if (!queued || !job || job.status === 'running') return;
    const next = queued;
    setQueued(null);
    launchExpedition(next.topic, next.numPapers);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job, queued]);

  function launchExpedition(topic, numPapers) {
    setStarting(true);
    setStartError(null);
    api
      .startIngest(apiKey, topic, numPapers)
      .then((newJob) => {
        setJob(newJob);
        localStorage.setItem(jobStorageKey(userId), newJob.job_id);
        setMode('log');
      })
      .catch((err) => setStartError(describeError(err, 'Could not reach the server.')))
      .finally(() => setStarting(false));
  }

  function handleSubmitExpedition(topic, numPapers) {
    if (job && job.status === 'running') {
      setQueued({ topic, numPapers });
      setMode('log');
      return;
    }
    launchExpedition(topic, numPapers);
  }

  function handleReturnToCabinet() {
    // job is kept (not cleared) - the Field Log panel on the cabinet shows
    // the last expedition, running or not, until a new one replaces it.
    setMode('cabinet');
    loadCabinet();
  }

  function handleRetry() {
    setMode('form');
  }

  const hasCorpus = papers && papers.length > 0;

  return (
    <div className="dashboard">
      <Masthead />

      <p className="dashboard__intro">A specimen is a paper in your corpus.</p>

      <div className="dashboard__ledger-line mono">
        {papers ? (
          <span>
            COLLECTION — {papers.length} specimen{papers.length === 1 ? '' : 's'} catalogued across{' '}
            {collections ? collections.length : '…'} comparison{collections?.length === 1 ? '' : 's'}
          </span>
        ) : (
          <span>Reading the ledger…</span>
        )}
        {mode === 'cabinet' && (
          <button type="button" className="dashboard__begin" onClick={() => setMode('form')}>
            <Icon.Compass size={15} />
            Begin an expedition
          </button>
        )}
      </div>

      {mode === 'cabinet' && <VerificationSnapshot />}

      <main className="dashboard__body">
        {mode === 'cabinet' && (
          <>
            <section className="dashboard__drawers" aria-label="Compare specimens">
              <h2 className="dashboard__section-title">Compare specimens</h2>
              {corpusError && <p className="dashboard__notice">The ledger could not be reached.</p>}
              {!corpusError && collections === null && <p className="dashboard__notice">Reading the ledger…</p>}
              {collections && collections.length === 0 && (
                <p className="dashboard__notice">
                  {hasCorpus ? (
                    <>
                      {papers.length} specimen{papers.length === 1 ? ' is' : 's are'} catalogued and ready.{' '}
                      <Link to="/corpus" className="dashboard__corpus-link">
                        Select some from the corpus
                      </Link>{' '}
                      to pose a research question and open a comparison.
                    </>
                  ) : (
                    'No specimens catalogued yet. An expedition must return before a comparison can be made.'
                  )}
                </p>
              )}
              <ol className="dashboard__drawer-list">
                {collections?.map((c) => (
                  <li key={c.collection_id} className="dashboard__drawer">
                    <Link to={`/collections/${c.collection_id}`} className="dashboard__drawer-link">
                      <span className="dashboard__drawer-question">{c.question}</span>
                      <span className="dashboard__drawer-meta mono">
                        {c.papers.length} specimen{c.papers.length === 1 ? '' : 's'}
                        {c.comparison ? ' · compared' : ''}
                        {c.followups ? ' · follow-ups drafted' : ''}
                      </span>
                    </Link>
                  </li>
                ))}
              </ol>
            </section>

            <section className="dashboard__field-log" aria-label="Field log">
              <h2 className="dashboard__section-title">Field Log</h2>
              {job ? (
                <div className="dashboard__last-expedition">
                  <p className="dashboard__last-topic">{job.topic}</p>
                  <p className="dashboard__last-status mono">{job.message}</p>
                  <button type="button" className="dashboard__reopen" onClick={() => setMode('log')}>
                    Reopen expedition log
                    <Icon.ArrowRight size={14} />
                  </button>
                </div>
              ) : (
                <p className="dashboard__notice">No expeditions logged.</p>
              )}
            </section>
          </>
        )}

        {mode === 'form' && (
          <section className="ledger-panel dashboard__panel">
            <h2 className="dashboard__panel-title">New Expedition</h2>
            <ExpeditionForm
              onSubmit={handleSubmitExpedition}
              onCancel={() => setMode(job ? 'log' : 'cabinet')}
              busy={starting}
              error={startError}
              hasQueuedJob={job && job.status === 'running'}
            />
          </section>
        )}

        {mode === 'log' && job && (
          <section className="ledger-panel dashboard__panel">
            <h2 className="dashboard__panel-title">
              Expedition{queued ? ` — next: ${queued.topic}` : ''}
            </h2>
            <ExpeditionLog job={job} onReturn={handleReturnToCabinet} onRetry={handleRetry} />
          </section>
        )}
      </main>
    </div>
  );
}
