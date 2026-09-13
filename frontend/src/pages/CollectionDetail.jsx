import { Fragment, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { api, describeError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Masthead } from '../components/Masthead';
import { Icon } from '../components/icons';
import './CollectionDetail.css';

const FOLLOWUP_LABEL = {
  author_proposed: 'Author-proposed',
  unresolved_in_selection: 'Unresolved in this selection',
  assistant_proposed: 'Assistant-proposed',
};

function truncateTitle(title, max = 26) {
  return title.length > max ? `${title.slice(0, max)}…` : title;
}

// One entry per cited paper (not per page), so a paper cited on several pages
// reads as "Title — pp. 9, 15" instead of repeating its truncated title once
// per page. The full, untruncated title rides along as a native tooltip.
function groupEvidence(evidence, titleByPaperId) {
  const byTitle = new Map();
  for (const e of evidence) {
    const fullTitle = titleByPaperId[e.paper_id] || e.paper_id;
    if (!byTitle.has(fullTitle)) byTitle.set(fullTitle, []);
    byTitle.get(fullTitle).push(e.page_number);
  }
  return Array.from(byTitle, ([fullTitle, pages]) => ({
    fullTitle,
    truncated: truncateTitle(fullTitle),
    pages,
  }));
}

function Evidence({ items, titleByPaperId, prefix, className }) {
  return (
    <p className={`${className} mono`}>
      {prefix}
      {groupEvidence(items, titleByPaperId).map((g, i) => (
        <span key={g.fullTitle} title={g.fullTitle}>
          {i > 0 && '   ·   '}
          {g.truncated} — {g.pages.length > 1 ? 'pp.' : 'p.'} {g.pages.join(', ')}
        </span>
      ))}
    </p>
  );
}

function CompareField({ label, value }) {
  const flat = value === 'unclear' || value === 'not applicable';
  return (
    <div className="compare-field">
      <dt className="mono">{label}</dt>
      <dd className={flat ? 'compare-field__flat' : ''}>{value}</dd>
    </div>
  );
}

function AddSpecimen({ apiKey, collectionId, existingIds, topic, onAdded }) {
  const [open, setOpen] = useState(false);
  const [papers, setPapers] = useState(null);
  const [paperId, setPaperId] = useState('');
  const [why, setWhy] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  function startOpen() {
    setOpen(true);
    if (!papers) {
      api
        .listPapers(apiKey)
        .then((data) => setPapers(data.papers))
        .catch(() => setError('Could not read the corpus.'));
    }
  }

  function handleAdd(event) {
    event.preventDefault();
    if (!paperId || !why.trim()) {
      setError('Pick a specimen and say why it belongs.');
      return;
    }
    setBusy(true);
    setError(null);
    api
      .addPaperToCollection(apiKey, collectionId, paperId, why.trim())
      .then((updated) => {
        onAdded(updated);
        setOpen(false);
        setPaperId('');
        setWhy('');
      })
      .catch((err) => setError(describeError(err, 'Could not add that specimen.')))
      .finally(() => setBusy(false));
  }

  const available = (papers || []).filter(
    (p) => !existingIds.has(p.paper_id) && (!topic || p.topic === topic)
  );

  if (!open) {
    return (
      <button type="button" className="collection-detail__add-toggle" onClick={startOpen}>
        + add a specimen
      </button>
    );
  }

  return (
    <form className="add-specimen" onSubmit={handleAdd}>
      {papers === null && <p className="notice">Reading the corpus…</p>}
      {papers && available.length === 0 && (
        <p className="notice">Every specimen from this expedition is already here.</p>
      )}
      {available.length > 0 && (
        <>
          <label className="add-specimen__field">
            <span>Specimen</span>
            <select value={paperId} onChange={(e) => setPaperId(e.target.value)}>
              <option value="" disabled>
                choose one…
              </option>
              {available.map((p) => (
                <option key={p.paper_id} value={p.paper_id}>
                  {p.title}
                </option>
              ))}
            </select>
          </label>
          <label className="add-specimen__field">
            <span>Why it belongs</span>
            <input value={why} onChange={(e) => setWhy(e.target.value)} placeholder="reason" />
          </label>
          {error && <p className="collection-detail__error">{error}</p>}
          <div className="add-specimen__actions">
            <button type="submit" className="collection-detail__add-toggle" disabled={busy}>
              {busy ? 'Adding…' : 'Add'}
            </button>
            <button type="button" className="add-specimen__cancel" onClick={() => setOpen(false)}>
              cancel
            </button>
          </div>
        </>
      )}
    </form>
  );
}

export function CollectionDetail() {
  const { id } = useParams();
  const { apiKey } = useAuth();

  const [collection, setCollection] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [drafting, setDrafting] = useState(false);
  const [actionError, setActionError] = useState(null);
  const [removingId, setRemovingId] = useState(null);

  useEffect(() => {
    api
      .getCollection(apiKey, id)
      .then(setCollection)
      .catch(() => setLoadError('This comparison could not be found.'));
  }, [apiKey, id]);

  function handleCompare() {
    setComparing(true);
    setActionError(null);
    api
      .compareCollection(apiKey, id)
      .then(setCollection)
      .catch((err) => setActionError(describeError(err, 'The comparison failed.')))
      .finally(() => setComparing(false));
  }

  function handleFollowups() {
    setDrafting(true);
    setActionError(null);
    api
      .followupsForCollection(apiKey, id)
      .then(setCollection)
      .catch((err) => setActionError(describeError(err, 'Drafting follow-ups failed.')))
      .finally(() => setDrafting(false));
  }

  function handleRemove(paperId) {
    setRemovingId(paperId);
    setActionError(null);
    api
      .removePaperFromCollection(apiKey, id, paperId)
      .then(setCollection)
      .catch((err) => setActionError(describeError(err, 'Could not remove that specimen.')))
      .finally(() => setRemovingId(null));
  }

  if (loadError) {
    return (
      <div className="collection-detail">
        <Masthead />
        <p className="notice collection-detail__notice">{loadError}</p>
      </div>
    );
  }

  if (!collection) {
    return (
      <div className="collection-detail">
        <Masthead />
        <p className="notice collection-detail__notice">Reading the ledger…</p>
      </div>
    );
  }

  const existingIds = new Set(collection.papers.map((p) => p.paper_id));
  const collectionTopic = collection.papers[0]?.topic;
  const titleByPaperId = Object.fromEntries(collection.papers.map((p) => [p.paper_id, p.title]));

  return (
    <div className="collection-detail">
      <Masthead />

      <div className="collection-detail__header">
        <h1 className="collection-detail__question">{collection.question}</h1>
        <p className="collection-detail__lede">
          {collection.papers.length} specimen{collection.papers.length === 1 ? '' : 's'} in this comparison.
          {collection.comparison && collection.papers.length > 0 && (
            <span className="collection-detail__stale-note">
              {' '}
              Adding or removing a specimen clears the comparison below until you re-run it.
            </span>
          )}
        </p>
      </div>

      <section className="collection-detail__papers">
        <h2 className="section-title">Specimens</h2>
        <ol className="ruled-list">
          {collection.papers.map((p) => (
            <li key={p.paper_id} className="collection-detail__paper-row">
              <div>
                <p className="collection-detail__paper-title">{p.title}</p>
                <p className="collection-detail__paper-why mono">{p.why_included}</p>
              </div>
              <button
                type="button"
                className="collection-detail__remove"
                disabled={removingId === p.paper_id}
                onClick={() => handleRemove(p.paper_id)}
              >
                {removingId === p.paper_id ? 'removing…' : 'remove'}
              </button>
            </li>
          ))}
        </ol>
        <AddSpecimen
          apiKey={apiKey}
          collectionId={id}
          existingIds={existingIds}
          topic={collectionTopic}
          onAdded={setCollection}
        />
      </section>

      {actionError && <p className="collection-detail__error">{actionError}</p>}

      <section className="collection-detail__actions">
        <button type="button" className="collection-detail__action" onClick={handleCompare} disabled={comparing}>
          {comparing ? <Icon.Loading className="spin" size={16} /> : <Icon.Stamp size={16} />}
          {comparing ? 'Comparing…' : collection.comparison ? 'Re-run comparison' : 'Run comparison'}
        </button>
        <button
          type="button"
          className="collection-detail__action"
          onClick={handleFollowups}
          disabled={drafting || !collection.comparison}
          title={!collection.comparison ? 'Run a comparison first' : undefined}
        >
          {drafting ? <Icon.Loading className="spin" size={16} /> : <Icon.Compass size={16} />}
          {drafting ? 'Drafting…' : 'Suggest follow-ups'}
        </button>
      </section>

      {collection.comparison && (
        <section className="collection-detail__comparison">
          <h2 className="section-title">What each specimen establishes</h2>
          {collection.comparison.map((row) => (
            <div key={row.paper_id} className="ledger-panel compare-record">
              <h3 className="compare-record__title">{row.title}</h3>
              <dl className="compare-record__fields">
                <CompareField label="Approach" value={row.approach} />
                <CompareField label="Evaluation setting" value={row.evaluation_setting} />
                <CompareField label="Main finding" value={row.main_finding} />
                <CompareField label="Author-stated limitation" value={row.author_limitation} />
              </dl>
              {row.supporting_pages?.length > 0 && (
                <p className="compare-record__pages mono">pages: {row.supporting_pages.join(', ')}</p>
              )}
            </div>
          ))}
        </section>
      )}

      {collection.contradictions &&
        (collection.contradictions.contradictions.length > 0 || collection.contradictions.gaps.length > 0) && (
          <section className="collection-detail__contradictions">
            <h2 className="section-title">Contradictions and gaps</h2>

            {collection.contradictions.contradictions.length > 0 && (
              <div className="finding-group">
                <p className="finding-group__label mono">Where the selection disagrees</p>
                <ol className="ruled-list">
                  {collection.contradictions.contradictions.map((c, i) => (
                    <li key={`c${i}`} className="contradiction">
                      <p className="contradiction__papers">
                        {c.paper_ids.map((pid, j) => (
                          <Fragment key={pid}>
                            {j > 0 && <span className="contradiction__versus mono">contradicts</span>}
                            <span className="contradiction__paper" title={titleByPaperId[pid] || pid}>
                              {titleByPaperId[pid] || pid}
                            </span>
                          </Fragment>
                        ))}
                      </p>
                      <p className="contradiction__description">{c.description}</p>
                      {c.evidence?.length > 0 && (
                        <Evidence
                          items={c.evidence}
                          titleByPaperId={titleByPaperId}
                          prefix=""
                          className="contradiction__evidence"
                        />
                      )}
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {collection.contradictions.gaps.length > 0 && (
              <div className="finding-group">
                <p className="finding-group__label mono">Left unanswered by this selection</p>
                <ul className="gap-list">
                  {collection.contradictions.gaps.map((g, i) => (
                    <li key={`g${i}`} className="gap-list__item">
                      {g.description}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}

      {collection.followups && (
        <section className="collection-detail__followups">
          <h2 className="section-title">Possible next studies</h2>
          <ol className="ruled-list">
            {collection.followups.map((f, i) => (
              <li key={i} className="followup">
                <span className={`followup__type followup__type--${f.type} mono`}>
                  {FOLLOWUP_LABEL[f.type] || f.type}
                </span>
                <p className="followup__title">{f.possible_followup}</p>
                <p className="followup__field">
                  <strong>Why investigate:</strong> {f.why_investigate}
                </p>
                <p className="followup__field">
                  <strong>Starting experiment:</strong> {f.starting_experiment}
                </p>
                <p className="followup__field">
                  <strong>Still to check:</strong> {f.still_to_check}
                </p>
                {f.evidence?.length > 0 && (
                  <Evidence
                    items={f.evidence}
                    titleByPaperId={titleByPaperId}
                    prefix="evidence: "
                    className="followup__evidence"
                  />
                )}
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}
