const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `Request failed (${status})`);
    this.status = status;
    this.detail = detail;
  }
}

// One place to turn a caught error into user-facing text, so a 429 reads the
// same way (and explains itself) everywhere it can happen, instead of each
// page inventing its own generic fallback that hides the real reason.
export function describeError(err, fallback) {
  if (err instanceof ApiError) {
    if (err.status === 429) {
      return `${err.detail} This daily cap exists to protect your account if the key is ever exposed or shared.`;
    }
    return err.detail || fallback;
  }
  return fallback;
}

async function request(path, { method = 'GET', apiKey, body } = {}) {
  const headers = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (apiKey) headers['X-API-Key'] = apiKey;

  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = response.statusText;
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) return null;
  return response.json();
}

export const api = {
  register: (email) => request('/users', { method: 'POST', body: { email: email || null } }),
  whoami: (apiKey) => request('/me', { apiKey }),

  getSettings: (apiKey) => request('/settings', { apiKey }),
  updateSettings: (apiKey, fields) => request('/settings', { method: 'PUT', apiKey, body: fields }),

  listPapers: (apiKey) => request('/papers', { apiKey }),
  search: (apiKey, query, topic) => request('/search', { method: 'POST', apiKey, body: { query, topic } }),
  ask: (apiKey, query, topic) => request('/ask', { method: 'POST', apiKey, body: { query, topic } }),

  startIngest: (apiKey, topic, numPapers) =>
    request('/ingest', { method: 'POST', apiKey, body: { topic, num_papers: numPapers } }),
  getIngestJob: (apiKey, jobId) => request(`/ingest/${jobId}`, { apiKey }),
  listIngestJobs: (apiKey) => request('/ingest', { apiKey }),

  listCollections: (apiKey) => request('/collections', { apiKey }),
  getCollection: (apiKey, id) => request(`/collections/${id}`, { apiKey }),
  createCollection: (apiKey, question, papers) =>
    request('/collections', { method: 'POST', apiKey, body: { question, papers } }),
  compareCollection: (apiKey, id) => request(`/collections/${id}/compare`, { method: 'POST', apiKey }),
  followupsForCollection: (apiKey, id) => request(`/collections/${id}/followups`, { method: 'POST', apiKey }),
  addPaperToCollection: (apiKey, id, paperId, whyIncluded) =>
    request(`/collections/${id}/papers`, {
      method: 'POST',
      apiKey,
      body: { paper_id: paperId, why_included: whyIncluded },
    }),
  removePaperFromCollection: (apiKey, id, paperId) =>
    request(`/collections/${id}/papers/${paperId}`, { method: 'DELETE', apiKey }),

  // Per-user and dynamic now - reflects your own corpus, not a shared benchmark.
  evaluation: (apiKey) => request('/evaluation', { apiKey }),
};
