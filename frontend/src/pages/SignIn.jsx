import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Icon } from '../components/icons';
import './SignIn.css';

export function SignIn() {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState('create'); // 'create' | 'existing'
  const [email, setEmail] = useState('');
  const [existingKey, setExistingKey] = useState('');
  const [issued, setIssued] = useState(null); // { userId, apiKey } once created
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function handleCreate(event) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await api.register(email.trim() || undefined);
      setIssued(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Could not reach the server.');
    } finally {
      setBusy(false);
    }
  }

  function handleContinue() {
    signIn(issued.user_id, issued.api_key);
    navigate('/');
  }

  async function handleExistingSubmit(event) {
    event.preventDefault();
    if (!existingKey.trim()) {
      setError('Paste the API key you saved when you created the account.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      // Only the key is actually checked - the account id is resolved from it,
      // never asked for.
      const { user_id } = await api.whoami(existingKey.trim());
      signIn(user_id, existingKey.trim());
      navigate('/');
    } catch (err) {
      setError(err instanceof ApiError ? 'That key was not recognized.' : 'Could not reach the server.');
    } finally {
      setBusy(false);
    }
  }

  async function copyKey() {
    try {
      await navigator.clipboard.writeText(issued.api_key);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <main className="signin">
      <div className="signin__card">
        <span className="signin__card-corner signin__card-corner--tl" aria-hidden="true" />
        <span className="signin__card-corner signin__card-corner--br" aria-hidden="true" />

        {issued ? (
          <div className="signin__issued">
            <h1 className="signin__title">Accession recorded</h1>
            <p className="signin__lede">
              This key is the only way back into this collection. It will not be shown again.
            </p>
            <div className="signin__key mono">
              <Icon.Stamp size={16} />
              {issued.api_key}
            </div>
            <button type="button" className="signin__copy" onClick={copyKey}>
              {copied ? 'Copied' : 'Copy key'}
            </button>
            <p className="signin__id mono">cataloguer id: {issued.user_id}</p>
            <label className="signin__confirm">
              <input type="checkbox" required onChange={(e) => setError(e.target.checked ? null : error)} />
              I have saved this key somewhere I can find it
            </label>
            <button type="button" className="signin__submit" onClick={handleContinue}>
              Enter the collection
              <Icon.ArrowRight />
            </button>
          </div>
        ) : mode === 'create' ? (
          <form onSubmit={handleCreate}>
            <h1 className="signin__title">Open a collection</h1>
            <p className="signin__lede">Each cataloguer keeps their own corpus, isolated from every other.</p>

            <label className="signin__field">
              <span>Email (optional, for your own reference)</span>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="not shared, not required"
              />
            </label>

            {error && <p className="signin__error">{error}</p>}

            <button type="submit" className="signin__submit" disabled={busy}>
              {busy ? <Icon.Loading className="spin" /> : <Icon.Compass />}
              {busy ? 'Opening…' : 'Start a new collection'}
            </button>

            <button type="button" className="signin__switch" onClick={() => setMode('existing')}>
              I already have a key
            </button>
          </form>
        ) : (
          <form onSubmit={handleExistingSubmit}>
            <h1 className="signin__title">Return to a collection</h1>
            <p className="signin__lede">Enter the key to your collection.</p>

            <label className="signin__field">
              <span>API key</span>
              <input
                className="mono"
                value={existingKey}
                onChange={(e) => setExistingKey(e.target.value)}
                placeholder="paste the key you saved"
                autoFocus
              />
            </label>

            {error && <p className="signin__error">{error}</p>}

            <button type="submit" className="signin__submit" disabled={busy}>
              {busy ? <Icon.Loading className="spin" /> : <Icon.ArrowRight />}
              {busy ? 'Checking…' : 'Enter the collection'}
            </button>

            <button type="button" className="signin__switch" onClick={() => setMode('create')}>
              Start a new collection instead
            </button>
          </form>
        )}
      </div>
    </main>
  );
}
