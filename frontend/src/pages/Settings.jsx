import { useEffect, useState } from 'react';
import { api, ApiError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { Masthead } from '../components/Masthead';
import { Icon } from '../components/icons';
import './Settings.css';

const MODEL_OPTIONS = [
  { value: 'gpt-5-nano', label: 'gpt-5-nano', note: 'default' },
  { value: 'gpt-5-mini', label: 'gpt-5-mini', note: 'a step up in capability, slower'}
];

const STYLE_OPTIONS = [
  { value: 'plain', label: 'Plain', note: 'everyday language, technical terms defined inline' },
  { value: 'standard', label: 'Standard', note: 'clear language for a technically literate reader' },
  { value: 'technical', label: 'Technical', note: 'precise academic language, no simplification' },
];

export function Settings() {
  const { apiKey } = useAuth();
  const [settings, setSettings] = useState(null);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api
      .getSettings(apiKey)
      .then(setSettings)
      .catch(() => setError('Could not read your settings.'));
  }, [apiKey]);

  function handleField(field, value) {
    setSettings((prev) => ({ ...prev, [field]: value }));
    setSaved(false);
  }

  function handleSave(event) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    api
      .updateSettings(apiKey, settings)
      .then((updated) => {
        setSettings(updated);
        setSaved(true);
      })
      .catch((err) => setError(err instanceof ApiError ? err.detail : 'Could not reach the server.'))
      .finally(() => setSaving(false));
  }

  return (
    <div className="settings">
      <Masthead />
      <p className="settings__intro">
        These apply to your account: how questions are answered, comparisons are drawn, and follow-ups are
        drafted. A change takes effect on your next question.
      </p>

      {error && <p className="notice settings__error">{error}</p>}

      {!settings && !error && <p className="notice settings__loading">Reading your settings…</p>}

      {settings && (
        <form className="ledger-panel settings__panel" onSubmit={handleSave}>
          <fieldset className="settings__group">
            <legend className="section-title">Language model</legend>
            {MODEL_OPTIONS.map((opt) => (
              <label key={opt.value} className="settings__option">
                <input
                  type="radio"
                  name="model"
                  value={opt.value}
                  checked={settings.model === opt.value}
                  onChange={() => handleField('model', opt.value)}
                />
                <span>
                  <span className="mono settings__option-label">{opt.label}</span>
                  <span className="settings__option-note">{opt.note}</span>
                </span>
              </label>
            ))}
          </fieldset>

          <fieldset className="settings__group">
            <legend className="section-title">Temperature</legend>
            <div className="settings__slider-row">
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={settings.temperature}
                onChange={(e) => handleField('temperature', Number(e.target.value))}
              />
              <span className="mono settings__slider-value">{settings.temperature.toFixed(2)}</span>
            </div>
            <p className="settings__note">
              Lower is more consistent and literal; higher allows more varied phrasing. Grounding checks apply
              regardless of this setting.
            </p>
          </fieldset>

          <fieldset className="settings__group">
            <legend className="section-title">Language sophistication</legend>
            {STYLE_OPTIONS.map((opt) => (
              <label key={opt.value} className="settings__option">
                <input
                  type="radio"
                  name="language_style"
                  value={opt.value}
                  checked={settings.language_style === opt.value}
                  onChange={() => handleField('language_style', opt.value)}
                />
                <span>
                  <span className="settings__option-label">{opt.label}</span>
                  <span className="settings__option-note">{opt.note}</span>
                </span>
              </label>
            ))}
          </fieldset>

          <div className="settings__actions">
            <button type="submit" className="settings__save" disabled={saving}>
              {saving ? <Icon.Loading className="spin" size={16} /> : <Icon.Stamp size={16} />}
              {saving ? 'Saving…' : 'Save settings'}
            </button>
            {saved && <span className="settings__saved mono">saved</span>}
          </div>
        </form>
      )}
    </div>
  );
}
