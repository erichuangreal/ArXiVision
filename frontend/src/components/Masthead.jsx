import { useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './Masthead.css';

const NAV_ITEMS = [
  { to: '/', label: 'Cabinet', end: true },
  { to: '/corpus', label: 'Corpus' },
  { to: '/ask', label: 'Ask' },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/settings', label: 'Settings' },
  { to: '/how-it-works', label: 'How this works' },
];

export function Masthead() {
  const { userId, signOut } = useAuth();
  const [navOpen, setNavOpen] = useState(false);

  return (
    <header className="masthead">
      <Link to="/" className="masthead__mark">
        ArXiVision
      </Link>

      {userId && (
        <button
          type="button"
          className="masthead__nav-toggle mono"
          aria-expanded={navOpen}
          aria-controls="masthead-nav"
          onClick={() => setNavOpen((open) => !open)}
        >
          {navOpen ? 'close' : 'menu'}
        </button>
      )}

      {userId && (
        <nav
          id="masthead-nav"
          className={`masthead__nav mono${navOpen ? ' masthead__nav--open' : ''}`}
          aria-label="Primary"
        >
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              onClick={() => setNavOpen(false)}
              className={({ isActive }) => 'masthead__nav-link' + (isActive ? ' masthead__nav-link--active' : '')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      )}

      {userId && (
        <div className="masthead__account mono">
          <span>cataloguer {userId.slice(0, 8)}</span>
          <button type="button" className="masthead__signout" onClick={signOut}>
            sign out
          </button>
        </div>
      )}
    </header>
  );
}
