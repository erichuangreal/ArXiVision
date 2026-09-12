import { Link, NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './Masthead.css';

const NAV_ITEMS = [
  { to: '/', label: 'Cabinet', end: true },
  { to: '/corpus', label: 'Corpus' },
  { to: '/ask', label: 'Ask' },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/settings', label: 'Settings' },
];

export function Masthead() {
  const { userId, signOut } = useAuth();

  return (
    <header className="masthead">
      <Link to="/" className="masthead__mark">
        Specimen Ledger
        <span className="masthead__mark-note">working name</span>
      </Link>

      {userId && (
        <nav className="masthead__nav mono" aria-label="Primary">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
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
