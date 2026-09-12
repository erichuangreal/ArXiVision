import { createContext, useCallback, useContext, useMemo, useState } from 'react';

const STORAGE_KEY = 'specimen-ledger.auth';
const AuthContext = createContext(null);

function loadStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(loadStored);

  const signIn = useCallback((userId, apiKey) => {
    const value = { userId, apiKey };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    } catch {
      // localStorage unavailable (private mode, etc.) - session still works in memory.
    }
    setAuth(value);
  }, []);

  const signOut = useCallback(() => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
    setAuth(null);
  }, []);

  const value = useMemo(
    () => ({ userId: auth?.userId ?? null, apiKey: auth?.apiKey ?? null, signIn, signOut }),
    [auth, signIn, signOut]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
