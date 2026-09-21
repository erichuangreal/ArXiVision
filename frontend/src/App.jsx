import { Navigate, Route, Routes } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { SignIn } from './pages/SignIn';
import { Dashboard } from './pages/Dashboard';
import { Corpus } from './pages/Corpus';
import { CreateCollection } from './pages/CreateCollection';
import { CollectionDetail } from './pages/CollectionDetail';
import { Ask } from './pages/Ask';
import { Settings } from './pages/Settings';
import { Evaluation } from './pages/Evaluation';
import { HowItWorks } from './pages/HowItWorks';

function RequireAuth({ children }) {
  const { apiKey } = useAuth();
  return apiKey ? children : <Navigate to="/sign-in" replace />;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/sign-in" element={<SignIn />} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <Dashboard />
          </RequireAuth>
        }
      />
      <Route
        path="/corpus"
        element={
          <RequireAuth>
            <Corpus />
          </RequireAuth>
        }
      />
      <Route
        path="/ask"
        element={
          <RequireAuth>
            <Ask />
          </RequireAuth>
        }
      />
      <Route
        path="/evaluation"
        element={
          <RequireAuth>
            <Evaluation />
          </RequireAuth>
        }
      />
      <Route
        path="/how-it-works"
        element={
          <RequireAuth>
            <HowItWorks />
          </RequireAuth>
        }
      />
      <Route
        path="/settings"
        element={
          <RequireAuth>
            <Settings />
          </RequireAuth>
        }
      />
      <Route
        path="/collections/new"
        element={
          <RequireAuth>
            <CreateCollection />
          </RequireAuth>
        }
      />
      <Route
        path="/collections/:id"
        element={
          <RequireAuth>
            <CollectionDetail />
          </RequireAuth>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}
