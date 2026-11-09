import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

/**
 * GuestRoute — redirects authenticated users to /dashboard.
 */
export default function GuestRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) return null;

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
}

