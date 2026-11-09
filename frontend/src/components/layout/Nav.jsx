import { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import Button from '../ui/Button';
import styles from './Nav.module.css';

const NAV_LINKS = [
  { label: 'How it works', href: '#pipeline' },
  { label: 'Performance', href: '#metrics' },
  { label: 'Responsible AI', href: '#responsible' },
];

export default function Nav() {
  const [open, setOpen] = useState(false);
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const isDashboard = location.pathname.startsWith('/dashboard');

  async function handleLogout() {
    await logout();
    navigate('/login', { replace: true });
    setOpen(false);
  }

  return (
    <header>
      <nav className={styles.nav} aria-label="Main navigation">
        <Link to="/" className={styles.wordmark}>
          Medi<span>Twin</span>
        </Link>

        {/* Hide landing links when anywhere inside /dashboard */}
        {!isDashboard && (
          <ul className={styles.links} role="list">
            {NAV_LINKS.map((l) => (
              <li key={l.label}>
                <a href={l.href}>{l.label}</a>
              </li>
            ))}
          </ul>
        )}

        <div className={styles.actions}>
          {user ? (
            <>
              <Button
                as={Link}
                to="/dashboard"
                variant={isDashboard ? 'secondary' : 'ghost'}
                size="sm"
                aria-current={isDashboard ? 'page' : undefined}
              >
                Dashboard
              </Button>
              <Button variant="outline" size="sm" onClick={handleLogout}>
                Logout
              </Button>
            </>
          ) : (
            <>
              <Button as={Link} to="/login" variant="ghost" size="sm">
                Login
              </Button>
              <Button as={Link} to="/signup" size="sm">
                Sign up
              </Button>
            </>
          )}
        </div>

        <button
          className={styles.hamburger}
          aria-expanded={open}
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen((v) => !v)}
        >
          <span />
          <span />
          <span />
        </button>
      </nav>

      <AnimatePresence>
        {open && (
          <motion.div
            className={styles.drawer}
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            role="dialog"
            aria-label="Mobile navigation"
          >
            {!isDashboard &&
              NAV_LINKS.map((l) => (
                <a key={l.label} href={l.href} onClick={() => setOpen(false)}>
                  {l.label}
                </a>
              ))}
            {user ? (
              <>
                <Link
                  to="/dashboard"
                  onClick={() => setOpen(false)}
                  style={isDashboard ? { color: 'var(--accent-teal)', fontWeight: 600 } : undefined}
                  aria-current={isDashboard ? 'page' : undefined}
                >
                  Dashboard
                </Link>
                <button onClick={handleLogout} style={{ textAlign: 'left' }}>
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link to="/login" onClick={() => setOpen(false)}>
                  Login
                </Link>
                <Link to="/signup" onClick={() => setOpen(false)}>
                  Sign up
                </Link>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
