import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Nav from '../components/layout/Nav';
import Footer from '../components/layout/Footer';
import Button from '../components/ui/Button';
import Reveal from '../components/ui/Reveal';
import styles from './Dashboard.module.css';

export default function Dashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout();
    navigate('/login', { replace: true });
  }

  return (
    <div className={styles.dashboard}>
      <Nav />
      <main className={styles.main}>
        <Reveal>
          <p className={styles.eyebrow}>// chapter 13 · coming soon</p>
          <h1 className={styles.greeting}>
            Welcome, <em>{user?.name || 'User'}</em>
          </h1>
          <p className={styles.body}>
            Dashboard arrives in Chapter 13. For now, your session is active
            and authenticated.
          </p>
          <p className={styles.meta}>
            {user?.email} · {user?.role}
          </p>
          <div className={styles.actions}>
            <Button variant="outline" size="md" onClick={handleLogout}>
              Log out
            </Button>
            <Button as={Link} to="/" variant="ghost" size="md">
              &larr; Home
            </Button>
          </div>
        </Reveal>
      </main>
      <Footer />
    </div>
  );
}

