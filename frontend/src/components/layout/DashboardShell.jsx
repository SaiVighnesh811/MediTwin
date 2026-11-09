import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import Nav from './Nav';
import Button from '../ui/Button';
import styles from './DashboardShell.module.css';

export default function DashboardShell({ children }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout();
    navigate('/login', { replace: true });
  }

  return (
    <>
      <Nav />
      <div className={styles.shell}>
        {/* Sidebar for Desktop */}
        <aside className={styles.sidebar}>
          <div className={styles.userCard}>
            <div className={styles.userName}>{user?.name || 'Patient'}</div>
            <div className={styles.userEmail}>{user?.email}</div>
          </div>

          <ul className={styles.navLinks}>
            <li>
              <NavLink
                to="/dashboard"
                end
                className={({ isActive }) =>
                  `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
                }
              >
                Overview
              </NavLink>
            </li>
            <li>
              <NavLink
                to="/dashboard/new"
                className={({ isActive }) =>
                  `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
                }
              >
                New Analysis
              </NavLink>
            </li>
            <li>
              <NavLink
                to="/dashboard/history"
                className={({ isActive }) =>
                  `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
                }
              >
                History
              </NavLink>
            </li>
          </ul>

          <div className={styles.sidebarFooter}>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              Logout
            </Button>
          </div>
        </aside>

        {/* Content area */}
        <main className={styles.contentArea}>{children}</main>

        {/* Mobile Navigation bar */}
        <nav className={styles.mobileBar} aria-label="Dashboard mobile navigation">
          <NavLink
            to="/dashboard"
            end
            className={({ isActive }) =>
              `${styles.mobileLink} ${isActive ? styles.mobileLinkActive : ''}`
            }
          >
            Overview
          </NavLink>
          <NavLink
            to="/dashboard/new"
            className={({ isActive }) =>
              `${styles.mobileLink} ${isActive ? styles.mobileLinkActive : ''}`
            }
          >
            + New
          </NavLink>
          <NavLink
            to="/dashboard/history"
            className={({ isActive }) =>
              `${styles.mobileLink} ${isActive ? styles.mobileLinkActive : ''}`
            }
          >
            History
          </NavLink>
          <button
            onClick={handleLogout}
            className={styles.mobileLink}
            style={{ background: 'none', border: 'none', cursor: 'pointer' }}
          >
            Logout
          </button>
        </nav>
      </div>
    </>
  );
}

