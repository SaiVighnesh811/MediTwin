import { Link } from 'react-router-dom';
import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.top}>
        <Link to="/" className={styles.wordmark}>
          Medi<span>Twin</span>
        </Link>
        <ul className={styles.links} role="list">
          <li><a href="#pipeline">Pipeline</a></li>
          <li><a href="#metrics">Performance</a></li>
          <li><a href="#responsible">Responsible AI</a></li>
          <li>
            <a
              href="https://github.com/SaiVighnesh811/MediTwin"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </li>
        </ul>
      </div>

      <p className={styles.disclaimer}>
        MediTwin is a decision-support research prototype. It is not a medical
        device, not approved for clinical use, and does not constitute a medical
        diagnosis. All outputs are for educational and research purposes only.
        Risk scores are based on survey-derived population statistics and must
        not be used for individual clinical decisions.
      </p>

      <p className={styles.copy}>
        &copy; {new Date().getFullYear()} MediTwin · Built for research
      </p>
    </footer>
  );
}

