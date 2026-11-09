import { Link } from 'react-router-dom';
import MagneticButton from '../components/ui/MagneticButton';
import Button from '../components/ui/Button';
import Reveal from '../components/ui/Reveal';
import styles from './FinalCTA.module.css';

export default function FinalCTA() {
  return (
    <section className={styles.section} aria-labelledby="cta-heading">
      <Reveal>
        <h2 id="cta-heading" className={styles.heading}>
          Your patient&apos;s <em>digital twin</em> awaits.
        </h2>
      </Reveal>
      <Reveal delay={0.1}>
        <p className={styles.sub}>
          MediTwin is an open-source research prototype. Explore the codebase,
          contribute, or sign up to follow the project as it evolves.
        </p>
      </Reveal>
      <Reveal delay={0.2}>
        <div className={styles.actions}>
          <MagneticButton as={Link} to="/signup" size="lg">
            Sign up
          </MagneticButton>
          <Button
            as="a"
            href="https://github.com/SaiVighnesh811/MediTwin"
            target="_blank"
            rel="noopener noreferrer"
            variant="outline"
            size="lg"
          >
            Read the source
          </Button>
        </div>
      </Reveal>
    </section>
  );
}

