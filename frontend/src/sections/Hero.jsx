import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import DigitalTwin from '../components/DigitalTwin';
import MagneticButton from '../components/ui/MagneticButton';
import Button from '../components/ui/Button';
import Reveal from '../components/ui/Reveal';
import styles from './Hero.module.css';

const WORDS = ['Your', "patient's", 'digital', 'twin,', 'active', 'in', 'every', 'decision.'];

export default function Hero() {
  return (
    <section className={styles.hero} aria-labelledby="hero-heading">
      <div className={styles.content}>
        <Reveal>
          <p className={styles.eyebrow}>// MediTwin · Research Prototype</p>
        </Reveal>

        <Reveal delay={0.08}>
          <h1 id="hero-heading" className={styles.headline}>
            {WORDS.map((word, i) => (
              <motion.span
                key={i}
                style={{ display: 'inline-block', marginRight: '0.28em' }}
                initial={{ opacity: 0, y: 28 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: 0.55,
                  delay: 0.15 + i * 0.06,
                  ease: [0.22, 0.03, 0.26, 1],
                }}
              >
                {word === 'twin,' ? <em>{word}</em> : word}
              </motion.span>
            ))}
          </h1>
        </Reveal>

        <Reveal delay={0.5}>
          <p className={styles.subline}>
            MediTwin synthesises clinical documents, medical imaging, and
            patient history into interpretable risk stratification — with
            explainable AI at every step.
          </p>
        </Reveal>

        <Reveal delay={0.62}>
          <div className={styles.ctas}>
            <MagneticButton as="a" href="#pipeline" size="lg">
              Explore the pipeline
            </MagneticButton>
            <Button
              as="a"
              href="https://github.com/SaiVighnesh811/MediTwin"
              target="_blank"
              rel="noopener noreferrer"
              variant="outline"
              size="lg"
            >
              View source
            </Button>
          </div>
        </Reveal>

        <Reveal delay={0.72}>
          <p className={styles.disclaimer}>
            Decision-support prototype only. Not a medical diagnosis.
          </p>
        </Reveal>
      </div>

      <div className={styles.canvasWrap} aria-hidden="true">
        <DigitalTwin />
        <span className={styles.canvasLabel}>digital twin · live</span>
      </div>
    </section>
  );
}

