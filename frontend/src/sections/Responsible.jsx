import Reveal from '../components/ui/Reveal';
import styles from './Responsible.module.css';

const PILLARS = [
  {
    title: 'Privacy by design',
    desc: 'All processing runs locally. No patient data leaves the device. No telemetry, no cloud inference, no external model APIs.',
  },
  {
    title: 'Explainability at every step',
    desc: 'Every prediction ships with SHAP feature attributions and LIME local explanations. Assumed feature values are explicitly labelled so clinicians know what was missing.',
  },
  {
    title: 'Education-only outputs',
    desc: 'Recommendations are framed as evidence-based information (ADA / WHO guidelines), not instructions. The system cannot prescribe, diagnose, or replace a clinician.',
  },
  {
    title: 'Honest about limitations',
    desc: "The Medium risk class is unreliable (precision ~3%). The pediatric chest X-ray model was measured on children's X-rays; the adult model's performance was not independently measured in this project. OCR was evaluated on synthetic images only. We state this everywhere.",
  },
];

export default function Responsible() {
  return (
    <section id="responsible" className={styles.section} aria-labelledby="responsible-heading">
      <Reveal>
        <p className={styles.label}>// responsible AI</p>
        <h2 id="responsible-heading" className={styles.heading}>
          Built to assist,<br />not to decide.
        </h2>
        <p className={styles.intro}>
          MediTwin is designed with clear ethical boundaries. The system supports clinical thinking
          — it does not replace it.
        </p>
      </Reveal>

      <Reveal delay={0.1}>
        <div className={styles.pillars}>
          {PILLARS.map((p, i) => (
            <div key={i} className={styles.pillar}>
              <span className={styles.pillarNum}>0{i + 1}</span>
              <div className={styles.pillarContent}>
                <h3 className={styles.pillarTitle}>{p.title}</h3>
                <p className={styles.pillarDesc}>{p.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </Reveal>
    </section>
  );
}

