import Reveal from '../components/ui/Reveal';
import styles from './IODiagram.module.css';

const INPUTS = [
  { code: 'DOC', label: 'Clinical documents', detail: 'Prescriptions, lab reports, discharge summaries (scanned image)' },
  { code: 'IMG', label: 'Chest X-rays', detail: 'JPEG or PNG images processed by the Medical Image Agent' },
  { code: 'SYM', label: 'Clinical symptoms', detail: 'Structured form: age, BMI, blood pressure, lifestyle factors' },
  { code: 'HX ', label: 'Patient history', detail: 'Prior diagnoses, medications, visit notes from Patient Memory' },
];

const OUTPUTS = [
  { code: 'RISK', label: 'Risk stratification', detail: 'Low / Medium / High diabetes risk with class probabilities' },
  { code: 'SHAP', label: 'Explainability', detail: 'SHAP feature contributions and LIME local explanations per prediction' },
  { code: 'REC ', label: 'Recommendations', detail: 'Evidence-based lifestyle guidance (ADA/WHO), with caveats for assumed features' },
];

export default function IODiagram() {
  return (
    <section className={styles.section} aria-labelledby="io-heading">
      <div>
        <Reveal>
          <p className={styles.label}>// system overview</p>
          <h2 id="io-heading" className={styles.heading}>What goes in.<br />What comes out.</h2>
        </Reveal>
      </div>

      <div>
        <div className={styles.columns}>
          {/* Inputs */}
          <Reveal delay={0.1}>
            <div className={styles.inputs}>
              {INPUTS.map((inp, i) => (
                <div key={i} className={styles.inputItem}>
                  <span className={styles.inputIcon}>{inp.code}</span>
                  <div>
                    <div style={{ fontFamily: 'var(--font-sans)', fontSize: 'var(--text-sm)', color: 'var(--text-primary)', fontWeight: 500, marginBottom: '0.2rem' }}>
                      {inp.label}
                    </div>
                    <span className={styles.inputText}>{inp.detail}</span>
                  </div>
                </div>
              ))}
            </div>
          </Reveal>

          {/* Flow arrow */}
          <Reveal delay={0.2}>
            <div className={styles.flow}>
              <div className={styles.flowArrow} />
              <span>8 agents</span>
              <div className={styles.flowArrow} />
            </div>
          </Reveal>

          {/* Outputs */}
          <Reveal delay={0.3}>
            <div className={styles.outputs}>
              {OUTPUTS.map((out, i) => (
                <div key={i} className={styles.outputItem}>
                  <span className={styles.outputLabel}>{out.code}</span>
                  <div style={{ fontFamily: 'var(--font-sans)', fontSize: 'var(--text-sm)', color: 'var(--text-primary)', fontWeight: 500, marginBottom: '0.2rem' }}>
                    {out.label}
                  </div>
                  <span className={styles.outputText}>{out.detail}</span>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

