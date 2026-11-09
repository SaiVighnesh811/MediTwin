import GlassPanel from '../components/ui/GlassPanel';
import Reveal from '../components/ui/Reveal';
import styles from './HonestMetrics.module.css';

export default function HonestMetrics() {
  return (
    <section id="metrics" className={styles.section} aria-labelledby="metrics-heading">
      <Reveal>
        <p className={styles.label}>// performance · measured honestly</p>
        <h2 id="metrics-heading" className={styles.heading}>Real numbers, real caveats.</h2>
        <p className={styles.subheading}>
          We report what the models actually achieve on test data. No cherry-picked
          metrics. Medium class performance for the risk model is unreliable — we say so plainly.
        </p>
      </Reveal>

      <div className={styles.grid}>
        {/* Risk Model */}
        <Reveal delay={0.1}>
          <GlassPanel className={styles.card}>
            <span className={styles.cardLabel}>risk model · BRFSS 2015 · XGBoost</span>
            <h3 className={styles.cardTitle}>Diabetes Risk Prediction</h3>
            <div className={styles.metrics}>
              <p className={styles.metric}>
                Balanced accuracy: <strong>51.45%</strong>
              </p>
              <p className={styles.metric}>
                High-risk recall: <strong>62.6%</strong>
              </p>
              <p className={styles.metric}>
                Medium-class precision: <strong>~3%</strong> (unreliable class)
              </p>
            </div>
            <p className={styles.caveat}>
              Trained on BRFSS survey self-report data. The &quot;Medium&quot; risk class is severely
              underrepresented; its predictions must not be acted upon. Not for clinical use.
            </p>
          </GlassPanel>
        </Reveal>

        {/* Chest X-ray */}
        <Reveal delay={0.2}>
          <GlassPanel className={styles.card}>
            <span className={styles.cardLabel}>chest x-ray · pediatric Kermany ResNet18 (measured)</span>
            <h3 className={styles.cardTitle}>Pediatric Chest X-Ray Screening (Measured)</h3>
            <div className={styles.metrics}>
              <p className={styles.metric}>
                Recall (sensitivity): <strong>98.46%</strong>
              </p>
              <p className={styles.metric}>
                Specificity: <strong>61.97%</strong>
              </p>
            </div>
            <div className={styles.matrixWrap} aria-label="Confusion matrix">
              <div className={`${styles.matrixCell} ${styles.tn}`}>TN 145</div>
              <div className={styles.matrixCell}>FP 89</div>
              <div className={styles.matrixCell}>FN 6</div>
              <div className={`${styles.matrixCell} ${styles.tp}`}>TP 384</div>
            </div>
            <p className={styles.caveat}>
              Third-party pretrained model, fine-tuned by its original author on pediatric chest X-rays (not trained by us). High false
              positive rate (89/234) limits specificity. Not validated for adult populations.
              The adult X-ray model has no independent measurements here. Not for clinical use.
            </p>
          </GlassPanel>
        </Reveal>

        {/* OCR */}
        <Reveal delay={0.3}>
          <GlassPanel className={styles.card}>
            <span className={styles.cardLabel}>OCR · EasyOCR · printed text</span>
            <h3 className={styles.cardTitle}>Document Extraction</h3>
            <div className={styles.metrics}>
              <p className={styles.metric}>
                Character error rate: <strong>1.22%</strong>
              </p>
              <p className={styles.metric}>
                Word error rate: <strong>5.09%</strong>
              </p>
            </div>
            <p className={styles.caveat}>
              Evaluated on synthetic PIL-generated printed images. Handwritten text, low-DPI scans
              and non-English scripts are not evaluated and may perform significantly worse. The separate 78-medicine handwriting classifier is evaluated in the project reports.
            </p>
          </GlassPanel>
        </Reveal>
      </div>

      <p className={styles.syntheticNote}>
        Patient Memory, Recommendation, and Coordination agents were validated with synthetic
        sanity tests only. No held-out clinical dataset was used for those components.
      </p>
    </section>
  );
}

