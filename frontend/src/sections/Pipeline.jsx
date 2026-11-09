import { useRef, useEffect, useState } from 'react';
import clsx from 'clsx';
import styles from './Pipeline.module.css';

const AGENTS = [
  {
    name: 'OCR Agent',
    desc: 'Reads scanned prescriptions and reports: EasyOCR for printed text (1.22% character error rate) and a ResNet18 classifier for handwritten medicine names.',
  },
  {
    name: 'NLP Agent',
    desc: 'Finds medical entities in clinical text, such as diagnoses, medicines and measurements, using a pretrained biomedical model (DistilBERT, not trained by us).',
  },
  {
    name: 'Medical Image Agent',
    desc: 'Screens chest X-rays using two models: a pediatric ResNet18 (Kermany dataset; recall 98.5%, specificity 62.0%, measured) and an adult DenseNet121 (TorchXRayVision, trained on public adult datasets; performance not independently measured here). Screening aid only, not for clinical use.',
  },
  {
    name: 'Risk Prediction Agent',
    desc: 'Estimates diabetes risk (Low, Medium or High) with XGBoost trained on BRFSS 2015 survey data. Balanced accuracy 51.45%; High-class recall 62.6%.',
  },
  {
    name: 'Explainable AI Agent',
    desc: 'Explains each prediction with SHAP and LIME, showing which factors pushed the risk up or down.',
  },
  {
    name: 'Recommendation Agent',
    desc: 'Gives educational lifestyle and prevention guidance from 10 transparent rules based on public CDC, AHA and WHO guidance. Not medical advice.',
  },
  {
    name: 'Patient Memory Agent',
    desc: 'Keeps a patient timeline of dated events, such as medicines and lab results, in a local JSON store, with duplicate detection.',
  },
  {
    name: 'Coordination Agent',
    desc: 'Runs the agents in order and assembles the 21 survey features: provided values first, then values extracted from documents, then typical training values. It reports how many were assumed.',
  },
];

function AgentCard({ agent, index }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.15 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={clsx(styles.agent, visible && styles.active)}
      style={{ transitionDelay: `${index * 60}ms` }}
    >
      <span className={styles.agentNum}>0{index + 1}</span>
      <span className={styles.agentName}>{agent.name}</span>
      <p className={styles.agentDesc}>{agent.desc}</p>
    </div>
  );
}

export default function Pipeline() {
  return (
    <section id="pipeline" className={styles.section}>
      <div className={styles.inner}>
        <h2 className={styles.heading}>
          Eight agents,<br /><span>one coherent twin.</span>
        </h2>
        <div className={styles.grid}>
          {AGENTS.map((a, i) => (
            <AgentCard key={i} agent={a} index={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
