import { useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import GlassPanel from '../ui/GlassPanel';
import Button from '../ui/Button';
import styles from './AnalysisResultView.module.css';
import {
  formatFeature,
  formatFeatureName,
  formatFeatureInText,
  formatRuleTitle,
  formatCategory,
  computeDivergingBars,
  getGroupedEntities,
  parseOCRFindings,
} from '../../lib/brfss';
import { api } from '../../lib/api';

/**
 * AnalysisResultView
 * Displays results according to strict clinical and honest reporting requirements:
 * 1. Disclaimer bar
 * 2. Reliability banner
 * 3. Never display a confident single large label if class is "Medium" or reliability is "low"
 * 4. SVG probability bars & diverging SHAP/LIME charts with real non-empty bars
 * 5. Honest limitations toggles on all cards
 * 6. Partial notice appears ONLY on real agent failures; unprovided inputs are shown neutrally as collapsed details.
 * 7. All BRFSS features and recommendation rule codes rendered as human-readable text.
 * 8. Recommendations displayed once as written by the agent with clean category tag and NO checkboxes.
 */
export default function AnalysisResultView({ data }) {
  const shouldReduceMotion = useReducedMotion();
  const [limeLoading, setLimeLoading] = useState(false);
  const [limeError, setLimeError] = useState(null);
  const [onDemandLime, setOnDemandLime] = useState(null);

  if (!data) return null;

  // Normalize shape whether raw ML response or saved Analysis document
  const result = data.result || data;
  const analysisType = data.type;
  const isImageResult =
    analysisType === 'image' ||
    Boolean(
      result.model_used?.includes('xray') ||
      result.model_used?.includes('densenet') ||
      result.model_used?.includes('resnet') ||
      result.pneumonia_probability !== undefined ||
      result.pneumonia_score !== undefined
    );

  const reliability = result.reliability || data.reliability;
  // If this is an image-only result, do not fall back to result.predicted_class for diabetes risk
  const risk =
    isImageResult && !result.risk_assessment
      ? {}
      : result.risk_assessment || (result.predicted_class && !isImageResult ? result : {});
  const xai =
    result.explainability ||
    (result.shap_top5 ? { top_shap_features: result.shap_top5, top_lime_features: result.lime_top5 } : {});
  const imaging =
    result.imaging ||
    (isImageResult ? result : null) ||
    (result.pneumonia_probability !== undefined ? result : null) ||
    (result.pneumonia_score !== undefined ? result : null);
  const ocr = result.ocr_findings || result.ocr || (result.extracted_fields ? result : null);
  const nlp = result.nlp_entities || result.nlp || (result.entities ? result : null);
  const ocrData = parseOCRFindings(ocr);
  const groupedEntities = getGroupedEntities(nlp);
  const hasNlpContent = Object.values(groupedEntities).some((arr) => arr.length > 0);
  const recs = result.recommendations || [];
  const agentStatus = result.agent_status || {};

  // Categorize agent executions: real failures vs neutral skipped (no input)
  const failedAgents = Object.entries(agentStatus).filter(([_, s]) =>
    typeof s === 'string' && s.toLowerCase().startsWith('failed')
  );
  const hasExplicitErrors =
    (result.errors && typeof result.errors === 'object' && Object.keys(result.errors).length > 0) ||
    (typeof result.error === 'string' && result.error.length > 0);
  const hasFailures = failedAgents.length > 0 || hasExplicitErrors;

  const skippedAgents = Object.entries(agentStatus).filter(([_, s]) =>
    typeof s === 'string' && s.toLowerCase().startsWith('skipped')
  );

  // Logic: Deciding how risk class is shown
  const predClass = risk.predicted_class || 'Not assessed';
  const isLowReliability = !reliability?.level || reliability.level === 'low';
  const isMediumClass = (predClass || '').toLowerCase() === 'medium';
  const showMutedClass = isLowReliability || isMediumClass;

  const probs = risk.probabilities || {};
  const completeness = risk.completeness || (risk.features_used ? `${risk.features_used} features evaluated` : '');
  const assumedFeatures = risk.assumed_features || [];

  // Parse explanation bars
  const rawShap = xai.top_shap_features || [];
  const rawLime = onDemandLime || xai.top_lime_features || [];

  const shapBars = computeDivergingBars(rawShap, 'shap');
  const limeBars = computeDivergingBars(rawLime, 'lime');

  // On-demand LIME runner (Case 3b fallback)
  async function handleRunLime() {
    setLimeLoading(true);
    setLimeError(null);
    try {
      const features = data.inputSummary?.featuresProvided || risk.features || {};
      const res = await api('/api/analysis/explain', {
        method: 'POST',
        body: JSON.stringify({ features, patient_id: result.patient_id }),
      });
      const limeRes = res.top_lime_features || res.result?.explainability?.top_lime_features || [];
      setOnDemandLime(limeRes);
    } catch (err) {
      setLimeError(err.message || 'Failed to generate LIME explanation');
    } finally {
      setLimeLoading(false);
    }
  }

  const hasLimeContent = limeBars.length > 0;
  const explainCardTitle = hasLimeContent
    ? 'Model Explainability (SHAP & LIME)'
    : 'Model Explainability (SHAP)';

  return (
    <div className={styles.container}>
      {/* a. Disclaimer Bar */}
      <div className={styles.disclaimerBar} role="note">
        <strong>CRITICAL DISCLAIMER:</strong> Decision-support prototype. Not a medical diagnosis.
      </div>

      {/* b. Input completeness banner */}
      {reliability && (
        <div
          className={`${styles.reliabilityBanner} ${styles[`rel_${reliability.level}`] || ''}`}
          role="status"
          aria-live="polite"
        >
          <div className={styles.reliabilityTitle}>
            Input completeness: {reliability.level}
          </div>
          <p style={{ margin: '0.25rem 0 0.5rem 0', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            This describes how many inputs were available, not the accuracy of the prediction.
          </p>
          {reliability.reasons?.length > 0 && (
            <ul className={styles.reliabilityReasons}>
              {reliability.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Partial Notice: amber notice appears ONLY when an agent really FAILED */}
      {hasFailures && (
        <div className={styles.partialNotice} role="alert">
          <strong>Notice: Partial Pipeline Output</strong>
          <p style={{ margin: '0.4rem 0 0 0' }}>
            Some components could not complete. Remaining agents succeeded with partial execution.
          </p>
          <ul style={{ margin: '0.4rem 0 0 1.25rem', fontSize: '0.85rem' }}>
            {failedAgents.map(([agent, s]) => {
              const reason = result.agent_errors?.[agent] || result.errors?.[agent];
              const showReason = reason && !s.includes(reason);
              return (
                <li key={agent}>
                  <code>{agent}</code>: {s}
                  {showReason && <span style={{ color: 'var(--text-secondary)', marginLeft: '0.5rem' }}>({reason})</span>}
                </li>
              );
            })}
            {failedAgents.length === 0 && (result.agent_errors || result.errors) && (
              Object.entries(result.agent_errors || result.errors).map(([agent, err]) => (
                <li key={agent}>
                  <code>{agent}</code>: failed <span style={{ color: 'var(--text-secondary)', marginLeft: '0.5rem' }}>({String(err)})</span>
                </li>
              ))
            )}
            {failedAgents.length === 0 && !result.agent_errors && !result.errors && result.error && (
              <li>{result.error}</li>
            )}
          </ul>
        </div>
      )}

      {/* Neutral skipped notice for unprovided inputs (collapsed by default) */}
      {skippedAgents.length > 0 && (
        <details className={styles.skippedBox}>
          <summary className={styles.skippedSummary}>
            Not run: no input provided ({skippedAgents.length} {skippedAgents.length === 1 ? 'agent' : 'agents'})
          </summary>
          <ul className={styles.skippedList}>
            {skippedAgents.map(([agent, s]) => (
              <li key={agent}>
                <code>{agent}</code>: {s}
              </li>
            ))}
          </ul>
        </details>
      )}

      {/* c. Risk Assessment Card */}
      {risk.predicted_class && (
        <GlassPanel>
          <h2 className={styles.riskClassHeading}>Epidemiological Risk Stratification</h2>

          {/* If Medium class or low reliability, show under a banner with muted formatting, never as a confident label */}
          {showMutedClass ? (
            <div className={styles.riskClassMuted}>
              Estimated Category: <strong>{predClass}</strong> (High Uncertainty — see reliability banner)
            </div>
          ) : (
            <div className={styles.riskClassConfident}>
              Class: {predClass} Risk
            </div>
          )}

          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginBottom: '1rem' }}>
            {completeness}
          </p>

          {/* Probability Bars */}
          <div style={{ marginTop: '1rem' }}>
            {['Low', 'Medium', 'High'].map((cls) => {
              const val = probs[cls] !== undefined ? probs[cls] : 0;
              const pct = Math.round(val * 100);
              return (
                <div key={cls} className={styles.barRow}>
                  <span>{cls}</span>
                  <div className={styles.barTrack}>
                    <div
                      className={`${styles.barFill} ${styles[`barFill_${cls}`]}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span style={{ textAlign: 'right' }}>{pct}%</span>
                </div>
              );
            })}
          </div>

          {/* Expandable assumed features translated to human-readable names */}
          {assumedFeatures.length > 0 && (
            <details className={styles.detailsBox}>
              <summary className={styles.detailsSummary}>
                View {assumedFeatures.length} assumed features (training set medians used)
              </summary>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.4rem', lineHeight: 1.6 }}>
                {assumedFeatures.map((f) => formatFeature(f)).join(', ')}
              </p>
            </details>
          )}

          {/* Limitations */}
          {result.limitations?.length > 0 && (
            <details className={styles.detailsBox}>
              <summary className={styles.detailsSummary}>Limitations & Model Verification Notes</summary>
              <ul className={styles.limitationsList}>
                {result.limitations.map((lim, i) => (
                  <li key={i}>{lim}</li>
                ))}
              </ul>
            </details>
          )}
        </GlassPanel>
      )}

      {/* d. Explainability Card (SHAP & LIME) */}
      {(shapBars.length > 0 || limeBars.length > 0 || data.inputSummary?.featuresProvided) && (
        <GlassPanel>
          <h2 className={styles.riskClassHeading}>{explainCardTitle}</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1rem' }}>
            Attribution values reflect statistical influence within the model, not causal biological mechanisms.
          </p>

          {/* SHAP Chart */}
          {shapBars.length > 0 && (
            <div>
              <h3 style={{ fontSize: '0.95rem', fontFamily: 'var(--font-mono)', marginBottom: '0.5rem', color: 'var(--accent-teal)' }}>
                Top Contributing Factors (SHAP Permutation)
              </h3>
              <div className={styles.axisCaption} aria-hidden="true">
                <span>&larr; lowers High-risk probability</span>
                <span>no effect (0)</span>
                <span>raises High-risk probability &rarr;</span>
              </div>

              <div className={styles.divergeChart}>
                {shapBars.map((bar, i) => {
                  const isPositive = bar.direction === 'right';
                  return (
                    <div key={i} style={{ marginBottom: '0.75rem' }}>
                      <div className={styles.divergeRow}>
                        <span className={styles.divergeLabel} title={bar.label}>
                          {bar.label}
                        </span>
                        <div className={styles.divergeCenter}>
                          <div className={styles.divergeZeroLine} />
                          <motion.div
                            className={isPositive ? styles.divergeBarRight : styles.divergeBarLeft}
                            initial={shouldReduceMotion ? { width: `${bar.widthPct}%` } : { width: 0 }}
                            animate={{ width: `${bar.widthPct}%` }}
                            transition={shouldReduceMotion ? { duration: 0 } : { duration: 0.5, ease: 'easeOut' }}
                          />
                        </div>
                        <span
                          className={`${styles.divergeValue} ${isPositive ? styles.valPositive : styles.valNegative}`}
                          title={`Influence: ${bar.formattedValue}`}
                        >
                          {bar.formattedValue} {bar.indicator}
                        </span>
                      </div>
                      {bar.sentence && (
                        <div className={styles.divergeSentence}>
                          {bar.sentence}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              <p className={styles.explainFooter}>
                Longer bars mean a stronger influence on this model's estimate. This describes the model, not medical causes.
              </p>
            </div>
          )}

          {/* LIME Chart (Case 3a: present in saved result) */}
          {limeBars.length > 0 && (
            <div style={{ marginTop: '1.75rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-hairline)' }}>
              <h3 style={{ fontSize: '0.95rem', fontFamily: 'var(--font-mono)', marginBottom: '0.5rem', color: 'var(--accent-teal)' }}>
                Top Local Rule Influences (LIME Local Surrogates)
              </h3>
              <div className={styles.axisCaption} aria-hidden="true">
                <span>&larr; lowers High-risk probability</span>
                <span>no effect (0)</span>
                <span>raises High-risk probability &rarr;</span>
              </div>

              <div className={styles.divergeChart}>
                {limeBars.map((bar, i) => {
                  const isPositive = bar.direction === 'right';
                  return (
                    <div key={i} style={{ marginBottom: '0.75rem' }}>
                      <div className={styles.divergeRow}>
                        <span className={styles.divergeLabel} title={bar.label}>
                          {bar.label}
                        </span>
                        <div className={styles.divergeCenter}>
                          <div className={styles.divergeZeroLine} />
                          <motion.div
                            className={isPositive ? styles.divergeBarRight : styles.divergeBarLeft}
                            initial={shouldReduceMotion ? { width: `${bar.widthPct}%` } : { width: 0 }}
                            animate={{ width: `${bar.widthPct}%` }}
                            transition={shouldReduceMotion ? { duration: 0 } : { duration: 0.5, ease: 'easeOut' }}
                          />
                        </div>
                        <span
                          className={`${styles.divergeValue} ${isPositive ? styles.valPositive : styles.valNegative}`}
                          title={`Weight: ${bar.formattedValue}`}
                        >
                          {bar.formattedValue} {bar.indicator}
                        </span>
                      </div>
                      {bar.sentence && (
                        <div className={styles.divergeSentence}>
                          {bar.sentence}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              <p className={styles.explainFooter}>
                Longer bars mean a stronger influence in the local linear surrogate model. This describes the model, not medical causes.
              </p>
            </div>
          )}

          {/* Fallback button if LIME not present (Case 3b) */}
          {limeBars.length === 0 && (
            <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-hairline)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleRunLime}
                  disabled={limeLoading}
                >
                  {limeLoading ? 'Calculating LIME weights...' : 'Explain with LIME'}
                </Button>
                {limeLoading && (
                  <span style={{ fontSize: '0.82rem', color: 'var(--accent-amber)' }}>
                    LIME can take a while — perturbing local feature samples...
                  </span>
                )}
              </div>
              {limeError && (
                <div style={{ color: 'var(--accent-rose)', fontSize: '0.82rem', marginTop: '0.5rem' }}>
                  {limeError}
                </div>
              )}
            </div>
          )}
        </GlassPanel>
      )}

      {/* e. Chest Imaging Card */}
      {imaging && (imaging.predicted_class || imaging.pneumonia_score !== undefined) && (
        <GlassPanel>
          <h2 className={styles.riskClassHeading}>Chest X-Ray Screening Aid</h2>

          {/* Adult model: TorchXRayVision DenseNet121-res224-all */}
          {imaging.model_used === 'adult_torchxrayvision_densenet121_all' ? (
            <>
              <div style={{ marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Model: TorchXRayVision DenseNet121-res224-all &nbsp;&middot;&nbsp; Population:{' '}
                  {imaging.population}
                </span>
              </div>

              {/* Pneumonia score gauge */}
              <div style={{ margin: '0.5rem 0 1rem 0' }}>
                <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginBottom: '0.3rem' }}>
                  Pneumonia model score
                  {typeof imaging.pneumonia_score === 'number' && (
                    <span> — {imaging.pneumonia_score >= 0.5 ? 'Score above the model\'s operating point' : 'Score below the model\'s operating point'}</span>
                  )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div style={{ flex: 1, height: 10, background: 'var(--bg-surface-elevated)', borderRadius: 5, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${Math.round((imaging.pneumonia_score || 0) * 100)}%`, background: 'var(--accent-teal)', borderRadius: 5 }} />
                  </div>
                  <span style={{ fontWeight: 600, minWidth: 40 }}>
                    {((imaging.pneumonia_score || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
                  Higher score = model sees more pneumonia-like patterns.
                </div>
              </div>

              {/* Related findings */}
              {imaging.related_findings && (
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem', color: 'var(--text-secondary)' }}>
                    Related findings
                  </div>
                  <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    {Object.entries(imaging.related_findings).map(([label, val]) => (
                      <div key={label} style={{ fontSize: '0.82rem' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>{label}: </span>
                        <span style={{ fontWeight: 600 }}>{val !== null ? `${(val * 100).toFixed(1)}%` : '—'}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className={styles.detailsBox}>
                <div style={{ fontSize: '0.82rem', color: 'var(--accent-amber)', marginBottom: '0.4rem' }}>
                  <strong>Important:</strong> Performance of this model was not independently measured in this project.
                  Training data overlaps every large public adult chest X-ray dataset, so independent accuracy cannot be reported.
                  Screening aid only — not a diagnosis. A licensed radiologist must interpret all results.
                </div>
                {imaging.limitations && (
                  <ul style={{ margin: '0.4rem 0 0 1rem', padding: 0, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    {imaging.limitations.map((l, i) => <li key={i}>{l}</li>)}
                  </ul>
                )}
              </div>
            </>
          ) : (
            /* Pediatric model: Kermany ResNet18 */
            <>
              <div style={{ margin: '0.5rem 0 1rem 0' }}>
                <span style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--accent-teal)' }}>
                  Screening Finding: {imaging.predicted_class}
                </span>
                <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  Pneumonia probability: {Math.round((imaging.pneumonia_probability || 0) * 100)}%
                  {imaging.threshold_used !== undefined && (
                    <span> (threshold: {imaging.threshold_used})</span>
                  )}
                </div>
              </div>
              <div className={styles.detailsBox}>
                <div style={{ fontSize: '0.82rem', color: 'var(--accent-amber)', marginBottom: '0.4rem' }}>
                  <strong>Important Clinical Caveat:</strong> Pediatric single-centre model (Guangzhou Women &amp; Children&apos;s Medical Center).
                  Official test specificity is 62.0% (false alarms on ~38% of normal cases). Screening aid only; never a definitive radiological diagnosis.
                </div>
                {imaging.limitations && (
                  <ul style={{ margin: '0.4rem 0 0 1rem', padding: 0, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    {imaging.limitations.map((l, i) => <li key={i}>{l}</li>)}
                  </ul>
                )}
              </div>
            </>
          )}
        </GlassPanel>
      )}

      {/* f. Document Extraction (OCR) Card */}
      {!isImageResult && (
        <GlassPanel>
          <h2 className={styles.riskClassHeading}>Document Extraction (Prescription &amp; Lab OCR)</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 1rem 0' }}>
            Read automatically from the image. Check against the original document.
          </p>
          {!ocrData.hasOcr ? (
            <div style={{ color: 'var(--text-tertiary)', fontSize: '0.88rem', fontStyle: 'italic' }}>
              No document uploaded
            </div>
          ) : (
            <div>
              {ocrData.rawText && (
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '0.3rem', fontWeight: 600 }}>
                    Extracted Text
                  </div>
                  <div style={{ background: 'var(--bg-surface-elevated)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.82rem', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap' }}>
                    {ocrData.rawText}
                  </div>
                </div>
              )}

              {ocrData.medicines.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ fontSize: '0.82rem', color: 'var(--accent-teal)', marginBottom: '0.4rem', fontWeight: 600 }}>
                    Extracted Medicines
                  </div>
                  <div style={{ display: 'grid', gap: '0.5rem' }}>
                    {ocrData.medicines.map((m, idx) => {
                      const medName = m.medicine || 'Medicine';
                      const dose = m.dosage || 'dose not read';
                      const parts = [medName, dose];
                      if (m.frequency) parts.push(m.frequency);
                      if (m.duration) parts.push(m.duration);
                      return (
                        <div key={idx} style={{ background: 'var(--bg-surface-elevated)', padding: '0.6rem 0.85rem', borderRadius: 'var(--radius-sm)', fontSize: '0.82rem' }}>
                          <strong>{medName}</strong>, {parts.slice(1).join(', ')}
                          {m.confidence !== undefined && (
                            <span style={{ color: 'var(--text-tertiary)', marginLeft: '0.5rem' }}>
                              (conf: {Math.round(m.confidence * 100)}%)
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {ocrData.handwriting.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ fontSize: '0.82rem', color: 'var(--accent-amber)', marginBottom: '0.4rem', fontWeight: 600 }}>
                    Handwritten Medicine Candidates
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {ocrData.handwriting.map((h, idx) => (
                      <div key={idx} style={{ background: 'var(--bg-surface-elevated)', padding: '0.4rem 0.7rem', borderRadius: 'var(--radius-sm)', fontSize: '0.8rem' }}>
                        <span>{h.medicine}</span>
                        {h.confidence !== undefined && (
                          <span style={{ color: 'var(--accent-teal)', fontWeight: 600, marginLeft: '0.35rem' }}>
                            {Math.round(h.confidence * 100)}%
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {ocrData.labValues.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ fontSize: '0.82rem', color: 'var(--accent-blue)', marginBottom: '0.4rem', fontWeight: 600 }}>
                    Extracted Lab Values
                  </div>
                  <div style={{ display: 'grid', gap: '0.5rem' }}>
                    {ocrData.labValues.map((l, idx) => {
                      const refText = l.reference
                        ? `(printed reference: ${l.reference}${l.flag ? ` — ${l.flag}` : ''})`
                        : '';
                      return (
                        <div key={idx} style={{ background: 'var(--bg-surface-elevated)', padding: '0.6rem 0.85rem', borderRadius: 'var(--radius-sm)', fontSize: '0.82rem' }}>
                          <strong>{l.test || 'Lab Test'}</strong>: {l.value} {l.unit} {refText}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </GlassPanel>
      )}

      {/* g. Clinical Notes & Biomedical Entities (NLP) Card */}
      {hasNlpContent && (
        <GlassPanel>
          <h2 className={styles.riskClassHeading}>Biomedical Entities (Clinical Notes NER)</h2>
          <div style={{ display: 'grid', gap: '1rem' }}>
            {['Condition', 'Medication', 'Symptom', 'Procedure'].map((category) => {
              const items = groupedEntities[category] || [];
              if (items.length === 0) return null;
              return (
                <div key={category}>
                  <div style={{ fontSize: '0.82rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                    {category}s ({items.length})
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                    {items.map((item, idx) => (
                      <span
                        key={idx}
                        className={`${styles.entityChip} ${styles[`chip_${category}`] || ''}`}
                        title={item.confidence ? `Confidence: ${Math.round(item.confidence * 100)}%` : undefined}
                      >
                        {item.text}
                        {item.negated && <span className={styles.negatedTag}>Denied</span>}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </GlassPanel>
      )}

      {/* g. Educational Recommendations */}
      {recs.length > 0 && (
        <GlassPanel>
          <h2 className={styles.riskClassHeading}>Preventive & Educational Health Guidance</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1rem' }}>
            Rule-based guidance derived from CDC, AHA, and WHO public prevention guidelines. Educational only, not medical advice.
          </p>

          {/* Uncertainty line above recommendations when reliability is low or predicted class is Medium */}
          {showMutedClass && (
            <div className={styles.recommendationUncertaintyNotice} role="note">
              This estimate is uncertain because many answers were assumed. Treat this as general guidance, not a conclusion.
            </div>
          )}

          <div>
            {recs.map((rec, i) => {
              const ruleId = typeof rec === 'object' ? rec.rule_id : null;
              const title = formatRuleTitle(ruleId) || `Guidance ${i + 1}`;
              const text = typeof rec === 'object' ? (rec.text || rec.guidance_text) : rec;
              const category = typeof rec === 'object' && rec.category ? formatCategory(rec.category) : null;

              return (
                <div key={i} className={styles.recCard}>
                  <div className={styles.recHeader}>
                    <h3 className={styles.recTitle}>{title}</h3>
                    {category && <span className={styles.recCategory}>{category}</span>}
                  </div>
                  <p className={styles.recText}>{text}</p>
                </div>
              );
            })}
          </div>
        </GlassPanel>
      )}
    </div>
  );
}
