import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardShell from '../components/layout/DashboardShell';
import GlassPanel from '../components/ui/GlassPanel';
import Button from '../components/ui/Button';
import {
  mapAgeToBand,
  computeBMI,
  GEN_HLTH_OPTIONS,
  SEX_OPTIONS,
  EDUCATION_OPTIONS,
  INCOME_OPTIONS,
  BINARY_FIELDS,
  buildBRFSSPayload,
} from '../lib/brfss';
import { analysisAPI } from '../lib/api';
import styles from './Wizard.module.css';

const MAX_IMAGE_SIZE = 8 * 1024 * 1024; // 8 MB

export default function Wizard() {
  const navigate = useNavigate();
  const [mode, setMode] = useState('full'); // 'full' | 'xray_only'
  const [step, setStep] = useState(1);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');

  // Step 1 State: Form answers (Full mode)
  const [formData, setFormData] = useState({
    age: '',
    sex: '',
    height: '',
    weight: '',
    genHlth: '',
    mentHlth: '',
    physHlth: '',
    education: '',
    income: '',
  });

  // X-ray-only mode age state (0–120)
  const [xrayAge, setXrayAge] = useState('');

  // Files & Text
  const [docFile, setDocFile] = useState(null);
  const [xrayFile, setXrayFile] = useState(null);
  const [clinicalText, setClinicalText] = useState('');
  const [uploadError, setUploadError] = useState('');

  // Timer while running
  useEffect(() => {
    let interval = null;
    if (running) {
      setElapsed(0);
      interval = setInterval(() => setElapsed((prev) => prev + 1), 1000);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [running]);

  // Derived BMI
  const computedBMI = computeBMI(formData.height, formData.weight);

  // Field change helper
  function handleFieldChange(name, value) {
    setFormData((prev) => ({ ...prev, [name]: value }));
  }

  // File validator
  function handleFileSelect(file, type) {
    setUploadError('');
    if (!file) return;

    const allowed = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!allowed.includes(file.type.toLowerCase())) {
      setUploadError('Only JPEG and PNG images are supported.');
      return;
    }
    if (file.size > MAX_IMAGE_SIZE) {
      setUploadError('File size exceeds the 8 MB limit.');
      return;
    }

    if (type === 'doc') setDocFile(file);
    if (type === 'xray') setXrayFile(file);
  }

  // Step 1 Validation
  const isStep1Valid =
    formData.age !== '' &&
    mapAgeToBand(Number(formData.age)) !== null &&
    (formData.sex === 0 || formData.sex === 1 || formData.sex === '0' || formData.sex === '1') &&
    computedBMI !== null &&
    formData.genHlth !== '';

  // Submit Handler for Full Analysis
  async function handleRunAnalysis() {
    setErrorMsg('');

    // Block submission if an X-ray is attached but no age was entered
    // (age is required to pick the correct X-ray model: pediatric vs adult)
    if (xrayFile && (formData.age === '' || Number.isNaN(Number(formData.age)))) {
      setErrorMsg(
        'Enter the patient\u2019s age so the right X-ray model can be selected (pediatric \u003c 18 years, adult \u2265 18 years).',
      );
      return;
    }

    setRunning(true);

    try {
      const brfssPayload = buildBRFSSPayload(formData);
      const fd = new FormData();

      if (Object.keys(brfssPayload).length > 0) {
        fd.append('features', JSON.stringify(brfssPayload));
      }
      if (clinicalText.trim()) {
        fd.append('text', clinicalText.trim());
      }
      if (docFile) {
        fd.append('document_image', docFile);
      }
      if (xrayFile) {
        fd.append('xray_image', xrayFile);
        // Forward the raw numeric age so the ML service can route to the correct model
        fd.append('age_years', String(Math.round(Number(formData.age))));
      }

      const res = await analysisAPI.analyze(fd);
      // Navigate to saved result view
      if (res.analysisId) {
        navigate(`/dashboard/analysis/${res.analysisId}`);
      } else {
        navigate('/dashboard/history');
      }
    } catch (err) {
      setErrorMsg(err.message || 'Analysis failed. Please try again.');
      setRunning(false);
    }
  }

  // Submit Handler for Chest X-ray Only Mode (Any Age)
  async function handleRunXrayOnly() {
    setErrorMsg('');

    if (xrayAge === '' || Number.isNaN(Number(xrayAge))) {
      setErrorMsg('Please enter the patient\u2019s age in years (0 to 120).');
      return;
    }
    const ageNum = parseInt(xrayAge, 10);
    if (ageNum < 0 || ageNum > 120) {
      setErrorMsg('Patient age must be between 0 and 120 years.');
      return;
    }
    if (!xrayFile) {
      setErrorMsg('Please upload a chest X-ray image (JPEG or PNG).');
      return;
    }

    setRunning(true);
    try {
      const fd = new FormData();
      fd.append('file', xrayFile);
      fd.append('age_years', String(ageNum));

      const res = await analysisAPI.image(fd);
      if (res.analysisId) {
        navigate(`/dashboard/analysis/${res.analysisId}`);
      } else {
        navigate('/dashboard/history');
      }
    } catch (err) {
      setErrorMsg(err.message || 'Image screening failed. Please try again.');
      setRunning(false);
    }
  }

  return (
    <DashboardShell>
      <div className={styles.wizardContainer}>
        {/* Mode Selector */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
          <button
            type="button"
            onClick={() => { setMode('full'); setErrorMsg(''); }}
            style={{
              padding: '16px',
              borderRadius: 'var(--radius-md)',
              border: `2px solid ${mode === 'full' ? 'var(--accent-teal)' : 'var(--border-hairline)'}`,
              background: mode === 'full' ? 'color-mix(in srgb, var(--accent-teal) 10%, transparent)' : 'var(--bg-surface-elevated)',
              color: 'var(--text-primary)',
              textAlign: 'left',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            <div style={{ fontWeight: 600, fontSize: '0.98rem', color: mode === 'full' ? 'var(--accent-teal)' : 'var(--text-primary)', marginBottom: '4px' }}>
              Full analysis (adult questionnaire)
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
              Multi-agent digital twin: health risk prediction, SHAP/LIME explainability, recommendations, and optional documents or X-rays.
            </div>
          </button>

          <button
            type="button"
            onClick={() => { setMode('xray_only'); setErrorMsg(''); }}
            style={{
              padding: '16px',
              borderRadius: 'var(--radius-md)',
              border: `2px solid ${mode === 'xray_only' ? 'var(--accent-teal)' : 'var(--border-hairline)'}`,
              background: mode === 'xray_only' ? 'color-mix(in srgb, var(--accent-teal) 10%, transparent)' : 'var(--bg-surface-elevated)',
              color: 'var(--text-primary)',
              textAlign: 'left',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            <div style={{ fontWeight: 600, fontSize: '0.98rem', color: mode === 'xray_only' ? 'var(--accent-teal)' : 'var(--text-primary)', marginBottom: '4px' }}>
              Chest X-ray only (any age)
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
              Targeted pneumonia screening aid for children and adults (age 0–120). Skips the adult survey.
            </div>
          </button>
        </div>

        {/* Stepper Header (shown only in full analysis mode) */}
        {mode === 'full' && (
          <div className={styles.stepper} aria-label="Wizard progress">
            <div className={`${styles.stepItem} ${step === 1 ? styles.stepActive : ''}`}>
              <span className={styles.stepCircle}>1</span> About You
            </div>
            <div className={`${styles.stepItem} ${step === 2 ? styles.stepActive : ''}`}>
              <span className={styles.stepCircle}>2</span> Documents & Images
            </div>
            <div className={`${styles.stepItem} ${step === 3 ? styles.stepActive : ''}`}>
              <span className={styles.stepCircle}>3</span> Review & Run
            </div>
          </div>
        )}

        {errorMsg && (
          <div
            style={{
              padding: '0.75rem 1rem',
              background: 'rgba(244, 63, 94, 0.15)',
              border: '1px solid var(--accent-rose)',
              color: 'var(--accent-rose)',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.88rem',
            }}
            role="alert"
          >
            {errorMsg}
          </div>
        )}

        {/* Running Progress View */}
        {running ? (
          <GlassPanel>
            <div className={styles.runningBox}>
              <div className={styles.timer}>{elapsed}s</div>
              <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.4rem' }}>
                {mode === 'xray_only' ? 'Analyzing Chest X-Ray…' : 'Analyzing Clinical Profile…'}
              </h2>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '500px', fontSize: '0.9rem' }}>
                Executing inference with sequential model locks. Image bytes are processed in memory and discarded after evaluation.
              </p>
            </div>
          </GlassPanel>
        ) : mode === 'xray_only' ? (
          /* X-RAY ONLY MODE */
          <GlassPanel>
            <h2 style={{ fontFamily: 'var(--font-serif)', marginBottom: '8px' }}>
              Chest X-Ray Screening (Any Age)
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '24px' }}>
              Screens a chest X-ray image for pneumonia indicators. Ages under 18 route to the pediatric Kermany ResNet18 model; ages 18 and over route to the adult TorchXRayVision DenseNet121 model.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* Age input */}
              <div className={styles.formGroup} style={{ maxWidth: '320px' }}>
                <label className={styles.label} htmlFor="input-xray-age">
                  Patient Age in Years (0 – 120)<span className={styles.required}>*</span>
                </label>
                <input
                  id="input-xray-age"
                  type="number"
                  min="0"
                  max="120"
                  className={styles.input}
                  placeholder="e.g. 8 or 45"
                  value={xrayAge}
                  onChange={(e) => setXrayAge(e.target.value)}
                />
                <span className={styles.helper}>
                  {xrayAge !== '' && !Number.isNaN(Number(xrayAge))
                    ? Number(xrayAge) < 18
                      ? 'Pediatric route (< 18 yrs): Kermany ResNet18 model'
                      : 'Adult route (≥ 18 yrs): TorchXRayVision DenseNet121 model'
                    : 'Required to select the correct screening model (0 to 120 years)'}
                </span>
              </div>

              {/* X-Ray File Upload */}
              <div className={styles.formGroup}>
                <label className={styles.label}>
                  Chest X-Ray Image (JPEG/PNG, max 8MB)<span className={styles.required}>*</span>
                </label>
                {!xrayFile ? (
                  <label className={styles.dropzone}>
                    <input
                      type="file"
                      accept="image/jpeg,image/png"
                      style={{ display: 'none' }}
                      onChange={(e) => handleFileSelect(e.target.files[0], 'xray')}
                    />
                    <span style={{ fontSize: '1.5rem' }}>🩻</span>
                    <span style={{ fontWeight: 500 }}>Click or drag &amp; drop chest X-ray image</span>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)' }}>
                      JPEG or PNG only, up to 8 MB
                    </span>
                  </label>
                ) : (
                  <div className={styles.filePreview}>
                    <span style={{ fontSize: '0.85rem' }}>
                      {xrayFile.name} ({(xrayFile.size / 1024).toFixed(0)} KB)
                    </span>
                    <Button variant="outline" size="sm" onClick={() => setXrayFile(null)}>
                      Remove
                    </Button>
                  </div>
                )}
                {uploadError && (
                  <span style={{ color: 'var(--accent-rose)', fontSize: '0.8rem', marginTop: '0.25rem' }}>
                    {uploadError}
                  </span>
                )}
              </div>

              <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-surface)', borderLeft: '3px solid var(--accent-teal)', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                <strong>Privacy &amp; Retention:</strong> Uploaded images are analyzed in memory and immediately deleted from disk. Only computed numerical scores and resulting screening indicators are saved in your account.
              </div>

              <div className={styles.actionFooter}>
                <div />
                <Button variant="primary" onClick={handleRunXrayOnly}>
                  Run Chest X-Ray Screening &rarr;
                </Button>
              </div>
            </div>
          </GlassPanel>
        ) : (
          <AnimatePresence mode="wait">
            {/* STEP 1: ABOUT YOU */}
            {step === 1 && (
              <motion.div
                key="step1"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
              >
                <GlassPanel>
                  <h2 style={{ fontFamily: 'var(--font-serif)', marginBottom: '0.5rem' }}>
                    1. Patient Health Profile
                  </h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                    Provide known survey attributes. Skipped answers are never forced; the model will assume typical population medians.
                  </p>

                  <div className={styles.formGrid}>
                    {/* Required: Age */}
                    <div className={styles.formGroup}>
                      <label className={styles.label} htmlFor="input-age">
                        Age (Years)<span className={styles.required}>*</span>
                      </label>
                      <input
                        id="input-age"
                        type="number"
                        min="18"
                        max="120"
                        className={styles.input}
                        placeholder="18 – 120"
                        value={formData.age}
                        onChange={(e) => handleFieldChange('age', e.target.value)}
                      />
                      <span className={styles.helper}>Must be between 18 and 120</span>
                    </div>

                    {/* Required: Sex */}
                    <div className={styles.formGroup}>
                      <label className={styles.label} htmlFor="select-sex">
                        Biological Sex<span className={styles.required}>*</span>
                      </label>
                      <select
                        id="select-sex"
                        className={styles.select}
                        value={formData.sex}
                        onChange={(e) => handleFieldChange('sex', e.target.value)}
                      >
                        <option value="">Select…</option>
                        {SEX_OPTIONS.map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Required: Height */}
                    <div className={styles.formGroup}>
                      <label className={styles.label} htmlFor="input-height">
                        Height (cm)<span className={styles.required}>*</span>
                      </label>
                      <input
                        id="input-height"
                        type="number"
                        min="100"
                        max="250"
                        className={styles.input}
                        placeholder="e.g. 175"
                        value={formData.height}
                        onChange={(e) => handleFieldChange('height', e.target.value)}
                      />
                    </div>

                    {/* Required: Weight */}
                    <div className={styles.formGroup}>
                      <label className={styles.label} htmlFor="input-weight">
                        Weight (kg)<span className={styles.required}>*</span>
                      </label>
                      <input
                        id="input-weight"
                        type="number"
                        min="25"
                        max="300"
                        className={styles.input}
                        placeholder="e.g. 70"
                        value={formData.weight}
                        onChange={(e) => handleFieldChange('weight', e.target.value)}
                      />
                      {computedBMI !== null ? (
                        <span className={styles.helper} style={{ color: 'var(--accent-teal)' }}>
                          Computed BMI: <strong>{computedBMI}</strong>
                        </span>
                      ) : (
                        <span className={styles.helper}>BMI shown once both height & weight are valid</span>
                      )}
                    </div>

                    {/* Required: General Health */}
                    <div className={styles.formGroup}>
                      <label className={styles.label} htmlFor="select-genhlth">
                        General Health Rating<span className={styles.required}>*</span>
                      </label>
                      <select
                        id="select-genhlth"
                        className={styles.select}
                        value={formData.genHlth}
                        onChange={(e) => handleFieldChange('genHlth', e.target.value)}
                      >
                        <option value="">Select general health…</option>
                        {GEN_HLTH_OPTIONS.map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Optional: Education */}
                    <div className={styles.formGroup}>
                      <label className={styles.label} htmlFor="select-education">
                        Education Level (Optional)
                      </label>
                      <select
                        id="select-education"
                        className={styles.select}
                        value={formData.education}
                        onChange={(e) => handleFieldChange('education', e.target.value)}
                      >
                        <option value="">Not specified / skip</option>
                        {EDUCATION_OPTIONS.map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Optional: Income */}
                    <div className={styles.formGroup}>
                      <label className={styles.label} htmlFor="select-income">
                        Income Band (US Survey Bands, Optional)
                      </label>
                      <select
                        id="select-income"
                        className={styles.select}
                        value={formData.income}
                        onChange={(e) => handleFieldChange('income', e.target.value)}
                      >
                        <option value="">Not specified / skip</option>
                        {INCOME_OPTIONS.map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <hr style={{ borderColor: 'var(--border-hairline)', margin: '1.5rem 0' }} />

                  <h3 style={{ fontSize: '0.95rem', fontFamily: 'var(--font-mono)', marginBottom: '1rem' }}>
                    Health & Lifestyle Indicators (Optional 3-state)
                  </h3>

                  <div className={styles.formGrid}>
                    {BINARY_FIELDS.map((f) => {
                      const currVal = formData[f.key];
                      return (
                        <div key={f.key} className={styles.formGroup}>
                          <span className={styles.label}>{f.label}</span>
                          <div className={styles.triStateGroup}>
                            <button
                              type="button"
                              className={`${styles.triStateBtn} ${currVal === 1 ? styles.triSelectedYes : ''}`}
                              onClick={() => handleFieldChange(f.key, 1)}
                            >
                              Yes
                            </button>
                            <button
                              type="button"
                              className={`${styles.triStateBtn} ${currVal === 0 ? styles.triSelectedNo : ''}`}
                              onClick={() => handleFieldChange(f.key, 0)}
                            >
                              No
                            </button>
                            <button
                              type="button"
                              className={`${styles.triStateBtn} ${currVal === undefined || currVal === null ? styles.triSelectedSkip : ''}`}
                              onClick={() => handleFieldChange(f.key, null)}
                            >
                              Skip
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className={styles.actionFooter}>
                    <div />
                    <Button
                      variant="primary"
                      disabled={!isStep1Valid}
                      onClick={() => setStep(2)}
                    >
                      Next: Documents & Images &rarr;
                    </Button>
                  </div>
                </GlassPanel>
              </motion.div>
            )}

            {/* STEP 2: DOCUMENTS & IMAGES */}
            {step === 2 && (
              <motion.div
                key="step2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
              >
                <GlassPanel>
                  <h2 style={{ fontFamily: 'var(--font-serif)', marginBottom: '0.5rem' }}>
                    2. Documents, Notes & Imaging
                  </h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                    All inputs on this step are completely optional. Upload prescription photos or clinical notes to extract dated history into patient memory.
                  </p>

                  {uploadError && (
                    <div
                      style={{
                        padding: '0.6rem 0.8rem',
                        background: 'rgba(244, 63, 94, 0.1)',
                        border: '1px solid var(--accent-rose)',
                        color: 'var(--accent-rose)',
                        borderRadius: 'var(--radius-sm)',
                        marginBottom: '1rem',
                        fontSize: '0.82rem',
                      }}
                    >
                      {uploadError}
                    </div>
                  )}

                  {/* Document upload */}
                  <div style={{ marginBottom: '1.5rem' }}>
                    <label className={styles.label}>Prescription or Lab Report Image (JPEG/PNG, max 8MB)</label>
                    {!docFile ? (
                      <label className={styles.dropzone}>
                        <input
                          type="file"
                          accept="image/jpeg,image/png"
                          style={{ display: 'none' }}
                          onChange={(e) => handleFileSelect(e.target.files[0], 'doc')}
                        />
                        <span>Click or drag & drop prescription/lab image</span>
                        <small style={{ color: 'var(--text-tertiary)' }}>Printed text & handwritten medicine classifier</small>
                      </label>
                    ) : (
                      <div className={styles.filePreview}>
                        <span style={{ fontSize: '0.85rem' }}>{docFile.name} ({(docFile.size / 1024).toFixed(0)} KB)</span>
                        <Button variant="outline" size="sm" onClick={() => setDocFile(null)}>
                          Remove
                        </Button>
                      </div>
                    )}
                  </div>

                  {/* X-Ray upload */}
                  <div style={{ marginBottom: '1.5rem' }}>
                    <label className={styles.label}>Chest X-Ray Image (JPEG/PNG, max 8MB)</label>
                    <p style={{ fontSize: '0.78rem', color: 'var(--accent-amber)', margin: '0.2rem 0 0.5rem 0' }}>
                      Screening aid only. Age &lt; 18 routes to pediatric Kermany ResNet18; age &ge; 18 routes to adult TorchXRayVision DenseNet121. Not for clinical diagnostic use.
                    </p>
                    {!xrayFile ? (
                      <label className={styles.dropzone}>
                        <input
                          type="file"
                          accept="image/jpeg,image/png"
                          style={{ display: 'none' }}
                          onChange={(e) => handleFileSelect(e.target.files[0], 'xray')}
                        />
                        <span>Click or drag & drop chest X-ray image</span>
                        <small style={{ color: 'var(--text-tertiary)' }}>Pneumonia screening (age-routed model)</small>
                      </label>
                    ) : (
                      <div className={styles.filePreview}>
                        <span style={{ fontSize: '0.85rem' }}>{xrayFile.name} ({(xrayFile.size / 1024).toFixed(0)} KB)</span>
                        <Button variant="outline" size="sm" onClick={() => setXrayFile(null)}>
                          Remove
                        </Button>
                      </div>
                    )}
                  </div>

                  {/* Clinical note text */}
                  <div className={styles.formGroup}>
                    <label className={styles.label} htmlFor="textarea-notes">
                      Pasted Clinical Notes or Medical History (Max 20,000 characters)
                    </label>
                    <textarea
                      id="textarea-notes"
                      rows={5}
                      maxLength={20000}
                      className={styles.textarea}
                      placeholder="e.g. Patient presents with elevated BP 140/90, taking Amlodipine 5mg daily..."
                      value={clinicalText}
                      onChange={(e) => setClinicalText(e.target.value)}
                    />
                    <span className={styles.helper}>{clinicalText.length} / 20,000 characters</span>
                  </div>

                  <div className={styles.actionFooter}>
                    <Button variant="outline" onClick={() => setStep(1)}>
                      &larr; Back
                    </Button>
                    <Button variant="primary" onClick={() => setStep(3)}>
                      Next: Review & Run &rarr;
                    </Button>
                  </div>
                </GlassPanel>
              </motion.div>
            )}

            {/* STEP 3: REVIEW & RUN */}
            {step === 3 && (
              <motion.div
                key="step3"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
              >
                <GlassPanel>
                  <h2 style={{ fontFamily: 'var(--font-serif)', marginBottom: '0.5rem' }}>
                    3. Review & Run Analysis
                  </h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                    Check your inputs before submitting. Only statistical indicators and outcomes are stored; image bytes are discarded after inference.
                  </p>

                  <div style={{ background: 'var(--bg-surface-elevated)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', marginBottom: '1.5rem', lineHeight: '1.8', fontSize: '0.9rem' }}>
                    <div><strong>Age:</strong> {formData.age} years (BRFSS Age Band {mapAgeToBand(Number(formData.age))})</div>
                    <div><strong>Sex:</strong> {Number(formData.sex) === 1 ? 'Male' : 'Female'}</div>
                    <div><strong>Height & Weight:</strong> {formData.height} cm, {formData.weight} kg (BMI {computedBMI})</div>
                    <div><strong>General Health:</strong> {GEN_HLTH_OPTIONS.find((o) => o.value === Number(formData.genHlth))?.label}</div>
                    <div>
                      <strong>Survey Indicators Provided:</strong>{' '}
                      {Object.keys(buildBRFSSPayload(formData)).length} of 21 survey features provided
                    </div>
                    <div><strong>Document Uploaded:</strong> {docFile ? docFile.name : 'None'}</div>
                    <div><strong>X-Ray Uploaded:</strong> {xrayFile ? xrayFile.name : 'None'}</div>
                    <div><strong>Clinical Note:</strong> {clinicalText.trim() ? `${clinicalText.length} characters` : 'None'}</div>
                  </div>

                  <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-surface)', borderLeft: '3px solid var(--accent-teal)', fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
                    <strong>Privacy & Retention:</strong> Uploaded images are analyzed in memory and immediately deleted from disk. Only your computed parameters and resulting risk assessment are preserved in your personal patient memory record.
                  </div>

                  <div className={styles.actionFooter}>
                    <Button variant="outline" onClick={() => setStep(2)}>
                      &larr; Back
                    </Button>
                    <Button variant="primary" onClick={handleRunAnalysis}>
                      Run Analysis &rarr;
                    </Button>
                  </div>
                </GlassPanel>
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </div>
    </DashboardShell>
  );
}

