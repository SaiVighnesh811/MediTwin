import { useState, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import DigitalTwin from '../components/DigitalTwin';
import Button from '../components/ui/Button';
import styles from './AuthForm.module.css';

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const PW_LETTER = /[A-Za-z]/;
const PW_DIGIT = /\d/;

function passwordStrength(pw) {
  let s = 0;
  if (pw.length >= 8) s++;
  if (PW_LETTER.test(pw) && PW_DIGIT.test(pw)) s++;
  if (pw.length >= 12 && /[^A-Za-z0-9]/.test(pw)) s++;
  return s; // 0-3
}

const STRENGTH_LABELS = ['', 'Weak', 'Fair', 'Strong'];

export default function Signup() {
  const { signup } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({ name: '', email: '', password: '' });
  const [errors, setErrors] = useState({});
  const [serverError, setServerError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPw, setShowPw] = useState(false);

  const nameRef = useRef(null);
  const emailRef = useRef(null);
  const pwRef = useRef(null);

  const strength = passwordStrength(form.password);

  function validate() {
    const e = {};
    if (!form.name.trim()) e.name = 'Name is required';
    if (!form.email.trim()) e.email = 'Email is required';
    else if (!EMAIL_RE.test(form.email)) e.email = 'Enter a valid email address';
    if (!form.password) e.password = 'Password is required';
    else if (form.password.length < 8) e.password = 'At least 8 characters';
    else if (!PW_LETTER.test(form.password) || !PW_DIGIT.test(form.password))
      e.password = 'Must include at least one letter and one number';
    return e;
  }

  async function handleSubmit(ev) {
    ev.preventDefault();
    setServerError('');
    const v = validate();
    setErrors(v);
    if (Object.keys(v).length) {
      const first = v.name ? nameRef : v.email ? emailRef : pwRef;
      first.current?.focus();
      return;
    }
    setLoading(true);
    try {
      await signup({
        name: form.name.trim(),
        email: form.email.trim().toLowerCase(),
        password: form.password,
      });
      navigate('/dashboard', { replace: true });
    } catch (err) {
      setServerError(err.message || 'Signup failed');
    } finally {
      setLoading(false);
    }
  }

  function onChange(field) {
    return (e) => {
      setForm((f) => ({ ...f, [field]: e.target.value }));
      if (errors[field]) setErrors((prev) => ({ ...prev, [field]: '' }));
    };
  }

  return (
    <div className={styles.page}>
      {/* Left decorative panel */}
      <div className={styles.panel} aria-hidden="true">
        <div className={styles.panelCanvas}>
          <DigitalTwin />
        </div>
        <div className={styles.panelOverlay}>
          <h2 className={styles.panelHeading}>
            Create your <em>digital twin</em>
          </h2>
          <p className={styles.panelSub}>
            Decision-support prototype. Not a medical diagnosis.
          </p>
        </div>
      </div>

      {/* Right form */}
      <div className={styles.formSide}>
        <div className={styles.formContainer}>
          <h1 className={styles.formTitle}>Create account</h1>
          <p className={styles.formSub}>
            Already have an account?{' '}
            <Link to="/login">Log in</Link>
          </p>

          <AnimatePresence>
            {serverError && (
              <motion.div
                className={styles.formError}
                role="alert"
                aria-live="assertive"
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2 }}
              >
                {serverError}
              </motion.div>
            )}
          </AnimatePresence>

          <form onSubmit={handleSubmit} noValidate>
            {/* Name */}
            <div className={styles.field}>
              <input
                ref={nameRef}
                id="signup-name"
                type="text"
                className={`${styles.input} ${errors.name ? styles.hasError : ''}`}
                placeholder=" "
                value={form.name}
                onChange={onChange('name')}
                autoComplete="name"
                aria-invalid={!!errors.name}
                aria-describedby={errors.name ? 'signup-name-err' : undefined}
              />
              <label htmlFor="signup-name" className={styles.floatLabel}>Full name</label>
              {errors.name && (
                <p id="signup-name-err" className={styles.fieldError} role="alert">
                  {errors.name}
                </p>
              )}
            </div>

            {/* Email */}
            <div className={styles.field}>
              <input
                ref={emailRef}
                id="signup-email"
                type="email"
                className={`${styles.input} ${errors.email ? styles.hasError : ''}`}
                placeholder=" "
                value={form.email}
                onChange={onChange('email')}
                autoComplete="email"
                aria-invalid={!!errors.email}
                aria-describedby={errors.email ? 'signup-email-err' : undefined}
              />
              <label htmlFor="signup-email" className={styles.floatLabel}>Email</label>
              {errors.email && (
                <p id="signup-email-err" className={styles.fieldError} role="alert">
                  {errors.email}
                </p>
              )}
            </div>

            {/* Password */}
            <div className={styles.field}>
              <input
                ref={pwRef}
                id="signup-password"
                type={showPw ? 'text' : 'password'}
                className={`${styles.input} ${errors.password ? styles.hasError : ''}`}
                placeholder=" "
                value={form.password}
                onChange={onChange('password')}
                autoComplete="new-password"
                aria-invalid={!!errors.password}
                aria-describedby="signup-pw-strength signup-pw-err"
              />
              <label htmlFor="signup-password" className={styles.floatLabel}>Password</label>
              <button
                type="button"
                className={styles.pwToggle}
                onClick={() => setShowPw((v) => !v)}
                aria-label={showPw ? 'Hide password' : 'Show password'}
              >
                {showPw ? 'HIDE' : 'SHOW'}
              </button>

              {/* Strength meter */}
              {form.password.length > 0 && (
                <>
                  <div className={styles.strengthBar} id="signup-pw-strength" aria-label={`Password strength: ${STRENGTH_LABELS[strength]}`}>
                    {[1, 2, 3].map((level) => (
                      <div
                        key={level}
                        className={`${styles.strengthSeg} ${strength >= level ? styles[`active${level}`] : ''}`}
                      />
                    ))}
                  </div>
                  <span className={styles.strengthLabel}>{STRENGTH_LABELS[strength]}</span>
                </>
              )}

              {errors.password && (
                <p id="signup-pw-err" className={styles.fieldError} role="alert">
                  {errors.password}
                </p>
              )}
            </div>

            <Button
              type="submit"
              className={styles.submitBtn}
              disabled={loading}
              size="lg"
            >
              {loading ? 'Creating account…' : 'Create account'}
            </Button>
          </form>

          <Link to="/" className={styles.backLink}>&larr; Back to home</Link>
        </div>
      </div>
    </div>
  );
}
