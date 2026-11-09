import { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import DigitalTwin from '../components/DigitalTwin';
import Button from '../components/ui/Button';
import styles from './AuthForm.module.css';

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({ email: '', password: '' });
  const [errors, setErrors] = useState({});
  const [serverError, setServerError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPw, setShowPw] = useState(false);

  const emailRef = useRef(null);
  const pwRef = useRef(null);

  function validate() {
    const e = {};
    if (!form.email.trim()) e.email = 'Email is required';
    else if (!EMAIL_RE.test(form.email)) e.email = 'Enter a valid email address';
    if (!form.password) e.password = 'Password is required';
    return e;
  }

  async function handleSubmit(ev) {
    ev.preventDefault();
    setServerError('');
    const v = validate();
    setErrors(v);
    if (Object.keys(v).length) {
      (v.email ? emailRef : pwRef).current?.focus();
      return;
    }
    setLoading(true);
    try {
      await login({ email: form.email.trim().toLowerCase(), password: form.password });
      navigate('/dashboard', { replace: true });
    } catch (err) {
      setServerError(err.message || 'Login failed');
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
            Welcome back to <em>MediTwin</em>
          </h2>
          <p className={styles.panelSub}>
            Decision-support prototype. Not a medical diagnosis.
          </p>
        </div>
      </div>

      {/* Right form */}
      <div className={styles.formSide}>
        <div className={styles.formContainer}>
          <h1 className={styles.formTitle}>Log in</h1>
          <p className={styles.formSub}>
            Don&apos;t have an account?{' '}
            <Link to="/signup">Sign up</Link>
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
            {/* Email */}
            <div className={styles.field}>
              <input
                ref={emailRef}
                id="login-email"
                type="email"
                className={`${styles.input} ${errors.email ? styles.hasError : ''}`}
                placeholder=" "
                value={form.email}
                onChange={onChange('email')}
                autoComplete="email"
                aria-invalid={!!errors.email}
                aria-describedby={errors.email ? 'login-email-err' : undefined}
              />
              <label htmlFor="login-email" className={styles.floatLabel}>Email</label>
              {errors.email && (
                <p id="login-email-err" className={styles.fieldError} role="alert">
                  {errors.email}
                </p>
              )}
            </div>

            {/* Password */}
            <div className={styles.field}>
              <input
                ref={pwRef}
                id="login-password"
                type={showPw ? 'text' : 'password'}
                className={`${styles.input} ${errors.password ? styles.hasError : ''}`}
                placeholder=" "
                value={form.password}
                onChange={onChange('password')}
                autoComplete="current-password"
                aria-invalid={!!errors.password}
                aria-describedby={errors.password ? 'login-pw-err' : undefined}
              />
              <label htmlFor="login-password" className={styles.floatLabel}>Password</label>
              <button
                type="button"
                className={styles.pwToggle}
                onClick={() => setShowPw((v) => !v)}
                aria-label={showPw ? 'Hide password' : 'Show password'}
              >
                {showPw ? 'HIDE' : 'SHOW'}
              </button>
              {errors.password && (
                <p id="login-pw-err" className={styles.fieldError} role="alert">
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
              {loading ? 'Signing in…' : 'Log in'}
            </Button>
          </form>

          <Link to="/" className={styles.backLink}>&larr; Back to home</Link>
        </div>
      </div>
    </div>
  );
}
