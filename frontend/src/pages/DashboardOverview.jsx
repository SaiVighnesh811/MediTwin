import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import DashboardShell from '../components/layout/DashboardShell';
import GlassPanel from '../components/ui/GlassPanel';
import Button from '../components/ui/Button';
import AnalysisResultView from '../components/analysis/AnalysisResultView';
import { analysisAPI } from '../lib/api';

/**
 * Computes dashboard counter numbers from saved analysis documents.
 * - Comprehensive runs: type === 'analyze'
 * - Risk predictions: type === 'risk' OR analyze with risk_assessment present
 * - Imaging aids: type === 'image' OR analyze with an image screening result present
 * - Types 'risk', 'image', 'ocr', 'nlp' still count under their own meaning
 */
export function computeDashboardCounts(analyses = [], totalFallback = 0) {
  let comprehensive = 0;
  let riskPredictions = 0;
  let imagingAids = 0;

  for (const a of analyses) {
    const res = a.result || {};
    const type = a.type;

    if (type === 'analyze') {
      comprehensive += 1;
    }

    // Risk prediction count: standalone type === 'risk', or analyze with risk_assessment (or risk)
    const hasRisk =
      type === 'risk' ||
      Boolean(res.risk_assessment && (res.risk_assessment.predicted_class || Object.keys(res.risk_assessment).length > 0)) ||
      Boolean(res.risk && (res.risk.predicted_class || Object.keys(res.risk).length > 0));

    if (hasRisk) {
      riskPredictions += 1;
    }

    // Imaging aid count: standalone type === 'image', or analyze with imaging result
    const imgObj = res.imaging || (res.pneumonia_probability !== undefined || res.pneumonia_score !== undefined ? res : null);
    const hasImage =
      type === 'image' ||
      Boolean(
        imgObj &&
        (imgObj.model_used || imgObj.predicted_class || imgObj.pneumonia_score !== undefined || imgObj.pneumonia_probability !== undefined)
      );

    if (hasImage) {
      imagingAids += 1;
    }
  }

  return {
    total: totalFallback || analyses.length,
    comprehensive,
    riskPredictions,
    imagingAids,
  };
}

export default function DashboardOverview() {
  const [latestAnalysis, setLatestAnalysis] = useState(null);
  const [counts, setCounts] = useState({ total: 0, comprehensive: 0, riskPredictions: 0, imagingAids: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    analysisAPI
      .list(1, 50)
      .then((data) => {
        const list = data.analyses || [];
        setLatestAnalysis(list[0] || null);
        const computed = computeDashboardCounts(list, data.total);
        setCounts(computed);
      })
      .catch((err) => setError(err.message || 'Failed to load analysis history'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <DashboardShell>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {/* Top welcome banner */}
        <GlassPanel>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--accent-teal)', textTransform: 'uppercase', marginBottom: '4px' }}>
                // Patient Digital Twin Dashboard
              </p>
              <h1 style={{ fontFamily: 'var(--font-serif)', fontSize: 'clamp(1.5rem, 3vw, 2.2rem)', margin: '4px 0 8px 0' }}>
                Welcome to your clinical workspace
              </h1>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '600px', margin: 0 }}>
                Run multi-agent analyses, review saved results and read the explanations.
              </p>
            </div>
            <Button as={Link} to="/dashboard/new" variant="primary" size="md">
              + New Analysis
            </Button>
          </div>
        </GlassPanel>

        {/* Counts summary row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '24px' }}>
          <GlassPanel style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>TOTAL ANALYSES</div>
            <div style={{ fontFamily: 'var(--font-serif)', fontSize: '2rem', color: 'var(--text-primary)', marginTop: '8px' }}>
              {loading ? '…' : counts.total}
            </div>
          </GlassPanel>
          <GlassPanel style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>COMPREHENSIVE RUNS</div>
            <div style={{ fontFamily: 'var(--font-serif)', fontSize: '2rem', color: 'var(--accent-teal)', marginTop: '8px' }}>
              {loading ? '…' : counts.comprehensive}
            </div>
          </GlassPanel>
          <GlassPanel style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>RISK PREDICTIONS</div>
            <div style={{ fontFamily: 'var(--font-serif)', fontSize: '2rem', color: 'var(--accent-blue)', marginTop: '8px' }}>
              {loading ? '…' : counts.riskPredictions}
            </div>
          </GlassPanel>
          <GlassPanel style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>IMAGING AIDS</div>
            <div style={{ fontFamily: 'var(--font-serif)', fontSize: '2rem', color: 'var(--accent-amber)', marginTop: '8px' }}>
              {loading ? '…' : counts.imagingAids}
            </div>
          </GlassPanel>
        </div>

        {/* Latest Analysis Section */}
        <div>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.25rem', marginBottom: '16px' }}>
            Latest Saved Analysis
          </h2>
          {loading ? (
            <GlassPanel>
              <div style={{ textAlign: 'center', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', padding: '16px 0' }}>
                Loading analysis record…
              </div>
            </GlassPanel>
          ) : error ? (
            <GlassPanel>
              <div style={{ color: 'var(--accent-rose)' }}>{error}</div>
            </GlassPanel>
          ) : !latestAnalysis ? (
            <GlassPanel>
              <div style={{ textAlign: 'center', padding: '16px 0' }}>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '16px' }}>
                  No analyses recorded yet. Start your first digital twin evaluation.
                </p>
                <Button as={Link} to="/dashboard/new" variant="primary">
                  Start Analysis
                </Button>
              </div>
            </GlassPanel>
          ) : (
            <div>
              <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Recorded on {new Date(latestAnalysis.createdAt).toLocaleString()} · ID: {latestAnalysis._id}
                </span>
                <Link to={`/dashboard/analysis/${latestAnalysis._id}`} style={{ color: 'var(--accent-teal)', fontSize: '0.85rem' }}>
                  Full Details &rarr;
                </Link>
              </div>
              <AnalysisResultView data={latestAnalysis} />
            </div>
          )}
        </div>
      </div>
    </DashboardShell>
  );
}

