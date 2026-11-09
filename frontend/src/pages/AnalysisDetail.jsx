import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import DashboardShell from '../components/layout/DashboardShell';
import GlassPanel from '../components/ui/GlassPanel';
import Button from '../components/ui/Button';
import AnalysisResultView from '../components/analysis/AnalysisResultView';
import { analysisAPI } from '../lib/api';

export default function AnalysisDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError('');
    analysisAPI
      .get(id)
      .then((data) => setAnalysis(data))
      .catch((err) => {
        if (err.status === 404) {
          setError('Analysis not found or you do not have permission to access it.');
        } else {
          setError(err.message || 'Failed to load analysis');
        }
      })
      .finally(() => setLoading(false));
  }, [id]);

  async function handleDelete() {
    try {
      await analysisAPI.delete(id);
      navigate('/dashboard/history', { replace: true });
    } catch (err) {
      setError(err.message || 'Failed to delete analysis');
    }
  }

  return (
    <DashboardShell>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <Link to="/dashboard/history" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textDecoration: 'none' }}>
              &larr; Back to History
            </Link>
            <h1 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.8rem', margin: '0.3rem 0 0 0' }}>
              Analysis Record Details
            </h1>
            {analysis && (
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
                ID: {analysis._id} · Recorded: {new Date(analysis.createdAt).toLocaleString()}
              </div>
            )}
          </div>

          {analysis && (
            <div>
              {confirmDelete ? (
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <Button variant="danger" size="sm" onClick={handleDelete}>
                    Confirm Delete
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(false)}>
                    Cancel
                  </Button>
                </div>
              ) : (
                <Button variant="outline" size="sm" onClick={() => setConfirmDelete(true)}>
                  Delete Record
                </Button>
              )}
            </div>
          )}
        </div>

        {loading ? (
          <GlassPanel>
            <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
              Loading analysis details…
            </div>
          </GlassPanel>
        ) : error ? (
          <GlassPanel>
            <div style={{ padding: '2rem', textAlign: 'center' }}>
              <p style={{ color: 'var(--accent-rose)', marginBottom: '1rem' }}>{error}</p>
              <Button as={Link} to="/dashboard/history" variant="outline">
                Return to History
              </Button>
            </div>
          </GlassPanel>
        ) : (
          <AnalysisResultView data={analysis} />
        )}
      </div>
    </DashboardShell>
  );
}

