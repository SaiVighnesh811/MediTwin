import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import DashboardShell from '../components/layout/DashboardShell';
import GlassPanel from '../components/ui/GlassPanel';
import Button from '../components/ui/Button';
import { analysisAPI } from '../lib/api';

export default function AnalysisHistory() {
  const [analyses, setAnalyses] = useState([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);

  function loadHistory(pageNum = 1) {
    setLoading(true);
    setError('');
    analysisAPI
      .list(pageNum, 15)
      .then((data) => {
        setAnalyses(data.analyses || []);
        setPage(data.page || 1);
        setTotalPages(data.pages || 1);
      })
      .catch((err) => setError(err.message || 'Failed to load analysis history'))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    loadHistory(page);
  }, [page]);

  async function handleDelete(id) {
    try {
      await analysisAPI.delete(id);
      setDeleteConfirmId(null);
      loadHistory(page);
    } catch (err) {
      setError(err.message || 'Failed to delete record');
    }
  }

  return (
    <DashboardShell>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h1 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.8rem', margin: 0 }}>
              Analysis History
            </h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: '0.2rem 0 0 0' }}>
              Historical clinical runs, imaging screenings, and diagnostic extractions.
            </p>
          </div>
          <Button as={Link} to="/dashboard/new" variant="primary" size="sm">
            + Run New Case
          </Button>
        </div>

        {error && (
          <div style={{ padding: '0.75rem', background: 'rgba(244, 63, 94, 0.1)', color: 'var(--accent-rose)', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem' }}>
            {error}
          </div>
        )}

        {loading ? (
          <GlassPanel>
            <div style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
              Loading historical entries…
            </div>
          </GlassPanel>
        ) : analyses.length === 0 ? (
          <GlassPanel>
            <div style={{ padding: '3rem 1.5rem', textAlign: 'center' }}>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                No analysis records saved yet.
              </p>
              <Button as={Link} to="/dashboard/new" variant="primary">
                Run Your First Analysis
              </Button>
            </div>
          </GlassPanel>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {analyses.map((item) => {
              const res = item.result || {};
              const riskCls = res.predicted_class || res.risk_assessment?.predicted_class || item.type;
              const dateStr = new Date(item.createdAt).toLocaleString();

              return (
                <GlassPanel key={item._id}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.75rem',
                            padding: '0.2rem 0.5rem',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-surface-elevated)',
                            color: 'var(--accent-teal)',
                            border: '1px solid var(--border-hairline)',
                            textTransform: 'uppercase',
                          }}
                        >
                          {item.type}
                        </span>
                        <strong style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem' }}>
                          Finding / Category: {riskCls}
                        </strong>
                      </div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.4rem' }}>
                        ID: {item._id} · Created: {dateStr}
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Button as={Link} to={`/dashboard/analysis/${item._id}`} variant="outline" size="sm">
                        View Details
                      </Button>

                      {deleteConfirmId === item._id ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                          <Button variant="danger" size="sm" onClick={() => handleDelete(item._id)}>
                            Confirm Delete
                          </Button>
                          <Button variant="ghost" size="sm" onClick={() => setDeleteConfirmId(null)}>
                            Cancel
                          </Button>
                        </div>
                      ) : (
                        <Button variant="ghost" size="sm" onClick={() => setDeleteConfirmId(item._id)}>
                          Delete
                        </Button>
                      )}
                    </div>
                  </div>
                </GlassPanel>
              );
            })}

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1rem', marginTop: '1rem' }}>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                >
                  &larr; Prev
                </Button>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Page {page} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                >
                  Next &rarr;
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </DashboardShell>
  );
}

