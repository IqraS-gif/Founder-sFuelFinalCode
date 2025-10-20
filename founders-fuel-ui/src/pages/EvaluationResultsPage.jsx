import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const styles = {
  page: {
    minHeight: 'calc(100vh - 80px)',
    padding: '3rem 1.5rem',
    display: 'flex',
    justifyContent: 'center',
    background: 'radial-gradient(circle at top left, #071028, #020617)',
    fontFamily: "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial",
    color: '#E6EEF3',
  },
  container: {
    width: '100%',
    maxWidth: '980px',
    borderRadius: '16px',
    padding: '1.25rem',
    backdropFilter: 'blur(12px)',
    background: 'linear-gradient(180deg, rgba(15,23,42,0.55), rgba(2,6,23,0.45))',
    border: '1px solid rgba(20,184,166,0.12)',
    boxShadow: '0 10px 30px rgba(2,6,23,0.7)',
    transition: 'transform .18s ease, box-shadow .18s ease',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '1rem',
  },
  title: {
    fontSize: '1.6rem',
    fontWeight: 700,
    color: '#14B8A6',
    textShadow: '0 6px 24px rgba(20,184,166,0.08)',
  },
  subtitle: {
    color: '#9FB3BD',
    marginTop: '0.25rem',
  },
  accordion: {
    marginTop: '1.25rem',
    display: 'grid',
    gap: '0.75rem',
  },
  card: {
    borderRadius: '12px',
    overflow: 'hidden',
    border: '1px solid rgba(255,255,255,0.03)',
    background: 'linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01))',
    transition: 'transform .18s ease, box-shadow .18s ease',
  },
  cardHeader: {
    padding: '1rem 1.25rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    cursor: 'pointer',
    gap: '1rem',
  },
  cardTitle: {
    fontWeight: 700,
    color: '#E6EEF3',
  },
  cardMeta: {
    color: '#88a6a9',
    fontSize: '0.9rem',
  },
  cardBody: {
    padding: '1rem 1.25rem',
    color: '#DFF6F2',
    lineHeight: 1.55,
    whiteSpace: 'pre-wrap',
    maxHeight: 0,
    overflow: 'hidden',
    transition: 'max-height 0.36s cubic-bezier(.2,.9,.2,1), padding 0.18s ease',
  },
  cardBodyOpen: {
    maxHeight: '1200px',
    paddingBottom: '1.25rem',
  },
  actions: {
    marginTop: '1rem',
    display: 'flex',
    gap: '0.75rem',
    justifyContent: 'flex-end',
  },
  backButton: {
    background: 'transparent',
    border: '1px solid rgba(255,255,255,0.06)',
    color: '#E6EEF3',
    padding: '0.6rem 0.9rem',
    borderRadius: '8px',
    cursor: 'pointer',
  },
  copyBtn: {
    background: 'linear-gradient(90deg, #14B8A6, #06B6D4)',
    border: 'none',
    color: '#071028',
    padding: '0.6rem 0.9rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontWeight: 700,
  }
};

function sanitizeText(text = '') {
  // remove bold markers and leading markdown headers, keep code blocks/newlines
  return String(text)
    .replace(/\*\*/g, '')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/\[(.*?)\]\((.*?)\)/g, '$1') // convert links [text](url) -> text
    .trim();
}

const EvaluationResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state || {};
  const evaluation = state.evaluation || null;
  const startupName = state.startupName || '';

  const [openIndex, setOpenIndex] = useState(null);

  useEffect(() => {
    if (!evaluation) {
      // No state => go back to evaluation input
      navigate('/evaluation', { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evaluation]);

  if (!evaluation) {
    return null;
  }

  // Convert evaluation object to an ordered list (if keys order matters you can tune this)
  const entries = Object.entries(evaluation);

  const toggle = (i) => setOpenIndex(openIndex === i ? null : i);

  const handleCopy = async () => {
    try {
      const fullText = entries.map(([k, v]) => `${k.replace(/_/g,' ').toUpperCase()}\n\n${sanitizeText(v || '')}\n\n`).join('\n');
      await navigator.clipboard.writeText(fullText);
      // brief visual feedback could be added
      alert('Evaluation copied to clipboard');
    } catch {
      alert('Copy failed');
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.header}>
          <div>
            <div style={styles.title}>Evaluation Report</div>
            <div style={styles.subtitle}>{startupName ? `Report for ${startupName}` : 'Comprehensive 7-domain analysis'}</div>
          </div>
        </div>

        <div style={styles.accordion}>
          {entries.map(([key, value], i) => {
            const title = key.replace(/_/g, ' ').toUpperCase();
            const content = sanitizeText(value || 'No content available for this section.');
            const isOpen = openIndex === i;
            return (
              <div
                key={key}
                style={{
                  ...styles.card,
                  transform: isOpen ? 'translateY(-4px)' : 'none',
                  boxShadow: isOpen ? '0 12px 40px rgba(2,6,23,0.6)' : 'none'
                }}
              >
                <div
                  style={styles.cardHeader}
                  onClick={() => toggle(i)}
                  role="button"
                  aria-expanded={isOpen}
                >
                  <div>
                    <div style={styles.cardTitle}>{title}</div>
                    <div style={styles.cardMeta}>{content.slice(0, 120)}{content.length > 120 ? '…' : ''}</div>
                  </div>
                  <div style={{ color: '#88a6a9', fontWeight: 700 }}>
                    {isOpen ? '−' : '+'}
                  </div>
                </div>

                <div
                  style={{
                    ...styles.cardBody,
                    ...(isOpen ? styles.cardBodyOpen : {})
                  }}
                >
                  <div style={{ opacity: isOpen ? 1 : 0.92 }}>
                    {content}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div style={styles.actions}>
          <button style={styles.backButton} onClick={() => navigate(-1)}>Back</button>
          <button style={styles.copyBtn} onClick={handleCopy}>Copy Report</button>
        </div>
      </div>
    </div>
  );
};

export default EvaluationResultsPage;