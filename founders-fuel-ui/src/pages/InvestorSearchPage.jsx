// import React, { useState } from 'react';
// import { useNavigate } from 'react-router-dom';
// import { findMatches } from '../api/apiClient';

// // --- STYLES OBJECT ---
// // All CSS styles are defined here as JavaScript objects.
// const styles = {
//   searchPageContainer: {
//     display: 'flex',
//     flexDirection: 'column',
//     justifyContent: 'center',
//     alignItems: 'center',
//     minHeight: 'calc(100vh - 100px)',
//     padding: '2rem',
//     width: '100%',
//   },
//   formWrapper: {
//     display: 'flex',
//     flexDirection: 'column',
//     alignItems: 'center',
//     gap: '1.5rem',
//     width: '100%',
//     maxWidth: '750px',
//   },
//   descriptionInput: {
//     width: '100%',
//     boxSizing: 'border-box',
//     minHeight: '140px',
//     backgroundColor: '#1E293B',
//     border: '1px solid #334155',
//     borderRadius: '12px',
//     padding: '1.25rem',
//     color: '#94A3B8',
//     fontSize: '1.1rem',
//     fontFamily: 'inherit',
//     resize: 'vertical',
//     transition: 'all 0.2s ease-in-out',
//   },
//   descriptionInputFocus: {
//     outline: 'none',
//     borderColor: '#14B8A6', // primary-accent
//     boxShadow: '0 0 0 4px rgba(20, 184, 166, 0.2)',
//     color: '#F8FAFC', // text-primary
//   },
//   inspirationPrompt: {
//     alignSelf: 'flex-start',
//     marginTop: '-1.25rem',
//     marginLeft: '0.5rem',
//     fontSize: '1.2rem',
//   },
//   dropdownContainer: {
//     display: 'flex',
//     justifyContent: 'center',
//     gap: '1rem',
//     width: '100%',
//     maxWidth: '450px',
//   },
//   selectWrapper: {
//     flex: 1,
//   },
//   select: {
//     width: '100%',
//     padding: '0.75rem 1rem',
//     backgroundColor: '#1E293B',
//     border: '1px solid #334155',
//     borderRadius: '8px',
//     color: '#94A3B8',
//     fontSize: '1rem',
//     appearance: 'none',
//     backgroundImage: `url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2394A3B8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e")`,
//     backgroundRepeat: 'no-repeat',
//     backgroundPosition: 'right 0.75rem center',
//     backgroundSize: '1em',
//     cursor: 'pointer',
//     transition: 'all 0.2s ease-in-out',
//   },
//   selectFocus: {
//     borderColor: '#14B8A6',
//   },
//   findMatchesBtn: {
//     background: 'none',
//     border: 'none',
//     color: '#94A3B8',
//     fontSize: '1.25rem',
//     fontWeight: 600,
//     cursor: 'pointer',
//     padding: '1rem',
//     marginTop: '1rem',
//     transition: 'color 0.2s, transform 0.2s',
//   },
//   findMatchesBtnHover: {
//     color: '#14B8A6',
//     transform: 'scale(1.05)',
//   },
//   findMatchesBtnDisabled: {
//     color: '#475569',
//     cursor: 'not-allowed',
//     transform: 'none',
//   },
//   errorMessage: {
//     color: '#ef4444',
//     fontWeight: 600,
//     fontSize: '1.1rem',
//     textAlign: 'center',
//     minHeight: '1.5rem',
//   },
// };

// const InvestorSearchPage = () => {
//   const [description, setDescription] = useState('');
//   const [stage, setStage] = useState('Series A');
//   const [investorType, setInvestorType] = useState('Venture Fund');
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');
//   const navigate = useNavigate();

//   // State to manage hover and focus for dynamic styling
//   const [isInputFocused, setInputFocused] = useState(false);
//   const [isButtonHovered, setButtonHovered] = useState(false);

//   const handleDescriptionChange = (e) => {
//     setDescription(e.target.value);
//     if (error) setError('');
//   };

//   const handleFindMatches = async () => {
//     if (description.trim().length < 10) {
//       setError('Please provide a more detailed startup description.');
//       return;
//     }
//     setLoading(true);
//     setError('');
//     try {
//       const response = await findMatches(description, stage, investorType);
//       navigate('/investor-results', { state: { results: response.data, description } });
//     } catch (err) {
//       const errorMessage = err.response?.data?.detail || 'An unexpected error occurred.';
//       setError(errorMessage);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div style={styles.searchPageContainer}>
//       {/* Inject a style tag to handle placeholder color, which can't be done inline */}
//       <style>{`
//         .description-input-custom::placeholder {
//           color: #475569;
//         }
//       `}</style>
      
//       <div style={styles.formWrapper}>
//         <textarea
//           className="description-input-custom" // Use className for placeholder styling
//           style={{
//             ...styles.descriptionInput,
//             ...(isInputFocused && styles.descriptionInputFocus),
//           }}
//           value={description}
//           onChange={handleDescriptionChange}
//           onFocus={() => setInputFocused(true)}
//           onBlur={() => setInputFocused(false)}
//           placeholder="e.g., A B2B SaaS platform using AI to optimize logistics for e-commerce companies."
//         />

//         <div style={styles.inspirationPrompt}>💡</div>

//         <div style={styles.dropdownContainer}>
//           <div style={styles.selectWrapper}>
//             <select style={styles.select} value={stage} onChange={(e) => setStage(e.target.value)}>
//               <option>Pre-Seed</option>
//               <option>Seed</option>
//               <option>Series A</option>
//               <option>Series B</option>
//               <option>Series C</option>
//             </select>
//           </div>
//           <div style={styles.selectWrapper}>
//             <select style={styles.select} value={investorType} onChange={(e) => setInvestorType(e.target.value)}>
//               <option>Venture Fund</option>
//               <option>Corporate VC</option>
//               <option>Angel Network</option>
//               <option>Angel Investor</option>
//               <option>Accelerator</option>
//             </select>
//           </div>
//         </div>

//         <div style={styles.errorMessage}>{error}</div>

//         <button
//           style={{
//             ...styles.findMatchesBtn,
//             ...(isButtonHovered && styles.findMatchesBtnHover),
//             ...(loading && styles.findMatchesBtnDisabled),
//           }}
//           onClick={handleFindMatches}
//           disabled={loading}
//           onMouseEnter={() => setButtonHovered(true)}
//           onMouseLeave={() => setButtonHovered(false)}
//         >
//           {loading ? 'Analyzing...' : '🚀 Find My Matches'}
//         </button>
//       </div>
//     </div>
//   );
// };

// export default InvestorSearchPage;
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { findMatches } from '../api/apiClient';

const EXAMPLE_QUERIES = [
  "AI-powered fintech platform for SMB lending",
  "B2B SaaS for supply chain optimization",
  "Sustainable fashion marketplace with ML recommendations",
  "Healthcare IoT devices for remote patient monitoring",
  "EdTech platform for personalized learning",
  "Cryptocurrency exchange with advanced trading features",
];

const styles = {
  searchPageContainer: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: 'calc(100vh - 100px)',
    padding: '2rem',
    width: '100%',
    background: 'radial-gradient(circle at top, #0f172a, #020617)',
    color: '#F8FAFC',
    fontFamily: "'Inter', sans-serif",
  },
  headerContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    marginTop: '3rem',
    marginBottom: '1.5rem',
    animation: 'fadeInDown 0.6s ease',
  },
  logoTitle: {
    fontSize: '2.75rem',
    fontWeight: 700,
    color: '#14B8A6',
    display: 'flex',
    alignItems: 'center',
    gap: '0.6rem',
    textShadow: '0 0 12px rgba(20,184,166,0.4)',
  },
  tagline: {
    fontSize: '1.2rem',
    color: '#94A3B8',
    maxWidth: '540px',
    lineHeight: 1.5,
  },
  formContainerWrapper: {
    backdropFilter: 'blur(18px)',
    background: 'rgba(30, 41, 59, 0.65)',
    border: '1px solid rgba(20, 184, 166, 0.35)',
    boxShadow: '0 0 35px rgba(20, 184, 166, 0.2)',
    borderRadius: '20px',
    padding: '2.25rem',
    width: '100%',
    maxWidth: '780px',
    transition: 'all 0.3s ease',
  },
  formContainerWrapperHover: {
    border: '1px solid #14B8A6',
    boxShadow: '0 0 50px rgba(20, 184, 166, 0.4)',
    transform: 'scale(1.01)',
  },
  formWrapper: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '1.2rem',
    width: '100%',
  },
  inputLabel: {
    alignSelf: 'flex-start',
    fontSize: '1.1rem',
    fontWeight: 600,
    color: '#E2E8F0',
    marginBottom: '0.3rem',
  },
  descriptionInput: {
    width: '100%',
    boxSizing: 'border-box',
    minHeight: '110px',
    backgroundColor: 'rgba(15, 23, 42, 0.7)',
    border: '1px solid #334155',
    borderRadius: '10px',
    padding: '1rem',
    color: '#F8FAFC',
    fontSize: '1rem',
    resize: 'vertical',
    transition: 'all 0.25s ease-in-out',
  },
  descriptionInputFocus: {
    outline: 'none',
    borderColor: '#14B8A6',
    boxShadow: '0 0 10px rgba(20, 184, 166, 0.5)',
  },
  exampleButton: {
    width: '100%',
    padding: '1rem',
    background: 'linear-gradient(90deg, #0EA5E9, #14B8A6)',
    border: 'none',
    borderRadius: '10px',
    color: '#0F172A',
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'transform 0.25s, box-shadow 0.25s',
  },
  exampleButtonHover: {
    transform: 'translateY(-3px)',
    boxShadow: '0 0 15px rgba(14, 165, 233, 0.5)',
  },
  examplesContainer: {
    width: '100%',
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '1rem',
    padding: '1.5rem',
    border: '1px solid rgba(20, 184, 166, 0.35)',
    borderRadius: '0 0 12px 12px',
    background: 'rgba(15, 23, 42, 0.85)',
    animation: 'fadeIn 0.4s ease',
  },
  exampleItem: {
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    padding: '1rem',
    borderRadius: '8px',
    color: '#F8FAFC',
    fontSize: '1rem',
    cursor: 'pointer',
    transition: 'all 0.25s ease',
    border: '1px solid transparent',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  exampleItemHover: {
    backgroundColor: 'rgba(20, 184, 166, 0.15)',
    borderColor: '#14B8A6',
    transform: 'translateY(-2px)',
  },
  dropdownContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '2rem',
    width: '100%',
    marginTop: '1rem',
  },
  selectWrapper: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: '0.3rem',
  },
  selectLabel: {
    fontSize: '1rem',
    fontWeight: 600,
    color: '#E2E8F0',
  },
  select: {
    width: '100%',
    padding: '0.8rem 1rem',
    backgroundColor: 'rgba(15, 23, 42, 0.85)',
    border: '1px solid #334155',
    borderRadius: '10px',
    color: '#F8FAFC',
    fontSize: '1rem',
    appearance: 'none',
    backgroundImage: `url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2314B8A6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e")`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'right 0.75rem center',
    backgroundSize: '1em',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  selectFocus: {
    borderColor: '#14B8A6',
    boxShadow: '0 0 10px rgba(20, 184, 166, 0.5)',
  },
  findMatchesBtn: {
    width: '100%',
    background: 'linear-gradient(90deg, #14B8A6, #06B6D4)',
    border: 'none',
    borderRadius: '10px',
    color: '#0F172A',
    fontSize: '1.25rem',
    fontWeight: 700,
    cursor: 'pointer',
    padding: '1rem 2rem',
    marginTop: '1.5rem',
    transition: 'transform 0.25s, box-shadow 0.25s',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '0.5rem',
  },
  findMatchesBtnHover: {
    transform: 'translateY(-3px)',
    boxShadow: '0 4px 15px rgba(20, 184, 166, 0.5)',
  },
  findMatchesBtnDisabled: {
    opacity: 0.7,
    cursor: 'not-allowed',
    boxShadow: 'none',
    transform: 'none',
  },
  errorMessage: {
    color: '#F87171',
    fontWeight: 600,
    fontSize: '1rem',
    textAlign: 'center',
    minHeight: '1.5rem',
    marginTop: '0.5rem',
  },
};

const InvestorSearchPage = () => {
  const [description, setDescription] = useState('Sustainable fashion marketplace with ML recommendations');
  const [stage, setStage] = useState('Seed');
  const [investorType, setInvestorType] = useState('Venture Fund');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showExamples, setShowExamples] = useState(false);
  const [hoveredExample, setHoveredExample] = useState(null);
  const [isFormHovered, setFormHovered] = useState(false);
  const [isButtonHovered, setButtonHovered] = useState(false);
  const [isExampleHovered, setExampleHovered] = useState(false);
  const [isInputFocused, setInputFocused] = useState(false);
  const [isSelectFocused, setSelectFocused] = useState(false);
  const navigate = useNavigate();

  const handleFindMatches = async () => {
    if (description.trim().length < 10) {
      setError('Please provide a more detailed startup description.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await findMatches(description, stage, investorType);
      navigate('/investor-results', { state: { results: response.data, description } });
    } catch (err) {
      setError(err.response?.data?.detail || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.searchPageContainer}>
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInDown { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
        select option {
          background-color: #0F172A;
          color: #F8FAFC;
          padding: 0.5rem 1rem;
        }
        select option:hover, select option:checked {
          background-color: #14B8A6;
          color: #0F172A;
        }
      `}</style>

      <div style={styles.headerContainer}>
        <div style={styles.logoTitle}>
          <span>🔥</span> Founders Fuel
        </div>
        <div style={styles.tagline}>
          Describe your startup. We'll find the top 4 most relevant investors.
        </div>
      </div>

      <div
        style={{
          ...styles.formContainerWrapper,
          ...(isFormHovered && styles.formContainerWrapperHover),
        }}
        onMouseEnter={() => setFormHovered(true)}
        onMouseLeave={() => setFormHovered(false)}
      >
        <div style={styles.formWrapper}>
          <div style={{ width: '100%' }}>
            <div style={styles.inputLabel}>Describe your startup</div>
            <textarea
              style={{
                ...styles.descriptionInput,
                ...(isInputFocused && styles.descriptionInputFocus),
              }}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              onFocus={() => setInputFocused(true)}
              onBlur={() => setInputFocused(false)}
              placeholder="e.g., A B2B SaaS platform using AI to optimize logistics for e-commerce companies."
            />
          </div>

          <button
            style={{
              ...styles.exampleButton,
              ...(isExampleHovered && styles.exampleButtonHover),
              ...(showExamples && { borderBottomLeftRadius: 0, borderBottomRightRadius: 0 }),
            }}
            onClick={() => setShowExamples(!showExamples)}
            onMouseEnter={() => setExampleHovered(true)}
            onMouseLeave={() => setExampleHovered(false)}
          >
            💡 {showExamples ? 'Hide Example Queries' : 'Show Example Queries'}
          </button>

          {showExamples && (
            <div style={styles.examplesContainer}>
              {EXAMPLE_QUERIES.map((query, i) => (
                <div
                  key={i}
                  style={{
                    ...styles.exampleItem,
                    ...(hoveredExample === i && styles.exampleItemHover),
                  }}
                  onClick={() => setDescription(query)}
                  onMouseEnter={() => setHoveredExample(i)}
                  onMouseLeave={() => setHoveredExample(null)}
                >
                  📄 {query}
                </div>
              ))}
            </div>
          )}

          <div style={styles.dropdownContainer}>
            <div style={styles.selectWrapper}>
              <div style={styles.selectLabel}>Funding Stage</div>
              <select
                style={{
                  ...styles.select,
                  ...(isSelectFocused && styles.selectFocus),
                }}
                value={stage}
                onFocus={() => setSelectFocused(true)}
                onBlur={() => setSelectFocused(false)}
                onChange={(e) => setStage(e.target.value)}
              >
                <option>Pre-Seed</option>
                <option>Seed</option>
                <option>Series A</option>
                <option>Series B</option>
                <option>Series C</option>
                <option>Series D</option>
              </select>
            </div>

            <div style={styles.selectWrapper}>
              <div style={styles.selectLabel}>Investor Type</div>
              <select
                style={{
                  ...styles.select,
                  ...(isSelectFocused && styles.selectFocus),
                }}
                value={investorType}
                onFocus={() => setSelectFocused(true)}
                onBlur={() => setSelectFocused(false)}
                onChange={(e) => setInvestorType(e.target.value)}
              >
                <option>Venture Fund</option>
                <option>Corporate VC</option>
                <option>Angel Network</option>
                <option>Angel Investor</option>
                <option>Accelerator</option>
              </select>
            </div>
          </div>

          <div style={styles.errorMessage}>{error}</div>

          <button
            style={{
              ...styles.findMatchesBtn,
              ...(isButtonHovered && styles.findMatchesBtnHover),
              ...(loading && styles.findMatchesBtnDisabled),
            }}
            onClick={handleFindMatches}
            disabled={loading}
            onMouseEnter={() => setButtonHovered(true)}
            onMouseLeave={() => setButtonHovered(false)}
          >
            {loading ? 'Analyzing...' : '🚀 Find My Matches'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default InvestorSearchPage;
