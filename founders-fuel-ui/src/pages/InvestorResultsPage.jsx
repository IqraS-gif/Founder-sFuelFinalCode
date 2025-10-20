// // // import React from 'react';
// // // import { useLocation, Link } from 'react-router-dom';
// // // import InvestorCard from '../components/analysis/InvestorCard'; // Create this new component


// // // const InvestorResultsPage = () => {
// // //   const { state } = useLocation();
  
// // //   // Guard clause if the page is accessed directly without data
// // //   if (!state || !state.results) {
// // //     return (
// // //       <div className="results-container" style={{ textAlign: 'center' }}>
// // //         <h2>No Data Available</h2>
// // //         <p>Please start a new search to find investor matches.</p>
// // //         <Link to="/investor-search" className="back-link">← Back to Search</Link>
// // //       </div>
// // //     );
// // //   }

// // //   const { results, description } = state;
// // //   const { extracted_keywords, matching_industries, insights, investors } = results;

// // //   const exportToCSV = () => {
// // //     const headers = ["Investor Name", "Website", "Type", "Stages", "Match Score", "Focus Areas"];
// // //     const rows = investors.map(inv => [
// // //       `"${inv.investor_name}"`,
// // //       `"${inv.website}"`,
// // //       `"${inv.fund_type}"`,
// // //       `"${inv.fund_stage}"`,
// // //       inv.match_score,
// // //       `"${inv.focus_areas.join(", ")}"`
// // //     ]);

// // //     const csvContent = "data:text/csv;charset=utf-8," 
// // //       + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    
// // //     const encodedUri = encodeURI(csvContent);
// // //     const link = document.createElement("a");
// // //     link.setAttribute("href", encodedUri);
// // //     link.setAttribute("download", `FoundersFuel_Matches_${new Date().toISOString().split('T')[0]}.csv`);
// // //     document.body.appendChild(link);
// // //     link.click();
// // //     document.body.removeChild(link);
// // //   };

// // //   return (
// // //     <div className="results-container">
// // //       <Link to="/investor-search" className="back-link">← New Search</Link>
      
// // //       <div className="summary-section">
// // //         <p className="searched-description"><strong>Your Startup:</strong> "{description}"</p>
        
// // //         <div className="tags-container">
// // //           <strong>Extracted Keywords:</strong>
// // //           {extracted_keywords.map(kw => <span key={kw} className="tag keyword-tag">{kw}</span>)}
// // //         </div>

// // //         <div className="tags-container">
// // //           <strong>Matching Industries:</strong>
// // //           {matching_industries.map(ind => <span key={ind} className="tag industry-tag">{ind}</span>)}
// // //         </div>
        
// // //         <div className="insights-container">
// // //           {insights.map((insight, index) => (
// // //             <p key={index} className="insight">
// // //               {insight.includes("Excellent") || insight.includes("Good") ? '✅' : '📈'} {insight}
// // //             </p>
// // //           ))}
// // //         </div>
// // //       </div>
      
// // //       <div className="results-header">
// // //         <h2>Found {investors.length} Highly-Matched Investors</h2>
// // //         {investors.length > 0 && (
// // //           <button onClick={exportToCSV} className="export-btn">
// // //             Export CSV
// // //           </button>
// // //         )}
// // //       </div>

// // //       {investors.length > 0 ? (
// // //         <div className="investor-grid">
// // //           {investors.map((investor, index) => (
// // //             <InvestorCard key={index} investor={investor} keywords={extracted_keywords} />
// // //           ))}
// // //         </div>
// // //       ) : (
// // //         <p>No suitable investors found based on your criteria. Try refining your description.</p>
// // //       )}
// // //     </div>
// // //   );
// // // };

// // // export default InvestorResultsPage;


// // // src/pages/InvestorResultsPage.jsx

// // import React from 'react';
// // import { useLocation, Link } from 'react-router-dom';
// // // The import below assumes you've created the InvestorCard component above
// // import InvestorCard from '../components/analysis/InvestorCard'; 


// // // Sample data structure for 'results' as it would be passed in 'state'
// // /*
// // const sampleResults = {
// //   extracted_keywords: ['AI', 'SaaS', 'healthcare'],
// //   matching_industries: ['Healthcare Technology', 'Artificial Intelligence'],
// //   insights: [
// //     'Excellent matches found! These investors align strongly with your startup.',
// //     'Diverse investor types found (2 different types)',
// //   ],
// //   investors: [
// //     {
// //       investor_name: 'Example Ventures',
// //       website: 'example-ventures.com',
// //       fund_type: 'Venture Fund',
// //       fund_stage: ['Seed', 'Series A'],
// //       match_score: 85,
// //       focus_areas: ['AI', 'SaaS', 'Healthcare', 'Fintech'],
// //     },
// //     {
// //       investor_name: 'Tech Growth Capital',
// //       website: 'techgrowth.com',
// //       fund_type: 'Venture Fund',
// //       fund_stage: ['Series A', 'Series B'],
// //       match_score: 78,
// //       focus_areas: ['Enterprise Software', 'AI', 'Machine Learning'],
// //     },
// //   ],
// // };
// // */

// // const InvestorResultsPage = () => {
// //   const { state } = useLocation();

// //   // Guard clause if the page is accessed directly without data
// //   if (!state || !state.results) {
// //     return (
// //       <div style={{ 
// //         backgroundColor: '#0F131C', 
// //         minHeight: '100vh', 
// //         color: '#E0E0E0', 
// //         padding: '40px', 
// //         textAlign: 'center' 
// //       }}>
// //         <h2 style={{ color: '#FFFFFF' }}>No Data Available</h2>
// //         <p style={{ color: '#B3B3B3' }}>Please start a new search to find investor matches.</p>
// //         <Link to="/investor-search" style={{ color: '#00BFFF', textDecoration: 'none' }}>← Back to Search</Link>
// //       </div>
// //     );
// //   }

// //   const { results, description } = state;
// //   const { extracted_keywords, matching_industries, insights, investors } = results;

// //   const exportToCSV = () => {
// //     const headers = ["Investor Name", "Website", "Type", "Stages", "Match Score", "Focus Areas"];
// //     const rows = investors.map(inv => [
// //       `"${inv.investor_name}"`,
// //       `"${inv.website}"`,
// //       `"${inv.fund_type}"`,
// //       `"${Array.isArray(inv.fund_stage) ? inv.fund_stage.join(" / ") : inv.fund_stage}"`,
// //       inv.match_score,
// //       `"${inv.focus_areas.join(", ")}"`
// //     ]);

// //     const csvContent = "data:text/csv;charset=utf-8," 
// //       + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    
// //     const encodedUri = encodeURI(csvContent);
// //     const link = document.createElement("a");
// //     link.setAttribute("href", encodedUri);
// //     link.setAttribute("download", `FoundersFuel_Matches_${new Date().toISOString().split('T')[0]}.csv`);
// //     document.body.appendChild(link);
// //     link.click();
// //     document.body.removeChild(link);
// //   };

// //   const containerStyle = {
// //     backgroundColor: '#0F131C', 
// //     minHeight: '100vh', 
// //     color: '#E0E0E0', 
// //     padding: '40px',
// //     paddingTop: '120px',
// //     fontFamily: 'Arial, sans-serif'
// //   };

// //   const tagStyle = {
// //     backgroundColor: '#334155',
// //     color: '#FFFFFF',
// //     borderRadius: '4px',
// //     padding: '4px 10px',
// //     fontSize: '14px',
// //     fontWeight: '500',
// //     marginRight: '8px',
// //     marginBottom: '8px',
// //     display: 'inline-block',
// //   };

// //   const keywordTagStyle = {
// //     ...tagStyle,
// //     backgroundColor: '#00BFFF', // Blue for keywords
// //   };

// //   const industryTagStyle = {
// //     ...tagStyle,
// //     backgroundColor: '#38A169', // Green for industries
// //   };

// //   const insightContainerStyle = {
// //     backgroundColor: '#1C2130',
// //     borderRadius: '8px',
// //     padding: '16px',
// //     marginBottom: '30px',
// //     border: '1px solid #334155',
// //   };
  
// //   const insightStyle = {
// //     margin: '8px 0',
// //     fontSize: '16px',
// //     fontWeight: '500',
// //     color: '#90EE90', // Light green for positive insights
// //     display: 'flex',
// //     alignItems: 'center',
// //     gap: '8px',
// //   };

// //   const resultsHeaderStyle = {
// //     display: 'flex',
// //     justifyContent: 'space-between',
// //     alignItems: 'center',
// //     marginBottom: '20px',
// //     paddingBottom: '10px',
// //     borderBottom: '1px solid #334155',
// //   };

// //   const exportButtonStyle = {
// //     backgroundColor: 'transparent',
// //     color: '#00BFFF',
// //     border: '1px solid #00BFFF',
// //     borderRadius: '4px',
// //     padding: '10px 15px',
// //     fontSize: '14px',
// //     cursor: 'pointer',
// //     fontWeight: '600',
// //     display: 'flex',
// //     alignItems: 'center',
// //     gap: '5px',
// //     transition: 'background-color 0.2s',
// //   };


// //   return (
// //     <div style={containerStyle}>
// //       <Link to="/investor-search" style={{ color: '#00BFFF', textDecoration: 'none', marginBottom: '20px', display: 'inline-block' }}>
// //         ← New Search
// //       </Link>
      
// //       <div className="summary-section" style={{ marginBottom: '30px' }}>
// //         <p style={{ color: '#B3B3B3', fontSize: '16px', marginBottom: '15px' }}>
// //           <strong>Your Startup:</strong> <span style={{ color: '#FFFFFF' }}>"{description}"</span>
// //         </p>
        
// //         {/* Extracted Keywords */}
// //         <div style={{ backgroundColor: '#1C2130', borderRadius: '8px', padding: '16px', marginBottom: '16px' }}>
// //           <h4 style={{ color: '#90EE90', fontSize: '16px', fontWeight: '600', margin: '0 0 10px 0' }}>🔗 Extracted Keywords:</h4>
// //           <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
// //             {extracted_keywords.map(kw => (
// //               <span key={kw} style={keywordTagStyle}>{kw}</span>
// //             ))}
// //           </div>
// //         </div>

// //         {/* Matching Industries */}
// //         <div style={{ backgroundColor: '#1C2130', borderRadius: '8px', padding: '16px', marginBottom: '16px' }}>
// //           <h4 style={{ color: '#90EE90', fontSize: '16px', fontWeight: '600', margin: '0 0 10px 0' }}>@ Matching Industries:</h4>
// //           <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
// //             {matching_industries.map(ind => (
// //               <span key={ind} style={industryTagStyle}>{ind}</span>
// //             ))}
// //           </div>
// //         </div>
        
// //         {/* Insights */}
// //         <div style={insightContainerStyle}>
// //           {insights.map((insight, index) => (
// //             <p key={index} style={insightStyle}>
// //               {insight.includes("Excellent") || insight.includes("Good") ? '📈' : '📊'} {insight}
// //             </p>
// //           ))}
// //         </div>
// //       </div>
      
// //       <div style={resultsHeaderStyle}>
// //         <h2 style={{ color: '#FFFFFF', fontSize: '24px', fontWeight: '700', margin: '0' }}>
// //           Found {investors.length} Highly-Matched Investors
// //         </h2>
// //         {investors.length > 0 && (
// //           <button 
// //             onClick={exportToCSV} 
// //             style={exportButtonStyle}
// //             onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(0, 191, 255, 0.1)'}
// //             onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
// //           >
// //             <span role="img" aria-label="Export">📄</span> Export CSV
// //           </button>
// //         )}
// //       </div>

// //       {investors.length > 0 ? (
// //         <div className="investor-grid" style={{
// //           display: 'grid',
// //           gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
// //           gap: '20px',
// //         }}>
// //           {/* Note: The InvestorCard component is styled with inline CSS in its own file */}
// //           {investors.map((investor, index) => (
// //             <InvestorCard key={index} investor={investor} keywords={extracted_keywords} />
// //           ))}
// //         </div>
// //       ) : (
// //         <p style={{ color: '#B3B3B3', fontSize: '16px', textAlign: 'center' }}>
// //           No suitable investors found based on your criteria. Try refining your description.
// //         </p>
// //       )}
// //     </div>
// //   );
// // };

// // export default InvestorResultsPage;


// // src/pages/InvestorResultsPage.jsx

// import React, { useState } from 'react'; 
// import { useLocation, Link } from 'react-router-dom';
// import InvestorCard from '../components/analysis/InvestorCard';

// const InvestorResultsPage = () => {
//   const { state } = useLocation();
  
//   const NAVBAR_HEIGHT = '60px'; // Set this to your actual fixed navbar height

//   // State for hover effects
//   const [isExportHovered, setIsExportHovered] = useState(false);
//   const [isNewSearchHovered, setIsNewSearchHovered] = useState(false);

//   if (!state || !state.results) {
//     return (
//       <div style={{ 
//         backgroundColor: '#1E1E2C', 
//         minHeight: '100vh', 
//         color: '#E0E0E0', 
//         padding: '40px', 
//         textAlign: 'center',
//         fontFamily: 'Inter, sans-serif'
//       }}>
//         <h2 style={{ color: '#FFFFFF' }}>No Data Available</h2>
//         <p style={{ color: '#A0A0B0' }}>Please start a new search to find investor matches.</p>
//         <Link 
//           to="/investor-search" 
//           style={{ 
//             color: '#7B68EE', 
//             textDecoration: 'none', 
//             fontWeight: '600' 
//           }}>
//           ← Back to Search
//         </Link>
//       </div>
//     );
//   }

//   const { results, description } = state;
//   const { extracted_keywords, matching_industries, insights, investors } = results;

//   const exportToCSV = () => {
//     const headers = ["Investor Name", "Website", "Type", "Stages", "Match Score", "Focus Areas"];
//     const rows = investors.map(inv => [
//       `"${inv.investor_name}"`,
//       `"${inv.website}"`,
//       `"${inv.fund_type}"`,
//       `"${Array.isArray(inv.fund_stage) ? inv.fund_stage.join(" / ") : inv.fund_stage}"`,
//       inv.match_score,
//       `"${inv.focus_areas.join(", ")}"`
//     ]);

//     const csvContent = "data:text/csv;charset=utf-8," 
//       + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    
//     const encodedUri = encodeURI(csvContent);
//     const link = document.createElement("a");
//     link.setAttribute("href", encodedUri);
//     link.setAttribute("download", `FoundersFuel_Matches_${new Date().toISOString().split('T')[0]}.csv`);
//     document.body.appendChild(link);
//     link.click();
//     document.body.removeChild(link);
//   };

//   /* ---------------------- INLINE STYLES (MATCHING IMAGE UI) ---------------------- */

//   const containerStyle = {
//     backgroundColor: '#1E1E2C',
//     minHeight: '100vh', 
//     color: '#E0E0E0', 
//     padding: '40px',
//     fontFamily: 'Inter, sans-serif',
//     paddingTop: `calc(40px + ${NAVBAR_HEIGHT})`, 
//   };

//   const topControlsStyle = {
//     display: 'flex',
//     justifyContent: 'space-between',
//     marginBottom: '30px',
//   };

//   const buttonBaseStyle = {
//     borderRadius: '8px',
//     padding: '10px 18px',
//     fontSize: '15px',
//     cursor: 'pointer',
//     fontWeight: '600',
//     display: 'flex',
//     alignItems: 'center',
//     gap: '8px',
//     transition: 'background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease',
//   }

//   const newSearchButtonStyle = {
//     ...buttonBaseStyle,
//     backgroundColor: 'transparent',
//     color: '#00BFFF', // Blue for 'New Search'
//     border: '1px solid #00BFFF',
//     textDecoration: 'none',
//     ...(isNewSearchHovered && {
//         backgroundColor: '#00BFFF', 
//         color: '#FFFFFF',
//         boxShadow: '0 4px 12px rgba(0, 191, 255, 0.4)',
//     })
//   };

//   const exportButtonStyle = {
//     ...buttonBaseStyle,
//     backgroundColor: '#7B68EE', // Purple/Violet for 'Export CSV'
//     color: '#FFFFFF', 
//     border: 'none',
//     ...(isExportHovered && {
//         backgroundColor: '#9370DB', // Lighter purple on hover
//         boxShadow: '0 4px 12px rgba(123, 104, 238, 0.6)',
//     })
//   };

//   const summaryCardStyle = {
//     backgroundColor: '#2A2A3A',
//     borderRadius: '12px',
//     padding: '25px',
//     marginBottom: '30px',
//     boxShadow: '0 6px 15px rgba(0, 0, 0, 0.3)',
//     border: '1px solid #02e9f9ff',
//   };

//   const descriptionTextStyle = { 
//     color: '#A0A0B0',
//     fontSize: '15px', 
//     marginBottom: '20px', 
//     fontWeight: '500',
//   };

//   const summaryGridStyle = {
//     display: 'grid',
//     gridTemplateColumns: '1fr 1fr', // Two columns for keywords/industries
//     gap: '20px',
//     marginBottom: '20px',
//   };

//   const tagBoxStyle = {
//     backgroundColor: '#38384D', 
//     borderRadius: '8px',
//     padding: '18px',
//     border: '1px solid #00eeffff',
//   };

//   const tagHeaderStyle = { 
//     color: '#FFFFFF', 
//     fontSize: '16px', 
//     fontWeight: '700', 
//     margin: '0 0 12px 0',
//     display: 'flex',
//     alignItems: 'center',
//     gap: '8px'
//   };

//   const tagBaseStyle = {
//     borderRadius: '6px',
//     padding: '7px 14px',
//     fontSize: '13px',
//     fontWeight: '600',
//     marginRight: '8px',
//     marginBottom: '8px',
//     display: 'inline-block',
//   };

//   const keywordTagStyle = {
//     ...tagBaseStyle,
//     backgroundColor: '#7B68EE', // Purple keywords
//     color: '#FFFFFF',
//   };

//   const industryTagStyle = {
//     ...tagBaseStyle,
//     backgroundColor: '#4CAF50', // Green industries
//     color: '#FFFFFF',
//   };

//   const insightsSectionStyle = {
//     backgroundColor: '#38384D',
//     borderRadius: '8px',
//     padding: '18px',
//     border: '1px solid #00ddffff',
//   };
  
//   const insightStyle = {
//     margin: '10px 0',
//     fontSize: '15px',
//     fontWeight: '500',
//     color: '#C0C0D0',
//     display: 'flex',
//     alignItems: 'center',
//     gap: '10px',
//   };

//   const mainHeaderStyle = { 
//     color: '#FFFFFF', 
//     fontSize: '32px', 
//     fontWeight: '800', 
//     margin: '0 0 30px 0',
//     letterSpacing: '-0.5px'
//   };

//   return (
//     <div style={containerStyle}>
//       <div style={topControlsStyle}>
//         <Link 
//           to="/investor-search" 
//           style={newSearchButtonStyle}
//           onMouseEnter={() => setIsNewSearchHovered(true)}
//           onMouseLeave={() => setIsNewSearchHovered(false)}
//         >
//           ← New Search
//         </Link>
//         {investors.length > 0 && (
//           <button 
//             onClick={exportToCSV} 
//             style={exportButtonStyle}
//             onMouseEnter={() => setIsExportHovered(true)}
//             onMouseLeave={() => setIsExportHovered(false)}
//           >
//             <span role="img" aria-label="Export">↓</span> Export CSV
//           </button>
//         )}
//       </div>
      
//       <div style={summaryCardStyle}>
//         <p style={descriptionTextStyle}>
//           Your Startup: <span style={{ color: '#FFFFFF', fontWeight: '600' }}>"{description}"</span>
//         </p>
        
//         <div style={summaryGridStyle}>
//           {/* Extracted Keywords */}
//           <div style={tagBoxStyle}>
//             <h4 style={tagHeaderStyle}><span role="img" aria-label="Keywords">◎</span> Extracted Keywords</h4>
//             <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
//               {extracted_keywords.map(kw => (
//                 <span key={kw} style={keywordTagStyle}>{kw}</span>
//               ))}
//             </div>
//           </div>

//           {/* Matching Industries */}
//           <div style={tagBoxStyle}>
//             <h4 style={tagHeaderStyle}><span role="img" aria-label="Industries">📈</span> Matching Industries</h4>
//             <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
//               {matching_industries.map(ind => (
//                 <span key={ind} style={industryTagStyle}>{ind}</span>
//               ))}
//             </div>
//           </div>
//         </div>
        
//         {/* Insights */}
//         <div style={insightsSectionStyle}>
//           {insights.map((insight, index) => (
//             <p key={index} style={insightStyle}>
//               {insight.includes("Excellent") || insight.includes("Good") || insight.includes("actively") ? '✨' : '⭐'} {insight}
//             </p>
//           ))}
//         </div>
//       </div>
      
//       <h2 style={mainHeaderStyle}>
//         Found {investors.length} Highly-Matched Investors
//       </h2>

//       {investors.length > 0 ? (
//         <div style={{
//           display: 'grid',
//           gridTemplateColumns: '1fr 1fr', // 👈 FIXED TWO CARDS PER ROW
//           gap: '25px', 
//         }}>
//           {investors.map((investor, index) => (
//             <InvestorCard 
//               key={index} 
//               investor={investor} 
//               keywords={extracted_keywords} 
//               maxFocusAreas={5} // Limit to 5 extra areas
//             />
//           ))}
//         </div>
//       ) : (
//         <p style={{ color: '#A0A0B0', fontSize: '16px', textAlign: 'center', padding: '50px' }}>
//           No suitable investors found. Try refining your description.
//         </p>
//       )}
//     </div>
//   );
// };

// // export default InvestorResultsPage;


// src/pages/InvestorResultsPage.jsx

import React, { useState } from 'react'; 
import { useLocation, Link } from 'react-router-dom';
import InvestorCard from '../components/analysis/InvestorCard';

const InvestorResultsPage = () => {
  const { state } = useLocation();
  
  const NAVBAR_HEIGHT = '60px'; 
  const [isExportHovered, setIsExportHovered] = useState(false);
  const [isNewSearchHovered, setIsNewSearchHovered] = useState(false);

  // ... (Guard clause remains the same)
  if (!state || !state.results) {
    // ... (Error return remains the same)
    return null; // Simplified
  }

  const { results, description } = state;
  const { extracted_keywords, matching_industries, insights, investors } = results;

  const exportToCSV = () => {
    // ... (export logic remains the same)
    const headers = ["Investor Name", "Website", "Type", "Stages", "Match Score", "Focus Areas"];
    const rows = investors.map(inv => [
      `"${inv.investor_name}"`,
      `"${inv.website}"`,
      `"${inv.fund_type}"`,
      `"${Array.isArray(inv.fund_stage) ? inv.fund_stage.join(" / ") : inv.fund_stage}"`,
      inv.match_score,
      `"${inv.focus_areas.join(", ")}"`
    ]);

    const csvContent = "data:text/csv;charset=utf-8," 
      + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `FoundersFuel_Matches_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  /* ---------------------- MODERN UI STYLES ---------------------- */

  const containerStyle = {
    backgroundColor: '#1E1E2C', // Deep dark background
    minHeight: '100vh', 
    color: '#E0E0E0', 
    padding: '40px',
    fontFamily: 'Inter, sans-serif',
    paddingTop: `calc(40px + ${NAVBAR_HEIGHT})`, 
  };

  const topControlsStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '30px',
  };

  const buttonBaseStyle = {
    borderRadius: '8px',
    padding: '10px 18px',
    fontSize: '15px',
    cursor: 'pointer',
    fontWeight: '600',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    transition: 'background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease',
  }

  const newSearchButtonStyle = {
    ...buttonBaseStyle,
    backgroundColor: 'transparent',
    color: '#00BFFF', 
    border: '1px solid #00BFFF',
    textDecoration: 'none',
    ...(isNewSearchHovered && {
        backgroundColor: '#00BFFF', 
        color: '#1E1E2C', // Dark text on hover
        boxShadow: '0 4px 15px rgba(0, 191, 255, 0.6)',
        borderColor: '#00BFFF'
    })
  };

  const exportButtonStyle = {
    ...buttonBaseStyle,
    backgroundColor: '#7B68EE', 
    color: '#FFFFFF', 
    border: 'none',
    ...(isExportHovered && {
        backgroundColor: '#9370DB', 
        boxShadow: '0 4px 15px rgba(123, 104, 238, 0.8)',
    })
  };

  const summaryCardStyle = {
    backgroundColor: 'rgba(42, 42, 58, 0.6)', // Slightly transparent for glassmorphism base
    borderRadius: '16px', // Slightly more rounded
    padding: '30px', // More breathing room
    marginBottom: '35px',
    boxShadow: '0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 0 1px rgba(123, 104, 238, 0.3)', // Layered shadow
    backdropFilter: 'blur(10px)', // Glassmorphism effect
    WebkitBackdropFilter: 'blur(10px)',
    border: '1px solid rgba(123, 104, 238, 0.4)', // Subtle border
  };

  const descriptionTextStyle = { 
    color: '#A0A0B0',
    fontSize: '16px', 
    marginBottom: '25px', 
    fontWeight: '500',
  };

  const summaryGridStyle = {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '20px',
    marginBottom: '20px',
  };

  const tagBoxStyle = {
    backgroundColor: 'rgba(56, 56, 77, 0.7)', // Lighter card surface
    borderRadius: '10px',
    padding: '20px',
    boxShadow: '0 4px 10px rgba(0, 0, 0, 0.3)',
  };

  const tagHeaderStyle = { 
    color: '#FFFFFF', 
    fontSize: '17px', 
    fontWeight: '700', 
    margin: '0 0 15px 0',
    display: 'flex',
    alignItems: 'center',
    gap: '10px'
  };

  const tagBaseStyle = {
    borderRadius: '20px', // Pill shape for tags
    padding: '8px 16px',
    fontSize: '14px',
    fontWeight: '600',
    marginRight: '8px',
    marginBottom: '8px',
    display: 'inline-block',
  };

  const keywordTagStyle = {
    ...tagBaseStyle,
    backgroundColor: '#7B68EE', // Purple keywords
    color: '#FFFFFF',
  };

  const industryTagStyle = {
    ...tagBaseStyle,
    backgroundColor: '#4CAF50', // Green industries
    color: '#FFFFFF',
  };

  const insightsSectionStyle = {
    backgroundColor: 'rgba(56, 56, 77, 0.7)',
    borderRadius: '10px',
    padding: '20px',
    marginTop: '20px',
    borderLeft: '4px solid #7B68EE', // Accent border for insights
  };
  
  const insightStyle = {
    margin: '12px 0',
    fontSize: '15px',
    fontWeight: '500',
    color: '#C0C0D0',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  };

  const mainHeaderStyle = { 
    color: '#FFFFFF', 
    fontSize: '36px', // Larger title
    fontWeight: '900', 
    margin: '0 0 35px 0',
    letterSpacing: '-1px'
  };

  return (
    <div style={containerStyle}>
      <div style={topControlsStyle}>
        <Link 
          to="/investor-search" 
          style={newSearchButtonStyle}
          onMouseEnter={() => setIsNewSearchHovered(true)}
          onMouseLeave={() => setIsNewSearchHovered(false)}
        >
          ← New Search
        </Link>
        {investors.length > 0 && (
          <button 
            onClick={exportToCSV} 
            style={exportButtonStyle}
            onMouseEnter={() => setIsExportHovered(true)}
            onMouseLeave={() => setIsExportHovered(false)}
          >
            <span role="img" aria-label="Export">📥</span> Export CSV
          </button>
        )}
      </div>
      
      <div style={summaryCardStyle}>
        <p style={descriptionTextStyle}>
          Your Startup: <span style={{ color: '#FFFFFF', fontWeight: '600' }}>"{description}"</span>
        </p>
        
        <div style={summaryGridStyle}>
          {/* Extracted Keywords - Using a modern key icon */}
          <div style={tagBoxStyle}>
            <h4 style={tagHeaderStyle}><span role="img" aria-label="Keywords">🔑</span> Extracted Keywords</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {extracted_keywords.map(kw => (
                <span key={kw} style={keywordTagStyle}>{kw}</span>
              ))}
            </div>
          </div>

          {/* Matching Industries - Using a modern chart icon */}
          <div style={tagBoxStyle}>
            <h4 style={tagHeaderStyle}><span role="img" aria-label="Industries">📊</span> Matching Industries</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {matching_industries.map(ind => (
                <span key={ind} style={industryTagStyle}>{ind}</span>
              ))}
            </div>
          </div>
        </div>
        
        {/* Insights - Using a star icon */}
        <div style={insightsSectionStyle}>
          {insights.map((insight, index) => (
            <p key={index} style={insightStyle}>
              {insight.includes("Excellent") || insight.includes("Good") || insight.includes("actively") ? '⭐' : '💡'} {insight}
            </p>
          ))}
        </div>
      </div>
      
      <h2 style={mainHeaderStyle}>
        Found {investors.length} Highly-Matched Investors
      </h2>

      {investors.length > 0 ? (
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr', 
          gap: '30px', // Increased gap
        }}>
          {investors.map((investor, index) => (
            <InvestorCard 
              key={index} 
              investor={investor} 
              keywords={extracted_keywords} 
              maxFocusAreas={5}
            />
          ))}
        </div>
      ) : (
        <p style={{ color: '#A0A0B0', fontSize: '16px', textAlign: 'center', padding: '50px' }}>
          No suitable investors found. Try refining your description.
        </p>
      )}
    </div>
  );
};

export default InvestorResultsPage;