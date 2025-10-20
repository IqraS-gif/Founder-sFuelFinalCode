// // // import React from 'react';


// // // // Helper to format URL
// // // const formatWebsiteUrl = (url) => {
// // //   if (!url || typeof url !== 'string' || url.trim().toLowerCase() === 'n/a') {
// // //     return null;
// // //   }
// // //   let cleanUrl = url.trim();
// // //   if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
// // //     cleanUrl = `https://${cleanUrl}`;
// // //   }
// // //   return cleanUrl;
// // // };

// // // const InvestorCard = ({ investor, keywords }) => {
// // //   const { investor_name, website, fund_type, fund_stage, focus_areas, match_score } = investor;
// // //   const formattedWebsite = formatWebsiteUrl(website);

// // //   const isHighlighted = (area) => {
// // //     const lowerArea = area.toLowerCase();
// // //     return keywords.some(kw => lowerArea.includes(kw.toLowerCase()));
// // //   };

// // //   return (
// // //     <div className="investor-card">
// // //       <h3 className="investor-name">⭐ {investor_name}</h3>
      
// // //       <div className="info-row">
// // //         <strong>Website:</strong> 
// // //         {formattedWebsite ? (
// // //           <a href={formattedWebsite} target="_blank" rel="noopener noreferrer">{website}</a>
// // //         ) : (
// // //           <span>N/A</span>
// // //         )}
// // //       </div>
// // //       <div className="info-row"><strong>Type:</strong> <span>{fund_type}</span></div>
// // //       <div className="info-row"><strong>Stages:</strong> <span>{fund_stage}</span></div>
// // //       <div className="info-row"><strong>Match Score:</strong> <span className="match-score">{match_score}</span></div>
      
// // //       <div className="focus-areas">
// // //         <strong>Focus Areas:</strong>
// // //         <div className="focus-tags">
// // //           {focus_areas.slice(0, 8).map((area, index) => (
// // //             <span key={index} className={`focus-tag ${isHighlighted(area) ? 'highlighted' : ''}`}>
// // //               {area}
// // //             </span>
// // //           ))}
// // //         </div>
// // //       </div>
// // //     </div>
// // //   );
// // // };

// // // export default InvestorCard;


// // import React from 'react';

// // // Helper to format URL
// // const formatWebsiteUrl = (url) => {
// //   if (!url || typeof url !== 'string' || url.trim().toLowerCase() === 'n/a') {
// //     return null;
// //   }
// //   let cleanUrl = url.trim();
// //   // Ensure we display the clean URL without the 'https://' prefix for aesthetic purposes
// //   let displayUrl = cleanUrl.replace(/^(https?:\/\/(www\.)?)/i, '');
  
// //   if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
// //     cleanUrl = `https://${cleanUrl}`;
// //   }
  
// //   return { href: cleanUrl, display: displayUrl };
// // };

// // const InvestorCard = ({ investor, keywords }) => {
// //   const { investor_name, website, fund_type, fund_stage, focus_areas, match_score } = investor;
// //   const formattedWebsite = formatWebsiteUrl(website);
  
// //   // Ensure fund_stage is a string for display, handling array or string inputs
// //   const stagesDisplay = Array.isArray(fund_stage) ? fund_stage.join(', ') : fund_stage;

// //   const isHighlighted = (area) => {
// //     const lowerArea = area.toLowerCase();
// //     return keywords.some(kw => lowerArea.includes(kw.toLowerCase()));
// //   };

// //   const cardStyle = {
// //     backgroundColor: '#1C2130',
// //     borderRadius: '8px',
// //     padding: '24px',
// //     boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
// //     display: 'flex',
// //     flexDirection: 'column',
// //     gap: '8px',
// //     border: '1px solid #334155', // Subtle border
// //     fontFamily: 'Arial, sans-serif'
// //   };

// //   const nameStyle = {
// //     color: '#FFD700', // Gold color for the star icon
// //     fontSize: '18px',
// //     fontWeight: '600',
// //     margin: '0 0 12px 0',
// //     display: 'flex',
// //     alignItems: 'center',
// //   };

// //   const infoRowStyle = {
// //     margin: '4px 0',
// //     fontSize: '14px',
// //     color: '#B3B3B3',
// //     display: 'flex',
// //     alignItems: 'center',
// //     gap: '8px',
// //   };

// //   const websiteLinkStyle = {
// //     color: '#00BFFF',
// //     textDecoration: 'none',
// //     fontWeight: '500',
// //     // Hover effect using a style object isn't possible, but basic link style is fine
// //   };

// //   const strongStyle = {
// //     color: '#E0E0E0',
// //     fontWeight: '500',
// //   };

// //   const matchScoreStyle = {
// //     color: '#E91E63', // Pink/Red for Match Score
// //     fontWeight: '700',
// //   };

// //   const focusAreaContainerStyle = {
// //     marginTop: '15px',
// //     borderTop: '1px dashed #334155',
// //     paddingTop: '15px',
// //   };

// //   const focusTagsContainerStyle = {
// //     display: 'flex',
// //     flexWrap: 'wrap',
// //     gap: '8px',
// //     marginTop: '8px',
// //   };

// //   const focusTagBaseStyle = {
// //     borderRadius: '4px',
// //     padding: '4px 10px',
// //     fontSize: '12px',
// //     fontWeight: '500',
// //     color: '#FFFFFF',
// //     backgroundColor: '#334155', // Default grey background
// //   };
  
// //   const focusTagHighlightedStyle = {
// //     ...focusTagBaseStyle,
// //     backgroundColor: '#00BFFF', // Blue for matched keywords
// //     boxShadow: '0 0 5px rgba(0, 191, 255, 0.5)',
// //   };

// //   return (
// //     <div style={cardStyle}>
// //       <h3 style={nameStyle}>
// //         ⭐ {investor_name}
// //       </h3>
      
// //       <div style={infoRowStyle}>
// //         <strong style={strongStyle}>🌐 Website:</strong> 
// //         {formattedWebsite ? (
// //           <a href={formattedWebsite.href} target="_blank" rel="noopener noreferrer" style={websiteLinkStyle}>
// //             {formattedWebsite.display}
// //           </a>
// //         ) : (
// //           <span style={{ color: '#FF6347' }}>N/A</span>
// //         )}
// //       </div>
      
// //       <div style={infoRowStyle}>
// //         <strong style={strongStyle}>🍊 Type:</strong> 
// //         <span style={{ color: '#FFFFFF' }}>{fund_type}</span>
// //       </div>
      
// //       <div style={infoRowStyle}>
// //         <strong style={strongStyle}>◾ Stages:</strong> 
// //         <span style={{ color: '#FFFFFF' }}>{stagesDisplay}</span>
// //       </div>
      
// //       <div style={infoRowStyle}>
// //         <strong style={{...strongStyle, color: '#E91E63'}}>💕 Match Score:</strong> 
// //         <span style={matchScoreStyle}>{match_score}</span>
// //       </div>
      
// //       <div style={focusAreaContainerStyle}>
// //         <strong style={strongStyle}>Focus Areas:</strong>
// //         <div style={focusTagsContainerStyle}>
// //           {focus_areas.slice(0, 8).map((area, index) => (
// //             <span 
// //               key={index} 
// //               style={isHighlighted(area) ? focusTagHighlightedStyle : focusTagBaseStyle}
// //             >
// //               {area}
// //             </span>
// //           ))}
// //         </div>
// //       </div>
// //     </div>
// //   );
// // };

// // export default InvestorCard;


// // src/components/analysis/InvestorCard.jsx - Updated for Image 3 + Limits

// import React, { useState } from 'react';

// const formatWebsiteUrl = (url) => {
//   if (!url || typeof url !== 'string' || url.trim().toLowerCase() === 'n/a') {
//     return null;
//   }
//   let cleanUrl = url.trim();
//   let displayUrl = cleanUrl.replace(/^(https?:\/\/(www\.)?)/i, '');
  
//   if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
//     cleanUrl = `https://${cleanUrl}`;
//   }
  
//   return { href: cleanUrl, display: displayUrl };
// };

// // Added maxFocusAreas prop
// const InvestorCard = ({ investor, keywords, maxFocusAreas = 5 }) => {
//   const { investor_name, website, fund_type, fund_stage, focus_areas, match_score } = investor;
//   const formattedWebsite = formatWebsiteUrl(website);
//   const stagesDisplay = Array.isArray(fund_stage) ? fund_stage.join(', ') : fund_stage;

//   const [isCardHovered, setIsCardHovered] = useState(false);

//   const isHighlighted = (area) => {
//     const lowerArea = area.toLowerCase();
//     return keywords.some(kw => lowerArea.includes(kw.toLowerCase()));
//   };

//   /* --- APPLY FOCUS AREA LIMIT --- */
//   const highlightedAreas = focus_areas.filter(isHighlighted);
//   const otherAreas = focus_areas.filter(area => !isHighlighted(area));
  
//   // Take all highlighted areas, then fill remaining spots (maxFocusAreas) with other areas
//   const finalFocusAreas = [
//     ...highlightedAreas, 
//     ...otherAreas.slice(0, maxFocusAreas - highlightedAreas.length > 0 ? maxFocusAreas - highlightedAreas.length : 0)
//   ];
//   /* ------------------------------- */

//   /* ---------------------- INLINE STYLES (MATCHING IMAGE 3) ---------------------- */

//   const cardStyle = {
//     backgroundColor: '#2A2A3A', 
//     borderRadius: '12px',
//     padding: '25px', 
//     boxShadow: '0 6px 15px rgba(0, 0, 0, 0.3)',
//     border: '1px solid #404050',
//     display: 'flex',
//     flexDirection: 'column',
//     fontFamily: 'Inter, sans-serif',
//     transition: 'transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease',
//     ...(isCardHovered && { // Enhanced Hover Effect
//         transform: 'translateY(-5px)', 
//         boxShadow: '0 12px 25px rgba(0, 0, 0, 0.5), 0 0 0 2px #7B68EE', 
//         borderColor: '#7B68EE',
//     })
//   };

//   const nameAndScoreStyle = {
//     display: 'flex',
//     justifyContent: 'space-between',
//     alignItems: 'center',
//     marginBottom: '20px',
//     borderBottom: '1px solid #38384D',
//     paddingBottom: '20px',
//   };

//   const nameStyle = {
//     color: '#FFFFFF', 
//     fontSize: '20px',
//     fontWeight: '700',
//     margin: '0',
//   };

//   const matchScoreBubbleStyle = {
//     backgroundColor: '#00BFFF', // Blue Bubble
//     color: '#FFFFFF',
//     borderRadius: '8px', 
//     padding: '8px 12px',
//     fontSize: '16px',
//     fontWeight: '700',
//     boxShadow: '0 2px 8px rgba(0, 191, 255, 0.4)',
//   };

//   const infoRowStyle = {
//     display: 'flex',
//     alignItems: 'center',
//     marginBottom: '8px', 
//     fontSize: '15px',
//   };

//   const infoLabelStyle = {
//     color: '#A0A0B0', // Gray label
//     fontWeight: '500',
//     width: '80px', // Fixed width for alignment
//     flexShrink: '0',
//   };

//   const infoValueStyle = {
//     color: '#FFFFFF', 
//     fontWeight: '500',
//   };

//   const websiteLinkStyle = {
//     color: '#00BFFF', 
//     textDecoration: 'none',
//     fontWeight: '500',
//     display: 'flex',
//     alignItems: 'center',
//     gap: '5px',
//     transition: 'color 0.2s ease',
//     // Simulate hover with inline state
//     ...(isCardHovered && { color: '#87CEEB' }),
//   };

//   const focusAreaContainerStyle = {
//     marginTop: '25px',
//   };

//   const focusAreaLabelStyle = {
//     color: '#A0A0B0', 
//     fontSize: '15px', 
//     fontWeight: '500', 
//     marginBottom: '12px',
//     display: 'block'
//   };

//   const focusTagsContainerStyle = {
//     display: 'flex',
//     flexWrap: 'wrap',
//     gap: '8px',
//   };

//   const focusTagBaseStyle = {
//     borderRadius: '6px',
//     padding: '7px 14px',
//     fontSize: '13px',
//     fontWeight: '600',
//     color: '#C0C0D0',
//     backgroundColor: '#38384D', // Darker gray for non-highlighted tags
//   };
  
//   const focusTagHighlightedStyle = {
//     ...focusTagBaseStyle,
//     backgroundColor: '#7B68EE', // Purple for matched keywords
//     color: '#FFFFFF', 
//     fontWeight: '700',
//     boxShadow: '0 0 5px rgba(123, 104, 238, 0.7)',
//   };

//   return (
//     <div 
//       style={cardStyle}
//       onMouseEnter={() => setIsCardHovered(true)}
//       onMouseLeave={() => setIsCardHovered(false)}
//     >
//       <div style={nameAndScoreStyle}>
//         <h3 style={nameStyle}>{investor_name}</h3>
//         {/* Match score is displayed as a percentage, matching the image */}
//         <div style={matchScoreBubbleStyle}>{match_score}%</div>
//       </div>
      
//       <div style={infoRowStyle}>
//         <span style={infoLabelStyle}>Website</span> 
//         {formattedWebsite ? (
//           <a 
//             href={formattedWebsite.href} 
//             target="_blank" 
//             rel="noopener noreferrer" 
//             style={websiteLinkStyle}
//           >
//             {formattedWebsite.display} <span role="img" aria-label="External link">↗</span>
//           </a>
//         ) : (
//           <span style={{ color: '#FF6347', ...infoValueStyle }}>N/A</span>
//         )}
//       </div>
      
//       <div style={infoRowStyle}>
//         <span style={infoLabelStyle}>Type</span> 
//         <span style={infoValueStyle}>{fund_type}</span>
//       </div>
      
//       <div style={infoRowStyle}>
//         <span style={infoLabelStyle}>Stages</span> 
//         <span style={infoValueStyle}>{stagesDisplay}</span>
//       </div>
      
//       <div style={focusAreaContainerStyle}>
//         <span style={focusAreaLabelStyle}>Focus Areas</span>
//         <div style={focusTagsContainerStyle}>
//           {finalFocusAreas.map((area, index) => (
//             <span 
//               key={index} 
//               style={isHighlighted(area) ? focusTagHighlightedStyle : focusTagBaseStyle}
//             >
//               {area}
//             </span>
//           ))}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default InvestorCard;


// src/components/analysis/InvestorCard.jsx

import React, { useState } from 'react';

const formatWebsiteUrl = (url) => {
  if (!url || typeof url !== 'string' || url.trim().toLowerCase() === 'n/a') {
    return null;
  }
  let cleanUrl = url.trim();
  let displayUrl = cleanUrl.replace(/^(https?:\/\/(www\.)?)/i, '');
  
  if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
    cleanUrl = `https://${cleanUrl}`;
  }
  
  return { href: cleanUrl, display: displayUrl };
};

const InvestorCard = ({ investor, keywords, maxFocusAreas = 5 }) => {
  const { investor_name, website, fund_type, fund_stage, focus_areas, match_score } = investor;
  const formattedWebsite = formatWebsiteUrl(website);
  const stagesDisplay = Array.isArray(fund_stage) ? fund_stage.join(', ') : fund_stage;

  const [isCardHovered, setIsCardHovered] = useState(false);

  const isHighlighted = (area) => {
    const lowerArea = area.toLowerCase();
    return keywords.some(kw => lowerArea.includes(kw.toLowerCase()));
  };

  /* --- APPLY FOCUS AREA LIMIT --- */
  const highlightedAreas = focus_areas.filter(isHighlighted);
  const otherAreas = focus_areas.filter(area => !isHighlighted(area));
  
  const areasToTakeFromOthers = Math.max(0, maxFocusAreas - highlightedAreas.length);
  const finalFocusAreas = [
      ...highlightedAreas, 
      ...otherAreas.slice(0, areasToTakeFromOthers)
  ];
  /* ------------------------------- */

  /* ---------------------- MODERN UI STYLES ---------------------- */

  const cardStyle = {
    backgroundColor: 'rgba(42, 42, 58, 0.5)', // Slightly transparent background
    borderRadius: '16px',
    padding: '30px', 
    boxShadow: '0 8px 20px rgba(0, 0, 0, 0.4)', // Base shadow
    border: '1px solid rgba(123, 104, 238, 0.3)', // Subtle border
    display: 'flex',
    flexDirection: 'column',
    fontFamily: 'Inter, sans-serif',
    transition: 'transform 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.4s cubic-bezier(0.25, 0.8, 0.25, 1)',
    backdropFilter: 'blur(8px)', // Glassmorphism effect
    WebkitBackdropFilter: 'blur(8px)',
    
    ...(isCardHovered && { // Advanced Hover Effect
        transform: 'translateY(-8px)', // Pronounced lift
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.6), 0 0 0 3px #00FFFF', // Stronger shadow + Cyan glow border
        borderColor: '#00FFFF',
    })
  };

  const nameAndScoreStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
    borderBottom: '1px solid #38384D',
    paddingBottom: '20px',
  };

  const nameStyle = {
    color: '#00FFFF', // Classic Cyan Color
    fontSize: '22px', // Slightly larger name
    fontWeight: '800',
    margin: '0',
  };

  const matchScoreBubbleStyle = {
    backgroundColor: '#00BFFF', 
    color: '#1E1E2C', // Dark text for contrast
    borderRadius: '10px', 
    padding: '8px 12px',
    fontSize: '16px',
    fontWeight: '800',
    boxShadow: '0 2px 8px rgba(0, 191, 255, 0.5)',
  };

  const infoRowStyle = {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '10px', 
    fontSize: '15px',
    borderBottom: '1px dotted rgba(160, 160, 176, 0.1)', // Subtle separator
    paddingBottom: '5px'
  };

  const infoLabelStyle = {
    color: '#c3c3cdff',
    fontWeight: '500',
    width: '100px', // More width for cleaner alignment
    flexShrink: '0',
    display: 'flex',
    alignItems: 'center',
    gap: '5px'
  };

  const infoValueStyle = {
    color: '#FFFFFF', 
    fontWeight: '500',
  };

  const websiteLinkStyle = {
    color: '#00BFFF', 
    textDecoration: 'none',
    fontWeight: '500',
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    transition: 'color 0.2s ease',
    ...(isCardHovered && { color: '#00FFFF' }), // Cyan on link hover
  };

  const focusAreaContainerStyle = {
    marginTop: '25px',
    paddingTop: '20px',
    borderTop: '1px solid #38384D', 
  };

  const focusAreaLabelStyle = {
    color: '#A0A0B0', 
    fontSize: '15px', 
    fontWeight: '500', 
    marginBottom: '12px',
    display: 'block'
  };

  const focusTagsContainerStyle = {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
  };

  const focusTagBaseStyle = {
    borderRadius: '20px',
    padding: '8px 16px',
    fontSize: '13px',
    fontWeight: '600',
    color: '#C0C0D0',
    backgroundColor: '#38384D', 
  };
  
  const focusTagHighlightedStyle = {
    ...focusTagBaseStyle,
    backgroundColor: '#7B68EE', 
    color: '#FFFFFF', 
    fontWeight: '700',
    boxShadow: '0 0 5px rgba(123, 104, 238, 0.7)',
  };

  return (
    <div 
      style={cardStyle}
      onMouseEnter={() => setIsCardHovered(true)}
      onMouseLeave={() => setIsCardHovered(false)}
    >
      <div style={nameAndScoreStyle}>
        <h3 style={nameStyle}>{investor_name}</h3>
        <div style={matchScoreBubbleStyle}>{match_score}%</div>
      </div>
      
      <div style={infoRowStyle}>
        <span style={infoLabelStyle}><span role="img" aria-label="Website">🌐</span> Website</span> 
        {formattedWebsite ? (
          <a 
            href={formattedWebsite.href} 
            target="_blank" 
            rel="noopener noreferrer" 
            style={websiteLinkStyle}
          >
            {formattedWebsite.display} <span role="img" aria-label="External link">↗</span>
          </a>
        ) : (
          <span style={{ color: '#FF6347', ...infoValueStyle }}>N/A</span>
        )}
      </div>
      
      <div style={infoRowStyle}>
        <span style={infoLabelStyle}><span role="img" aria-label="Type">💼</span> Type</span> 
        <span style={infoValueStyle}>{fund_type}</span>
      </div>
      
      <div style={infoRowStyle}>
        <span style={infoLabelStyle}><span role="img" aria-label="Stages">📌</span> Stages</span> 
        <span style={infoValueStyle}>{stagesDisplay}</span>
      </div>
      
      <div style={focusAreaContainerStyle}>
        <span style={focusAreaLabelStyle}>Focus Areas</span>
        <div style={focusTagsContainerStyle}>
          {finalFocusAreas.map((area, index) => (
            <span 
              key={index} 
              style={isHighlighted(area) ? focusTagHighlightedStyle : focusTagBaseStyle}
            >
              {area}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default InvestorCard;