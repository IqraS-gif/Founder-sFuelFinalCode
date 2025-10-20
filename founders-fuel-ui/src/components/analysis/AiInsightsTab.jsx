// // src/components/analysis/AiInsightsTab.jsx

// import React from 'react';
// import styled from 'styled-components';

// // --- Styled Components ---

// const InsightsGrid = styled.div`
//   display: flex;
//   flex-direction: column;
//   gap: 1.5rem;
// `;

// const InsightCard = styled.div`
//   background-color: #1E293B;
//   border: 1px solid #334155;
//   border-radius: 12px;
//   padding: 1.5rem;
//   display: grid;
//   grid-template-columns: 200px 1fr; /* Fixed width for score, flexible for feedback */
//   gap: 2rem;
//   align-items: center;
// `;

// const ScoreSection = styled.div`
//   text-align: center;
//   border-right: 1px solid #334155;
//   padding-right: 2rem;
  
//   h3 {
//     margin-top: 0;
//     color: #94a3b8;
//     font-weight: 500;
//     text-transform: uppercase;
//     font-size: 0.9rem;
//     letter-spacing: 0.5px;
//   }
  
//   .score-display {
//     font-size: 4rem;
//     font-weight: 700;
//     color: #fff;
//     line-height: 1;

//     span {
//       font-size: 2rem;
//       font-weight: 500;
//       color: #64748b;
//     }
//   }

//   .progress-bg {
//     width: 100%;
//     height: 8px;
//     background-color: #334155;
//     border-radius: 4px;
//     margin-top: 1rem;
//     overflow: hidden; /* Ensures the foreground bar has rounded corners */
//   }
//   .progress-fg {
//     height: 100%;
//     background-color: #14f1d9;
//     border-radius: 4px;
//   }
// `;

// const FeedbackSection = styled.div`
//   h4 {
//     margin-top: 0;
//     font-size: 1.1rem;
//     color: #cbd5e1;
//   }
//   p {
//     margin: 0;
//     font-size: 1rem;
//     line-height: 1.7;
//     color: #94a3b8;
//   }
// `;

// // --- The Component ---
// function AiInsightsTab({ data }) {
//   const { competitive_moat, gtm_strategy, team_market_fit } = data;

//   const strategicAreas = [
//     {
//       title: 'Competitive Advantage',
//       description: 'Your ability to defend against competition.',
//       data: competitive_moat,
//     },
//     {
//       title: 'Go-to-Market Strategy',
//       description: 'Clarity and viability of your market approach.',
//       data: gtm_strategy,
//     },
//     {
//       title: 'Team-Market Fit',
//       description: 'How well your team capabilities match market needs.',
//       data: team_market_fit,
//     },
//   ];

//   return (
//     <InsightsGrid>
//       {strategicAreas.map((area) => (
//         <InsightCard key={area.title}>
//           <ScoreSection>
//             <h3>{area.title}</h3>
//             <div className="score-display">
//               {area.data.score}
//               <span> /10</span>
//             </div>
//             <div className="progress-bg">
//               <div className="progress-fg" style={{ width: `${area.data.score * 10}%` }}></div>
//             </div>
//           </ScoreSection>
//           <FeedbackSection>
//             <h4>Detailed Feedback</h4>
//             <p>{area.data.feedback}</p>
//           </FeedbackSection>
//         </InsightCard>
//       ))}
//     </InsightsGrid>
//   );
// }

// export default AiInsightsTab;


// src/components/analysis/AiInsightsTab.jsx

import React from 'react';
import styled from 'styled-components';

// --- Helper for Score-based Colors ---

// Function to determine color based on score
const getColorForScore = (score) => {
  if (score >= 7) return '#14f1d9'; // Strong score: Teal/Cyan
  if (score >= 4) return '#FCD34D'; // Moderate score: Amber/Yellow
  return '#F87171'; // Low score: Red/Coral
};

// Function to determine background color based on score (lighter/darker shades for contrast)
const getBgColorForScore = (score) => {
  if (score >= 7) return 'rgba(20, 241, 217, 0.1)'; // Lightest teal/cyan
  if (score >= 4) return 'rgba(252, 211, 77, 0.1)'; // Lightest amber/yellow
  return 'rgba(248, 113, 113, 0.1)'; // Lightest red/coral
};

// --- Styled Components ---

const InsightsGrid = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem; /* Increased gap for better separation */
`;

const InsightCard = styled.div`
  background-color: ${props => getBgColorForScore(props.$score)}; /* Dynamic background highlight */
  border: 1px solid ${props => getColorForScore(props.$score)}; /* Dynamic border color */
  border-radius: 16px; /* Slightly more rounded corners */
  padding: 2rem; /* Increased padding */
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 3rem; /* Increased gap between score and feedback */
  align-items: center;
  transition: all 0.3s ease; /* Smooth transition for hover effects */

  &:hover {
    box-shadow: 0 4px 15px rgba(20, 241, 217, 0.15); /* Subtle lift and glow on hover */
    transform: translateY(-2px);
  }
`;

const ScoreSection = styled.div`
  text-align: center;
  border-right: 2px solid ${props => getColorForScore(props.$score)}; /* Thicker, colored separator */
  padding-right: 3rem; /* Increased padding for visual balance */
  
  h3 {
    margin-top: 0;
    color: #cbd5e1; /* Changed title color for better readability/contrast */
    font-weight: 600;
    text-transform: uppercase;
    font-size: 1rem; /* Slightly larger title */
    letter-spacing: 1px; /* Increased letter-spacing */
  }
  
  .score-display {
    font-size: 5rem; /* Larger, more impactful score number */
    font-weight: 800;
    color: ${props => getColorForScore(props.$score)}; /* Dynamic score color */
    line-height: 1;
    margin: 1rem 0;

    span {
      font-size: 2.2rem;
      font-weight: 500;
      color: #94a3b8;
    }
  }

  /* Enhanced Progress Bar */
  .progress-bg {
    width: 100%;
    height: 10px; /* Thicker bar */
    background-color: #334155;
    border-radius: 5px;
    margin-top: 1.5rem;
    overflow: hidden;
    position: relative;
  }
  
  .progress-fg {
    height: 100%;
    background-color: ${props => getColorForScore(props.$score)}; /* Dynamic progress bar color */
    border-radius: 5px;
    transition: width 0.5s ease-out; /* Animation for width change */
  }

  .progress-indicator {
    position: absolute;
    top: -1.5rem;
    right: 0;
    transform: translateX(50%);
    color: #cbd5e1;
    font-size: 0.8rem;
  }
`;

const FeedbackSection = styled.div`
  h4 {
    margin-top: 0;
    font-size: 1.25rem; /* Larger, more prominent section title */
    color: #fff; /* White title for strong focus */
    margin-bottom: 0.5rem;
  }
  
  .description {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1rem;
    font-style: italic;
  }

  p {
    margin: 0;
    font-size: 1.05rem; /* Slightly larger feedback text */
    line-height: 1.6;
    color: #cbd5e1; /* Lighter color for better readability against dark background */
  }
`;

// --- The Component ---
function AiInsightsTab({ data }) {
  const { competitive_moat, gtm_strategy, team_market_fit } = data;

  const strategicAreas = [
    {
      title: 'Competitive Advantage',
      description: 'Your ability to defend against competition (Moat Strength).',
      data: competitive_moat,
    },
    {
      title: 'Go-to-Market Strategy',
      description: 'Clarity and viability of your market approach (GTM Viability).',
      data: gtm_strategy,
    },
    {
      title: 'Team-Market Fit',
      description: 'How well your team capabilities match market needs (TMF Rating).',
      data: team_market_fit,
    },
  ];

  return (
    <InsightsGrid>
      {strategicAreas.map((area) => (
        <InsightCard key={area.title} $score={area.data.score}>
          <ScoreSection $score={area.data.score}>
            <h3>{area.title}</h3>
            <div className="score-display">
              {area.data.score}
              <span> /10</span>
            </div>
            <div className="progress-bg">
              <div className="progress-fg" style={{ width: `${area.data.score * 10}%` }}></div>
            </div>
          </ScoreSection>
          <FeedbackSection>
            <h4>{area.title} Analysis Summary 📝</h4>
            <p className="description">{area.description}</p>
            <p>{area.data.feedback}</p>
          </FeedbackSection>
        </InsightCard>
      ))}
    </InsightsGrid>
  );
}

export default AiInsightsTab;