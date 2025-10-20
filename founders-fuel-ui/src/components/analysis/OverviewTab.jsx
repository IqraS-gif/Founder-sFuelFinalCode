// // 
// // src/components/analysis/OverviewTab.jsx
// // FINAL VERSION WITH SCORE BREAKDOWN, ENHANCED UI, HOVER EFFECTS ON ALL CARDS, AND MAIN SCORE BARS

// import React from 'react';
// import styled from 'styled-components';
// import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

// // --- Helper Functions and Data ---
// const getMetricIcon = (title) => {
//   switch (title) {
//     case 'Investment Readiness': return '🚀';
//     case 'Competitive Moat': return '🛡️';
//     case 'GTM Strategy': return '🗺️';
//     case 'Team-Market Fit': return '🤝';
//     default: return '📊';
//   }
// }

// const dimensionDescriptions = {
//   clarity: 'How clear and understandable your pitch is to investors.',
//   completeness: 'Whether all essential elements of a pitch are covered.',
//   persuasiveness: 'How compelling and convincing your arguments are.',
//   data_driven: 'Quality and quantity of supporting data and metrics.',
// };

// // Function to determine color based on score (Green/Yellow/Red) - for score bar
// const getColorForScore = (score) => {
//     if (score >= 8) return '#34d399'; // High - Green
//     if (score >= 5) return '#fbbf24'; // Medium - Yellow
//     return '#f87171'; // Low - Red
// }


// // --- Styled Components ---

// const TabGrid = styled.div`
//   display: grid;
//   grid-template-columns: repeat(12, 1fr);
//   gap: 1.5rem;
//   width: 100%;
// `;

// // 1. DimensionalScoresContainer is declared first to avoid ReferenceError in Card
// const DimensionalScoresContainer = styled.div`
//   grid-column: 1 / -1; 
//   display: grid;
//   grid-template-columns: repeat(2, 1fr); 
//   gap: 3rem; 
//   align-items: center;
//   padding: 2.5rem; 
//   background-color: #1E293B; /* Include Card's base styles */
//   border: 1px solid #334155;
//   border-radius: 12px;
//   box-shadow: 0 4px 12px rgba(0,0,0,0.2);
//   transition: all 0.3s ease-in-out; 

//   /* Explicitly prevent lifting on hover for this large card */
//   &:hover {
//       transform: translateY(0);
//       box-shadow: 0 4px 12px rgba(0,0,0,0.2);
//       border-color: #334155;
//   }

//   @media (max-width: 1024px) {
//     grid-template-columns: 1fr; 
//   }
// `;

// // 2. Card is declared next, referencing DimensionalScoresContainer in the selector
// const Card = styled.div`
//   background-color: #1E293B;
//   border: 1px solid #334155;
//   border-radius: 12px;
//   padding: 1.5rem;
//   box-shadow: 0 4px 12px rgba(0,0,0,0.2);
//   transition: all 0.3s ease-in-out; 
//   cursor: default;

//   /* Universal Hover Effect applied only if it's NOT the DimensionalScoresContainer */
//   &:not(${DimensionalScoresContainer}) { 
//     &:hover {
//       transform: translateY(-4px); 
//       border-color: #475569; 
//       box-shadow: 0 8px 20px rgba(0,0,0,0.3);
//     }
//   }
// `;

// const VerdictCard = styled(Card)`
//   grid-column: 1 / -1;
//   border-left: 4px solid #14f1d9;

//   &:hover {
//     border-color: #14f1d9; 
//     transform: translateY(0);
//   }

//   h3 { 
//     display: flex; 
//     align-items: center; 
//     gap: 0.75rem; 
//     margin-top: 0; 
//     color: #fff; 
//     font-size: 1.6rem; 
//     font-weight: 800; 
// }
//   h3 span { color: #14f1d9; font-size: 2rem; }
//   p { 
//     font-size: 1.15rem; 
//     line-height: 1.7; 
//     color: #cbd5e1; 
//     margin: 0; 
//     font-style: italic;
// }
// `;

// const MetricCard = styled(Card)`
//   grid-column: span 3;
//   display: flex;
//   flex-direction: column;
//   cursor: pointer;

//   &:hover {
//     border-color: #14f1d9; 
//     box-shadow: 0 6px 16px rgba(20, 241, 217, 0.15); 
//   }

//   h4 { 
//     margin-top: 0; 
//     color: #94a3b8; 
//     font-weight: 500; 
//     text-transform: uppercase; 
//     font-size: 0.8rem; 
//     letter-spacing: 0.5px;
//     display: flex;
//     align-items: center;
//     gap: 0.5rem;
//     border-bottom: 1px solid #334155;
//     padding-bottom: 0.75rem;
//     margin-bottom: 1rem;
//   }
//   .score { 
//     font-size: 3.8rem; 
//     font-weight: 900; 
//     color: #fff; 
//     line-height: 1;
//     margin-top: auto; 
//   }
//   .score span { 
//     font-size: 1.8rem; 
//     font-weight: 500; 
//     color: #64748b; 
//   }
// `;

// // Score Bar Components
// const MetricScoreBar = styled.div`
//     width: 100%;
//     height: 5px;
//     background-color: #334155;
//     border-radius: 2.5px;
//     margin-top: 1rem;
//     overflow: hidden;
// `;

// const ScoreBarFill = styled.div`
//     height: 100%;
//     border-radius: 2.5px;
//     background-color: ${props => props.$color}; 
//     width: ${props => props.$width}%;
//     transition: width 1s ease-out;
// `;


// const ChartWrapper = styled.div`
//   /* Chart lives here */
// `;

// const ScoreBreakdown = styled.div`
//   display: flex;
//   flex-direction: column;
//   gap: 1.5rem;
//   padding-right: 1rem;
// `;

// const BreakdownItem = styled.div`
//   .header {
//     display: flex;
//     justify-content: space-between;
//     align-items: center;
//     margin-bottom: 0.25rem;
//   }
//   .title { 
//     font-weight: 700; 
//     color: #fff; 
//     font-size: 1.1rem; 
//     transition: color 0.3s ease-in-out;
//   }
//   .score { 
//     font-weight: 700; 
//     color: #14f1d9; 
//     font-size: 1.1rem;
//   }
//   .description { 
//     font-size: 0.85rem; 
//     color: #94a3b8; 
//     margin: 0; 
//   }
//   
//   /* Progress Bar */
//   .progress-bg {
//     width: 100%;
//     height: 8px; 
//     background-color: #334155;
//     border-radius: 4px;
//     margin-top: 0.75rem;
//   }
//   .progress-fg {
//     height: 100%;
//     background-color: #14f1d9;
//     border-radius: 4px;
//     transition: width 0.5s ease-out, background-color 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
//   }

//   &:hover {
//     .progress-fg {
//       background-color: #0dcfb6;
//       box-shadow: 0 0 8px rgba(20, 241, 217, 0.4); 
//     }
//     .title {
//       color: #14f1d9; 
//     }
//   }
// `;

// const ListCard = styled(Card)`
//   grid-column: span 6;
//   h4 { 
//     display: flex; 
//     align-items: center; 
//     gap: 0.75rem; 
//     margin-top: 0; 
//     font-size: 1.5rem; 
//     font-weight: 700; 
//     border-bottom: 1px solid #334155;
//     padding-bottom: 0.75rem;
//     margin-bottom: 1.5rem;
//  }
//   ul { list-style: none; padding-left: 0; margin: 0; }
// `;

// const ListItem = styled.li`
//   margin-bottom: 1rem; line-height: 1.6; padding-left: 1.5rem; position: relative; color: #cbd5e1;
//   &::before { content: '•'; position: absolute; left: 0; font-size: 1.2rem; font-weight: 900; transition: color 0.3s ease-in-out; }

//   transition: color 0.3s ease-in-out, transform 0.2s ease-in-out; 
//   cursor: pointer; 

//   &:hover {
//     color: #fff; 
//     transform: translateX(5px); 
//   }
//   &:hover::before {
//     color: #14f1d9; 
//   }
// `;

// const StrengthsListCard = styled(ListCard)`
//   h4 { color: #34d399; }
//   ${ListItem}::before { color: #34d399; }
//   ${ListItem}:hover { color: #34d399; }
// `;

// const ImprovementsListCard = styled(ListCard)`
//   h4 { color: #f87171; }
//   ${ListItem}::before { color: #f87171; }
//   ${ListItem}:hover { color: #f87171; }
// `;


// // --- The Component ---
// function OverviewTab({ data }) {
//   const chartData = Object.entries(data.dimensional_scores).map(([key, value]) => ({
//     dimension: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
//     score: value,
//   }));

//   const mainMetrics = [
//     { title: 'Investment Readiness', score: Math.round(data.investment_readiness_score) },
//     { title: 'Competitive Moat', score: data.competitive_moat.score },
//     { title: 'GTM Strategy', score: data.gtm_strategy.score },
//     { title: 'Team-Market Fit', score: data.team_market_fit.score },
//   ];

//   return (
//     <TabGrid>
//       <VerdictCard>
//         <h3><span>🎯</span> VC Verdict</h3>
//         <p>"{data.vc_verdict}"</p>
//       </VerdictCard>

//       {/* Map over the main metrics, now including the score bar */}
//       {mainMetrics.map((metric) => (
//         <MetricCard key={metric.title}>
//           <h4>{getMetricIcon(metric.title)} {metric.title}</h4>
//           <div className="score">{metric.score}<span> /10</span></div>
          
//           {/* Score Bar Implementation */}
//           <MetricScoreBar>
//               <ScoreBarFill 
//                   $width={metric.score * 10} 
//                   $color={getColorForScore(metric.score)}
//               />
//           </MetricScoreBar>
//         </MetricCard>
//       ))}
//       
//       {/* --- Dimensional Scores with Chart and Breakdown --- */}
//       <DimensionalScoresContainer>
//         <ChartWrapper>
//           <h3 style={{ color: '#fff', textAlign: 'center', marginBottom: '1.5rem', fontSize: '1.4rem', fontWeight: 700 }}>Dimensional Score Radar</h3>
//           <ResponsiveContainer width="100%" height={300}>
//             <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
//               <PolarGrid stroke="#334155" />
//               <PolarAngleAxis dataKey="dimension" tick={{ fill: '#94a3b8', fontSize: 14 }} />
//               <PolarRadiusAxis angle={30} domain={[0, 10]} tick={false} axisLine={false} />
//               {/* Define gradient for a richer look */}
//              <defs>
//                <linearGradient id="scoreColor" x1="0" y1="0" x2="0" y2="1">
//                  <stop offset="5%" stopColor="#14f1d9" stopOpacity={0.8}/>
//                  <stop offset="95%" stopColor="#14f1d9" stopOpacity={0.3}/>
//                </linearGradient>
//              </defs>
//               <Radar name="Score" dataKey="score" stroke="#14f1d9" fill="url(#scoreColor)" fillOpacity={0.8} strokeWidth={2} />
//             </RadarChart>
//           </ResponsiveContainer>
//         </ChartWrapper>

//         <ScoreBreakdown>
//           <h3 style={{ color: '#fff', marginTop: '0', marginBottom: '1rem', fontSize: '1.4rem', fontWeight: 700 }}>Dimensional Breakdown</h3>
//           {Object.entries(data.dimensional_scores).map(([key, value]) => (
//             <BreakdownItem key={key}>
//               <div className="header">
//                 <span className="title">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
//                 <span className="score">{value}<span style={{ color: '#64748b', fontSize: '1rem' }}>/10</span></span>
//               </div>
//               <p className="description">{dimensionDescriptions[key] || ''}</p>
//               <div className="progress-bg">
//                 <div className="progress-fg" style={{ width: `${value * 10}%` }}></div>
//               </div>
//             </BreakdownItem>
//           ))}
//         </ScoreBreakdown>
//       </DimensionalScoresContainer>
//       
//       <StrengthsListCard>
//         <h4>✅ Key Strengths</h4>
//         <ul>
//           {data.key_strengths.map((strength, index) => (
//             <ListItem key={index}>{strength}</ListItem>
//           ))}
//         </ul>
//       </StrengthsListCard>

//       <ImprovementsListCard>
//         <h4>🚩 Priority Improvements</h4>
//         <ul>
//           {data.key_improvement_suggestions.map((improvement, index) => (
//             <ListItem key={index}>{improvement}</ListItem>
//           ))}
//         </ul>
//       </ImprovementsListCard>
//     </TabGrid>
//   );
// }

// export default OverviewTab;


// src/components/analysis/OverviewTab.jsx
// FIXED VERSION - SCORES NOW DISPLAY CORRECTLY
// src/components/analysis/OverviewTab.jsx
// FIXED VERSION - PROPER COMPONENT ORDER
// src/components/analysis/OverviewTab.jsx
// FIXED VERSION - PROPER COMPONENT ORDER & STYLING CHANGES

import React, { useState, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

// --- Animations ---
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
`;

const pulse = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.02); }
  100% { transform: scale(1); }
`;

const slideIn = keyframes`
  from { transform: translateX(-20px); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
`;

// --- Helper Functions and Data ---
const getMetricIcon = (title) => {
  switch (title) {
    case 'Investment Readiness': return '🚀';
    case 'Competitive Moat': return '🛡️';
    case 'GTM Strategy': return '🗺️';
    case 'Team-Market Fit': return '🤝';
    default: return '📊';
  }
};

const dimensionDescriptions = {
  clarity: 'How clear and understandable your pitch is to investors.',
  completeness: 'Whether all essential elements of a pitch are covered.',
  persuasiveness: 'How compelling and convincing your arguments are.',
  data_driven: 'Quality and quantity of supporting data and metrics.',
};

const getColorForScore = (score) => {
  if (score >= 8) return '#34d399';
  if (score >= 5) return '#fbbf24';
  return '#f87171';
};

const getListHoverColor = (isStrength) => isStrength ? '#34d399' : '#f87171';

// --- Styled Components - PROPER ORDER ---

// 1. TabGrid must be defined FIRST since it's used in the main component
const TabGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 1.5rem;
  width: 100%;
  animation: ${fadeIn} 0.6s ease-out;
`;

// 2. Base Card component
const Card = styled.div`
  background: linear-gradient(145deg, #1e293b, #1a2436);
  border: 1px solid #334155;
  border-radius: 16px;
  padding: 1.75rem;
  box-shadow:
    0 4px 20px rgba(0, 0, 0, 0.25),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: default;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
  }

  &:hover {
    transform: translateY(-6px);
    border-color: #475569;
    box-shadow:
      0 12px 30px rgba(0, 0, 0, 0.35),
      inset 0 1px 0 rgba(255, 255, 255, 0.1);
  }
`;

// 3. Card variants that extend the base Card
const VerdictCard = styled(Card)`
  grid-column: 1 / -1;
  background: linear-gradient(145deg, #1e293b, #17202f);
  border-left: 4px solid #14f1d9;
  position: relative;

  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(20, 241, 217, 0.03) 0%, transparent 50%);
    pointer-events: none;
  }

  &:hover {
    border-color: #14f1d9;
    box-shadow:
      0 12px 30px rgba(20, 241, 217, 0.15),
      inset 0 1px 0 rgba(255, 255, 255, 0.1);
    animation: ${pulse} 2s ease-in-out infinite;
  }

  h3 {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 0;
    color: #fff;
    font-size: 1.7rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  h3 span {
    font-size: 2.2rem;
    background: none;
    -webkit-text-fill-color: initial;
  }

  p {
    font-size: 1.2rem;
    line-height: 1.7;
    color: #cbd5e1;
    margin: 0.5rem 0 0 0;
    font-style: italic;
    position: relative;
    padding-left: 1rem;

    &::before {
      content: '"';
      position: absolute;
      left: 0;
      top: -0.5rem;
      font-size: 2rem;
      color: #14f1d9;
      font-family: serif;
    }
  }
`;

const MetricCard = styled(Card)`
  grid-column: span 3;
  display: flex;
  flex-direction: column;
  cursor: pointer;
  animation: ${slideIn} 0.6s ease-out backwards;
  animation-delay: ${props => props.$delay || '0s'};

  &:hover {
    border-color: #14f1d9;
    box-shadow:
      0 8px 25px rgba(20, 241, 217, 0.2),
      inset 0 1px 0 rgba(255, 255, 255, 0.1);

    .score {
      background: linear-gradient(135deg, #14f1d9, #0dcfb6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .metric-icon {
      transform: scale(1.2) rotate(5deg);
    }
  }

  h4 {
    margin-top: 0;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.85rem;
    letter-spacing: 0.75px;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    border-bottom: 1px solid #334155;
    padding-bottom: 1rem;
    margin-bottom: 1.25rem;
  }

  .metric-icon {
    font-size: 1.2rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .score {
    font-size: 4rem;
    font-weight: 900;
    color: #fff;
    line-height: 1;
    margin-top: auto;
    transition: all 0.3s ease;
  }

  .score span {
    font-size: 1.8rem;
    font-weight: 500;
    color: #64748b;
  }
`;

const MetricScoreBar = styled.div`
  width: 100%;
  height: 6px;
  background: linear-gradient(90deg, #334155, #475569);
  border-radius: 3px;
  margin-top: 1.25rem;
  overflow: hidden;
  position: relative;

  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  ${MetricCard}:hover &::after {
    opacity: 1;
  }
`;

const ScoreBarFill = styled.div`
  height: 100%;
  border-radius: 3px;
  background: ${props => props.$color};
  width: ${props => props.$width}%;
  transition: all 1s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;

  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    animation: shimmer 2s infinite;
  }

  @keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
  }
`;

const DimensionalScoresContainer = styled(Card)`
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 3rem;
  align-items: center;
  padding: 2.5rem;
  background: linear-gradient(145deg, #1e293b, #1a2436);

  &:hover {
    transform: translateY(-4px);
    border-color: #475569;
  }

  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
    gap: 2rem;
  }
`;

const ChartWrapper = styled.div`
  position: relative;

  &::before {
    content: '';
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    background: radial-gradient(circle at center, rgba(20, 241, 217, 0.1) 0%, transparent 70%);
    border-radius: 20px;
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  &:hover::before {
    opacity: 1;
  }
`;

const ScoreBreakdown = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.75rem;
  padding-right: 1rem;
`;

const BreakdownItem = styled.div`
  padding: 1.25rem;
  background: rgba(30, 41, 59, 0.7);
  border-radius: 12px;
  border: 1px solid #334155;
  transition: all 0.3s ease;
  cursor: pointer;
  animation: ${fadeIn} 0.6s ease-out backwards;
  animation-delay: ${props => props.$delay || '0s'};

  &:hover {
    transform: translateX(8px);
    border-color: #14f1d9;
    background: rgba(30, 41, 59, 0.9);
    box-shadow: 0 4px 15px rgba(20, 241, 217, 0.15);

    .title {
      color: #14f1d9;
    }

    .progress-fg {
      background: linear-gradient(90deg, #14f1d9, #0dcfb6);
      box-shadow: 0 0 12px rgba(20, 241, 217, 0.4);
    }
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .title {
    font-weight: 700;
    color: #fff;
    font-size: 1.15rem;
    transition: color 0.3s ease;
  }

  .score {
    font-weight: 800;
    background: linear-gradient(135deg, #14f1d9, #0dcfb6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.2rem;
  }

  .description {
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
  }

  .progress-bg {
    width: 100%;
    height: 10px;
    background: linear-gradient(90deg, #334155, #475569);
    border-radius: 5px;
    margin-top: 1rem;
    overflow: hidden;
    position: relative;
  }

  .progress-fg {
    height: 100%;
    background: linear-gradient(90deg, #14f1d9, #0dcfb6);
    border-radius: 5px;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
  }
`;

const ListCard = styled(Card)`
  grid-column: span 6;
  animation: ${slideIn} 0.6s ease-out backwards;
  animation-delay: ${props => props.$delay || '0s'};

  h4 {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 0;
    font-size: 1.6rem;
    font-weight: 800;
    border-bottom: 1px solid #334155;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, ${props => props.$gradientStart}, ${props => props.$gradientEnd});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
 }

  ul {
    list-style: none;
    padding-left: 0;
    margin: 0;
  }
`;

const ListItem = styled.li`
  margin-bottom: 1.25rem;
  line-height: 1.6;
  padding: 1rem 1rem 1rem 2.5rem;
  position: relative;
  color: #cbd5e1;
  background: rgba(30, 41, 59, 0.5);
  border-radius: 8px;
  border: 1px solid transparent;
  transition: all 0.3s ease;
  cursor: pointer;

  &::before {
    content: '${props => props.$bullet}';
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 1.2rem;
    font-weight: 900;
    transition: all 0.3s ease;
  }

  &:hover {
    transform: translateX(8px);
    background: rgba(30, 41, 59, 0.8);
    border-color: ${props => props.$hoverColor};
    color: #fff;
    box-shadow: 0 4px 12px ${props => props.$hoverShadow};

    &::before {
      transform: translateY(-50%) scale(1.3);
    }
  }
`;
// --- OPTION 3: Animated Underline ---
const drawIn = keyframes`
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
`;

const SectionTitle = styled.h3`
  color: #14f1d9; /* Clean, bright text */
  font-size: 1.7rem;
  font-weight: 800;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;

  &::after {
    content: '';
    flex: 1;
    height: 2px;
    background: #14f1d9; /* Solid accent color */
    margin-left: 1rem;
    transform-origin: left; /* Animation starts from the left */
    animation: ${drawIn} 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards;
  }
`;
// --- The Component ---
function OverviewTab({ data }) {
  const [animatedScores, setAnimatedScores] = useState({});

  useEffect(() => {
    // Initialize all scores to 0 first, then animate to actual values
    const initialScores = {
      investment: 0,
      moat: 0,
      gtm: 0,
      team: 0,
    };

    setAnimatedScores(initialScores);

    const timer = setTimeout(() => {
      setAnimatedScores({
        investment: Math.round(data.investment_readiness_score),
        moat: data.competitive_moat.score,
        gtm: data.gtm_strategy.score,
        team: data.team_market_fit.score,
      });
    }, 300);

    return () => clearTimeout(timer);
  }, [data]);

  const chartData = Object.entries(data.dimensional_scores).map(([key, value]) => ({
    dimension: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    score: value,
  }));

  const mainMetrics = [
    {
      title: 'Investment Readiness',
      key: 'investment',
      score: Math.round(data.investment_readiness_score),
      delay: '0.1s'
    },
    {
      title: 'Competitive Moat',
      key: 'moat',
      score: data.competitive_moat.score,
      delay: '0.2s'
    },
    {
      title: 'GTM Strategy',
      key: 'gtm',
      score: data.gtm_strategy.score,
      delay: '0.3s'
    },
    {
      title: 'Team-Market Fit',
      key: 'team',
      score: data.team_market_fit.score,
      delay: '0.4s'
    },
  ];

  return (
    <TabGrid>
      <VerdictCard>
        <h3><span>🎯</span> VC Verdict</h3>
        <p>{data.vc_verdict}</p>
      </VerdictCard>

      {/* Metric Cards with Score Bars */}
      {mainMetrics.map((metric, index) => (
        <MetricCard key={metric.title} $delay={metric.delay}>
          <h4>
            <span className="metric-icon">{getMetricIcon(metric.title)}</span>
            {metric.title}
          </h4>
          <div className="score">
            {animatedScores[metric.key] || 0}
            <span> /10</span>
          </div>

          <MetricScoreBar>
            <ScoreBarFill
              $width={(animatedScores[metric.key] || 0) * 10}
              $color={getColorForScore(metric.score)}
            />
          </MetricScoreBar>
        </MetricCard>
      ))}

      {/* Dimensional Scores with Chart and Breakdown */}
      <DimensionalScoresContainer>
        <ChartWrapper>
          {/* --- EMOJI CONFIRMED HERE --- */}
          <SectionTitle>📊 Dimensional Score Radar</SectionTitle>
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
              <PolarGrid stroke="#334155" strokeWidth={1} />
              <PolarAngleAxis
                dataKey="dimension"
                tick={{ fill: '#94a3b8', fontSize: 13, fontWeight: 600 }}
              />
              <PolarRadiusAxis angle={30} domain={[0, 10]} tick={false} axisLine={false} />

              <defs>
                <linearGradient id="scoreColor" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#14f1d9" stopOpacity={0.8}/>
                  <stop offset="50%" stopColor="#14f1d9" stopOpacity={0.4}/>
                  <stop offset="100%" stopColor="#14f1d9" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <Radar
                name="Score"
                dataKey="score"
                stroke="#14f1d9"
                fill="url(#scoreColor)"
                fillOpacity={0.8}
                strokeWidth={3}
              />
            </RadarChart>
          </ResponsiveContainer>
        </ChartWrapper>

        <ScoreBreakdown>
          {/* --- EMOJI CONFIRMED HERE --- */}
          <SectionTitle>🔍 Dimensional Breakdown</SectionTitle>
          {Object.entries(data.dimensional_scores).map(([key, value], index) => (
            <BreakdownItem key={key} $delay={`${0.1 + index * 0.1}s`}>
              <div className="header">
                <span className="title">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                <span className="score">{value}<span style={{ color: '#64748b', fontSize: '1rem' }}>/10</span></span>
              </div>
              <p className="description">{dimensionDescriptions[key] || ''}</p>
              <div className="progress-bg">
                <div className="progress-fg" style={{ width: `${value * 10}%` }}></div>
              </div>
            </BreakdownItem>
          ))}
        </ScoreBreakdown>
      </DimensionalScoresContainer>

      <ListCard
        $gradientStart="#34d399"
        $gradientEnd="#10b981"
        $delay="0.5s"
      >
        {/* --- EMOJI CONFIRMED HERE --- */}
        <h4>✅ Key Strengths</h4>
        <ul>
          {data.key_strengths.map((strength, index) => (
            <ListItem
              key={index}
              $bullet="★"
              $hoverColor="#34d399"
              $hoverShadow="rgba(52, 211, 153, 0.2)"
            >
              {strength}
            </ListItem>
          ))}
        </ul>
      </ListCard>

      <ListCard
        $gradientStart="#f87171"
        $gradientEnd="#ef4444"
        $delay="0.6s"
      >
        {/* --- EMOJI CONFIRMED HERE --- */}
        <h4>🚩 Priority Improvements</h4>
        <ul>
          {data.key_improvement_suggestions.map((improvement, index) => (
            <ListItem
              key={index}
              $bullet="●"
              $hoverColor="#f87171"
              $hoverShadow="rgba(248, 113, 113, 0.2)"
            >
              {improvement}
            </ListItem>
          ))}
        </ul>
      </ListCard>
    </TabGrid>
  );
}

export default OverviewTab;