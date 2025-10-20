// // // src/components/analysis/TechnicalTab.jsx
// // // FINAL VERSION WITH NER INSIGHTS

// // import React from 'react';
// // import styled from 'styled-components';

// // // --- Styled Components ---

// // const TabGrid = styled.div`
// //   display: grid;
// //   grid-template-columns: repeat(2, 1fr);
// //   gap: 1.5rem;
// //   width: 100%;
// // `;

// // const Card = styled.div`
// //   background-color: #1E293B;
// //   border: 1px solid #334155;
// //   border-radius: 12px;
// //   padding: 1.5rem;
// //   box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  
// //   h3 {
// //     margin-top: 0;
// //     font-size: 1.2rem;
// //     color: #fff;
// //     border-bottom: 1px solid #334155;
// //     padding-bottom: 1rem;
// //     margin-bottom: 1.5rem;
// //   }
// //       p.caption { /* The new "why this matters" text */
// //     margin: 0.75rem 0 0 0;
// //     font-size: 0.85rem;
// //     color: #64748b; /* Muted color */
// //     line-height: 1.5;
// //     border-left: 3px solid #334155;
// //     padding-left: 1rem;
// //   }
// // `;

// // const ChecklistCard = styled(Card)`
// //   grid-column: 1 / -1;
// // `;

// // const ChecklistGrid = styled.div`
// //   display: grid;
// //   grid-template-columns: repeat(2, 1fr);
// //   gap: 1rem;
// // `;

// // const ChecklistItem = styled.div`
// //   display: flex;
// //   align-items: center;
// //   gap: 0.75rem;
// //   padding: 0.75rem;
// //   border-radius: 8px;
// //   background-color: ${({ found }) => (found ? 'rgba(52, 211, 153, 0.1)' : 'rgba(248, 113, 113, 0.1)')};
// //   color: ${({ found }) => (found ? '#34d399' : '#f87171')};
  
// //   .icon { font-size: 1.2rem; }
// //   .text { font-weight: 500; }
// // `;

// // const NlpGrid = styled.div`
// //   display: flex;
// //   flex-direction: column;
// //   gap: 1.5rem;
// // `;

// // const NlpMetricBox = styled.div`
// //   h4 { margin-top: 0; margin-bottom: 0.75rem; font-size: 1rem; font-weight: 600; color: #cbd5e1; }
// //   p { margin: 0; font-size: 2rem; font-weight: 700; color: #fff; }
// //   .subtext { font-size: 0.9rem; color: #94a3b8; margin-top: 0.25rem; }
// // `;

// // // --- NEW STYLED COMPONENTS FOR NER INSIGHTS ---
// // const InsightsContainer = styled.div`
// //   display: flex;
// //   flex-direction: column;
// //   gap: 1.5rem;
// // `;

// // const InsightItem = styled.div`
// //   display: flex;
// //   gap: 1rem;
// //   align-items: flex-start;
// // `;

// // const InsightIcon = styled.div`
// //   font-size: 1.5rem;
// //   margin-top: 0.25rem;
// // `;

// // const InsightContent = styled.div`
// //   h4 {
// //     margin: 0 0 0.5rem 0;
// //     font-size: 1rem;
// //     font-weight: 600;
// //     color: #cbd5e1;
// //   }
// //   p {
// //     margin: 0;
// //     font-size: 0.95rem;
// //     color: ${({ status }) => {
// //       if (status === 'good') return '#34d399';
// //       if (status === 'warning') return '#facc15';
// //       if (status === 'bad') return '#f87171';
// //       return '#fff';
// //     }};
// //   }
// // `;


// // // --- Helper Function to Determine Status ---
// // const getInsightStatus = (text) => {
// //   if (!text) return 'neutral';
// //   if (text.includes('❌') || text.includes('No')) return 'bad';
// //   if (text.includes('⚠️')) return 'warning';
// //   return 'good';
// // };


// // // --- The Component ---
// // function TechnicalTab({ data }) {
// //   const { checklist, prob_sol_alignment, redundant_slides, buzzword_count, ner_insights } = data;

// //   const checklistItems = Object.entries(checklist);
// //   const completionPercentage = (checklistItems.filter(([, found]) => found).length / checklistItems.length) * 100;

// //   return (
// //     <TabGrid>
// //       <ChecklistCard>
// //         <h3>✅ Pitch Deck Completeness Checklist</h3>
// //         <p style={{ marginTop: 0, color: '#e2e8f0' }}>Completion Score: <span style={{fontWeight: 'bold', fontSize: '1.2rem'}}>{completionPercentage.toFixed(0)}%</span></p>
// //         <div style={{ width: '100%', backgroundColor: '#334155', borderRadius: '4px', height: '8px', marginBottom: '1.5rem' }}>
// //           <div style={{ width: `${completionPercentage}%`, backgroundColor: '#14f1d9', borderRadius: '4px', height: '100%' }} />
// //         </div>
// //         <ChecklistGrid>
// //           {checklistItems.map(([item, found]) => (
// //             <ChecklistItem key={item} found={found}>
// //               <span className="icon">{found ? '✔' : '❌'}</span>
// //               <span className="text">{item}</span>
// //             </ChecklistItem>
// //           ))}
// //         </ChecklistGrid>
// //       </ChecklistCard>

// //       <Card>
// //         <h3>🧠 Advanced NLP Analysis</h3>
// //         <NlpGrid>
// //           <NlpMetricBox>
// //             <h4>🔗 Slide Cohesion</h4>
// //             <p>{(prob_sol_alignment * 100).toFixed(0)}%</p>
// //             <div className="subtext">Problem-Solution Alignment</div>
// //           </NlpMetricBox>
// //           <hr style={{ border: 'none', borderTop: '1px solid #334155' }} />
// //           <NlpMetricBox>
// //             <h4>🔄 Content Redundancy</h4>
// //             <p style={{ color: redundant_slides.length > 0 ? '#facc15' : '#fff' }}>
// //               {redundant_slides.length}
// //             </p>
// //             <div className="subtext">{redundant_slides.length > 0 ? 'Redundant slide pairs found' : 'No significant redundancy'}</div>
// //           </NlpMetricBox>
// //           <hr style={{ border: 'none', borderTop: '1px solid #334155' }} />
// //           <NlpMetricBox>
// //             <h4>🚩 Buzzword Analysis</h4>
// //             <p>{buzzword_count}</p>
// //             <div className="subtext">Buzzwords detected</div>
// //           </NlpMetricBox>
// //         </NlpGrid>
// //       </Card>
      
// //       {/* --- REPLACED PLACEHOLDER WITH REAL NER UI --- */}
// //       <Card>
// //         <h3>🏷️ Business Intelligence (NER)</h3>
// //         <InsightsContainer>
// //           <InsightItem>
// //             <InsightIcon>👥</InsightIcon>
// //             <InsightContent status={getInsightStatus(ner_insights.team_summary)}>
// //               <h4>Team & Leadership</h4>
// //               <p>{ner_insights.team_summary || 'No data available.'}</p>
// //             </InsightContent>
// //           </InsightItem>
          
// //           <InsightItem>
// //             <InsightIcon>🏢</InsightIcon>
// //             <InsightContent status={getInsightStatus(ner_insights.competition_summary)}>
// //               <h4>Competitive Landscape</h4>
// //               <p>{ner_insights.competition_summary || 'No data available.'}</p>
// //             </InsightContent>
// //           </InsightItem>

// //           <InsightItem>
// //             <InsightIcon>💰</InsightIcon>
// //             <InsightContent status={getInsightStatus(ner_insights.financial_summary)}>
// //               <h4>Financial Context</h4>
// //               <p>{ner_insights.financial_summary || 'No data available.'}</p>
// //             </InsightContent>
// //           </InsightItem>

// //           <InsightItem>
// //             <InsightIcon>🌍</InsightIcon>
// //             <InsightContent status={getInsightStatus(ner_insights.market_geography)}>
// //               <h4>Market Geography</h4>
// //               <p>{ner_insights.market_geography || 'No data available.'}</p>
// //             </InsightContent>
// //           </InsightItem>
// //         </InsightsContainer>
// //       </Card>
// //     </TabGrid>
// //   );
// // }

// // export default TechnicalTab;
// // src/components/analysis/TechnicalTab.jsx
// // ENHANCED WITH DYNAMIC ANIMATIONS & MICRO-INTERACTIONS

// import React, { useState, useEffect } from 'react';
// import styled, { keyframes } from 'styled-components';
// import { useInView } from 'react-intersection-observer';

// // --- Keyframes for Animations ---

// const fillAnimation = keyframes`
//   from { width: 0%; }
// `;

// const fadeIn = keyframes`
//   from {
//     opacity: 0;
//     transform: translateY(10px);
//   }
//   to {
//     opacity: 1;
//     transform: translateY(0);
//   }
// `;

// const shineEffect = keyframes`
//   0% { transform: translateX(-100%) skewX(-25deg); }
//   100% { transform: translateX(200%) skewX(-25deg); }
// `;

// const expandDivider = keyframes`
//   from {
//     transform: scaleX(0);
//   }
//   to {
//     transform: scaleX(1);
//   }
// `;


// // --- Styled Components ---

// const TabGrid = styled.div`
//   display: grid;
//   grid-template-columns: repeat(2, 1fr);
//   gap: 2rem;
//   width: 100%;
// `;

// const Card = styled.div`
//   background: linear-gradient(145deg, #2A3A50, #1E293B);
//   border: 1px solid #334155;
//   border-radius: 16px;
//   padding: 2rem;
//   box-shadow: 0 8px 32px rgba(0,0,0,0.3);
//   transition: transform 0.3s ease, box-shadow 0.3s ease;
//   position: relative; // Needed for shine effect
//   overflow: hidden;   // Needed for shine effect

//   &::before {
//     content: '';
//     position: absolute;
//     top: 0;
//     left: 0;
//     width: 50%;
//     height: 100%;
//     background: linear-gradient(to right, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 0.08) 50%, rgba(255, 255, 255, 0) 100%);
//     transform: translateX(-100%) skewX(-25deg);
//     pointer-events: none;
//     z-index: 1;
//   }

//   &:hover {
//     transform: translateY(-5px);
//     box-shadow: 0 12px 40px rgba(0,0,0,0.4);

//     &::before {
//       animation: ${shineEffect} 1s ease-in-out;
//     }
//   }

//   h3 {
//     margin-top: 0;
//     font-size: 1.3rem;
//     font-weight: 600;
//     color: #F1F5F9;
//     border-bottom: 1px solid #475569;
//     padding-bottom: 1rem;
//     margin-bottom: 1.5rem;
//     display: flex;
//     align-items: center;
//     gap: 0.75rem;
//   }
// `;

// const ChecklistCard = styled(Card)`
//   grid-column: 1 / -1;
// `;

// const ProgressBarContainer = styled.div`
//   width: 100%;
//   background-color: #334155;
//   border-radius: 8px;
//   height: 12px;
//   margin-bottom: 1.5rem;
//   overflow: hidden;
// `;

// const ProgressBar = styled.div`
//   width: ${({ percentage }) => percentage}%;
//   background: linear-gradient(90deg, #14F1D9, #31C5E1);
//   border-radius: 8px;
//   height: 100%;
//   animation: ${fillAnimation} 1s cubic-bezier(0.25, 0.1, 0.25, 1) forwards;
// `;

// const ChecklistGrid = styled.div`
//   display: grid;
//   grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
//   gap: 1rem;
// `;

// const ChecklistItem = styled.div`
//   display: flex;
//   align-items: center;
//   gap: 0.75rem;
//   padding: 0.75rem 1rem;
//   border-radius: 8px;
//   background-color: ${({ found }) => (found ? 'rgba(20, 241, 217, 0.08)' : 'rgba(248, 113, 113, 0.08)')};
//   border: 1px solid ${({ found }) => (found ? 'rgba(20, 241, 217, 0.2)' : 'rgba(248, 113, 113, 0.2)')};
//   color: ${({ found }) => (found ? '#14F1D9' : '#f87171')};
  
//   // Staggered animation
//   opacity: 0;
//   animation: ${fadeIn} 0.5s ease forwards;
//   animation-delay: ${({ delay }) => delay}s;

//   .icon { font-size: 1.2rem; line-height: 1; }
//   .text { font-weight: 500; color: #E2E8F0; }
// `;

// const NlpGrid = styled.div`
//   display: flex;
//   flex-direction: column;
//   gap: 1.5rem;
// `;

// const NlpMetricBox = styled.div`
//   opacity: 0;
//   animation: ${fadeIn} 0.6s ease-out forwards;
//   animation-delay: ${({ delay }) => delay}s;

//   h4 { margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 500; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; }
//   p { margin: 0; font-size: 2.5rem; font-weight: 700; color: #F1F5F9; line-height: 1; }
//   .subtext { font-size: 0.9rem; color: #64748B; margin-top: 0.5rem; }
// `;

// const AnimatedDivider = styled.hr`
//   border: none;
//   border-top: 1px solid #334155;
//   transform-origin: center;
//   animation: ${expandDivider} 0.8s cubic-bezier(0.25, 0.1, 0.25, 1) forwards;
//   animation-delay: ${({ delay }) => delay}s;
// `;

// const InsightsContainer = styled.div`
//   display: flex;
//   flex-direction: column;
//   gap: 2rem;
// `;

// const InsightItem = styled.div`
//   display: flex;
//   gap: 1.5rem;
//   align-items: flex-start;
//   opacity: 0;
//   animation: ${fadeIn} 0.6s ease forwards;
//   animation-delay: ${({ delay }) => delay}s;
// `;

// const InsightIcon = styled.div`
//   font-size: 1.75rem;
//   margin-top: 0.25rem;
//   color: #64748B;
//   flex-shrink: 0;
//   width: 32px;
//   text-align: center;
// `;

// const InsightContent = styled.div`
//   h4 { margin: 0 0 0.5rem 0; font-size: 1.1rem; font-weight: 600; color: #CBD5E1; }
//   p {
//     margin: 0;
//     font-size: 1rem;
//     line-height: 1.6;
//     color: ${({ status }) => {
//       if (status === 'good') return '#34D399';
//       if (status === 'warning') return '#FACC15';
//       if (status === 'bad') return '#F87171';
//       return '#E2E8F0';
//     }};
//   }
// `;

// // --- Helper Components & Hooks ---
// const AnimatedCounter = ({ endValue, duration = 1500, style }) => {
//   const [count, setCount] = useState(0);
//   const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.5 });

//   useEffect(() => {
//     if (!inView || endValue == null) return; // ✅ avoid undefined
//     const numericPart = parseFloat(endValue) || 0;

//     let start = 0;
//     let startTime = null;

//     const animate = (currentTime) => {
//       if (!startTime) startTime = currentTime;
//       const progress = Math.min((currentTime - startTime) / duration, 1);
//       const currentNum = Math.floor(progress * (numericPart - start) + start);
//       setCount(currentNum);
//       if (progress < 1) requestAnimationFrame(animate);
//     };
//     requestAnimationFrame(animate);
//   }, [inView, endValue, duration]);

//   const safeEndValue = endValue ?? 0;
//   const isPercent = safeEndValue.toString().includes('%');

//   return (
//     <p ref={ref} style={style}>
//       {count}
//       {isPercent ? '%' : ''}
//     </p>
//   );
// };


// const getInsightStatus = (text) => {
//   if (!text) return 'neutral';
//   const lowerText = text.toLowerCase();
//   if (lowerText.includes('❌') || lowerText.includes('no mention')) return 'bad';
//   if (lowerText.includes('⚠️') || lowerText.includes('limited')) return 'warning';
//   return 'good';
// };


// // --- The Main Component ---
// function TechnicalTab({ data }) {
//   const { checklist, prob_sol_alignment, redundant_slides, buzzword_count, ner_insights } = data;

//   const checklistItems = Object.entries(checklist);
//   const completionPercentage = (checklistItems.filter(([, found]) => found).length / checklistItems.length) * 100;

//   return (
//     <TabGrid>
//       <ChecklistCard>
//         <h3>✅ Pitch Deck Completeness</h3>
//         <p style={{ marginTop: 0, color: '#e2e8f0' }}>
//           Completion Score: <span style={{fontWeight: 'bold', fontSize: '1.4rem', color: '#14F1D9'}}>{completionPercentage.toFixed(0)}%</span>
//         </p>
//         <ProgressBarContainer>
//           <ProgressBar percentage={completionPercentage} />
//         </ProgressBarContainer>
//         <ChecklistGrid>
//           {checklistItems.map(([item, found], index) => (
//             <ChecklistItem key={item} found={found} delay={index * 0.05}>
//               <span className="icon">{found ? '✔' : '❌'}</span>
//               <span className="text">{item}</span>
//             </ChecklistItem>
//           ))}
//         </ChecklistGrid>
//       </ChecklistCard>

//       <Card>
//         <h3>🧠 NLP Analysis</h3>
//         <NlpGrid>
//           <NlpMetricBox delay={0.1}>
//             <h4>Slide Cohesion</h4>
//             <AnimatedCounter endValue={`${(prob_sol_alignment * 100).toFixed(0)}%`} />
//             <div className="subtext">Problem-Solution Alignment</div>
//           </NlpMetricBox>
//           <AnimatedDivider delay={0.2} />
//           <NlpMetricBox delay={0.3}>
//             <h4>Content Redundancy</h4>
//             <AnimatedCounter
//                 endValue={redundant_slides.length}
//                 style={{ color: redundant_slides.length > 0 ? '#FACC15' : '#F1F5F9' }}
//             />
//             <div className="subtext">{redundant_slides.length > 0 ? 'Redundant slide pairs' : 'No significant redundancy'}</div>
//           </NlpMetricBox>
//           <AnimatedDivider delay={0.4} />
//           <NlpMetricBox delay={0.5}>
//             <h4>Buzzword Analysis</h4>
//             <AnimatedCounter endValue={buzzword_count} />
//             <div className="subtext">Buzzwords detected</div>
//           </NlpMetricBox>
//         </NlpGrid>
//       </Card>
      
//       <Card>
//         <h3>🏷️ Business Intelligence</h3>
//         <InsightsContainer>
//           {Object.entries(ner_insights).map(([key, value], index) => {
//             const details = {
//               team_summary: { icon: '👥', title: 'Team & Leadership' },
//               competition_summary: { icon: '🏢', title: 'Competitive Landscape' },
//               financial_summary: { icon: '💰', title: 'Financial Context' },
//               market_geography: { icon: '🌍', title: 'Market Geography' },
//             }[key];
            
//             return (
//               <InsightItem key={key} delay={index * 0.15}>
//                 <InsightIcon>{details.icon}</InsightIcon>
//                 <InsightContent status={getInsightStatus(value)}>
//                   <h4>{details.title}</h4>
//                   <p>{value || 'No data available.'}</p>
//                 </InsightContent>
//               </InsightItem>
//             );
//           })}
//         </InsightsContainer>
//       </Card>
//     </TabGrid>
//   );
// }

// // You'll need to install react-intersection-observer
// // npm install react-intersection-observer
// // or
// // yarn add react-intersection-observer

// export default TechnicalTab;


import React, { useState, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';
import { useInView } from 'react-intersection-observer';

// --- Keyframes for Animations ---

const fillAnimation = keyframes`
  from { width: 0%; }
`;

const fadeIn = keyframes`
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

// --- Styled Components ---

const TabGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.5rem;
  width: 100%;
`;

const Card = styled.div`
  background: rgba(30, 41, 59, 0.6);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 20px;
  padding: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  animation: ${fadeIn} 0.6s ease-out;
  overflow: hidden;
  position: relative;
  
  animation-delay: ${({ delay }) => delay || '0s'};

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.1), transparent);
    transition: left 0.5s;
  }

  &:hover {
    transform: translateY(-5px);
    border-color: rgba(20, 241, 217, 0.4);
    box-shadow: 0 12px 40px rgba(20, 241, 217, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    
    &::before {
      left: 100%;
    }
  }

  h3 {
    margin-top: 0;
    font-size: 1.3rem;
    font-weight: 700;
    color: #F1F5F9; 
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
`;

const ChecklistCard = styled(Card)`
  grid-column: 1 / -1;

  h3 {
    background: linear-gradient(135deg, #14f1d9 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
`;

// --- PROGRESS BAR SECTION: Restored to original code ---
const ProgressBarContainer = styled.div`
  width: 100%;
  background-color: #334155; /* Original background */
  border-radius: 8px;          /* Original radius */
  height: 12px;
  margin-bottom: 1.5rem;
  overflow: hidden;
  /* No box-shadow */
`;

const ProgressBar = styled.div`
  width: ${({ percentage }) => percentage}%;
  height: 100%;
  background: linear-gradient(90deg, #14F1D9, #31C5E1); /* Original gradient */
  border-radius: 8px;
  animation: ${fillAnimation} 1s cubic-bezier(0.25, 0.1, 0.25, 1) forwards; /* Original animation */
`;
// --- END OF PROGRESS BAR SECTION ---

const ChecklistGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
`;

const ChecklistItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  border-radius: 12px;
  transition: all 0.3s ease;
  cursor: default;
  
  background: ${({ found }) => (found
    ? 'linear-gradient(135deg, rgba(52, 211, 153, 0.15) 0%, rgba(16, 185, 129, 0.1) 100%)'
    : 'linear-gradient(135deg, rgba(248, 113, 113, 0.15) 0%, rgba(239, 68, 68, 0.1) 100%)'
  )};
  border: 1px solid ${({ found }) => (found ? 'rgba(52, 211, 153, 0.3)' : 'rgba(248, 113, 113, 0.3)')};
  color: ${({ found }) => (found ? '#34d399' : '#f87171')};

  &:hover {
    transform: translateX(5px);
    border-color: ${({ found }) => (found ? 'rgba(52, 211, 153, 0.6)' : 'rgba(248, 113, 113, 0.6)')};
    box-shadow: ${({ found }) => (found
      ? '0 4px 12px rgba(52, 211, 153, 0.2)'
      : '0 4px 12px rgba(248, 113, 113, 0.2)'
    )};
  }

  .icon { font-size: 1.3rem; }
  .text { font-weight: 600; color: #E2E8F0; font-size: 0.95rem; }
`;

const NlpGrid = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const NlpMetricBox = styled.div`
  padding: 1rem;
  border-radius: 12px;
  background: rgba(51, 65, 85, 0.3);
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(51, 65, 85, 0.5);
    transform: scale(1.02);
  }

  h4 {
    margin: 0 0 0.75rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: #cbd5e1;
  }
  p {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #14f1d9 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .subtext {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 0.25rem;
  }
`;

const StyledDivider = styled.hr`
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
  margin: 0;
`;

const InsightsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const InsightItem = styled.div`
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  padding: 1rem;
  border-radius: 12px;
  background: rgba(51, 65, 85, 0.3);
  border: 1px solid rgba(148, 163, 184, 0.1);
  transition: all 0.3s ease;

  &:hover {
    background: rgba(51, 65, 85, 0.5);
    border-color: rgba(20, 241, 217, 0.3);
    transform: translateX(5px);
    box-shadow: 0 4px 20px rgba(20, 241, 217, 0.1);
  }
`;

const InsightIcon = styled.div`
  font-size: 1.8rem;
  margin-top: 0.25rem;
  filter: drop-shadow(0 0 8px rgba(20, 241, 217, 0.3));
`;

const InsightContent = styled.div`
  h4 {
    margin: 0 0 0.5rem 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: #cbd5e1;
  }
  p {
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.6;
    font-weight: 500;
    color: ${({ status }) => {
      if (status === 'good') return '#34D399';
      if (status === 'warning') return '#FACC15';
      if (status === 'bad') return '#F87171';
      return '#E2E8F0';
    }};
  }
`;

// --- Helper Components & Hooks ---
const AnimatedCounter = ({ endValue, duration = 1500, style, className }) => {
    const [count, setCount] = useState(0);
    const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.5 });
  
    useEffect(() => {
      if (!inView) return;
  
      const numericPart = parseFloat(endValue) || 0;
      let start = 0;
      let startTime = null;
  
      const animate = (currentTime) => {
        if (!startTime) startTime = currentTime;
        const progress = Math.min((currentTime - startTime) / duration, 1);
        const currentNum = Math.floor(progress * (numericPart - start) + start);
        setCount(currentNum);
        if (progress < 1) requestAnimationFrame(animate);
      };
      requestAnimationFrame(animate);
    }, [inView, endValue, duration]);
  
    const isPercent = typeof endValue === 'string' && endValue.includes('%');
  
    return (
      <p ref={ref} style={style} className={className}>
        {count}
        {isPercent ? '%' : ''}
      </p>
    );
};

const getInsightStatus = (text) => {
  if (!text) return 'neutral';
  const lowerText = text.toLowerCase();
  if (lowerText.includes('❌') || lowerText.includes('no mention')) return 'bad';
  if (lowerText.includes('⚠️') || lowerText.includes('limited')) return 'warning';
  return 'good';
};


// --- The Main Component ---
function TechnicalTab({ data }) {
  const { checklist, prob_sol_alignment, redundant_slides, buzzword_count, ner_insights } = data;

  const checklistItems = Object.entries(checklist);
  const completionPercentage = (checklistItems.filter(([, found]) => found).length / checklistItems.length) * 100;

  return (
    <TabGrid>
      <ChecklistCard delay="0.1s">
        <h3>✅ Pitch Deck Completeness</h3>
        <p style={{ marginTop: 0, color: '#e2e8f0', fontSize: '1rem' }}>
          Completion Score: 
          <span style={{
            fontWeight: 'bold',
            fontSize: '1.4rem',
            marginLeft: '8px',
            background: 'linear-gradient(135deg, #14f1d9 0%, #a78bfa 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            {completionPercentage.toFixed(0)}%
          </span>
        </p>
        <ProgressBarContainer>
          <ProgressBar percentage={completionPercentage} />
        </ProgressBarContainer>
        <ChecklistGrid>
          {checklistItems.map(([item, found]) => (
            <ChecklistItem key={item} found={found}>
              <span className="icon">{found ? '✔' : '❌'}</span>
              <span className="text">{item}</span>
            </ChecklistItem>
          ))}
        </ChecklistGrid>
      </ChecklistCard>

      <Card delay="0.2s">
        <h3>🧠 Advanced NLP Analysis</h3>
        <NlpGrid>
          <NlpMetricBox>
            <h4>🔗 Slide Cohesion</h4>
            <AnimatedCounter endValue={`${(prob_sol_alignment * 100).toFixed(0)}%`} />
            <div className="subtext">Problem-Solution Alignment</div>
          </NlpMetricBox>
          <StyledDivider />
          <NlpMetricBox>
            <h4>🔄 Content Redundancy</h4>
            <AnimatedCounter
                endValue={redundant_slides.length}
                className={redundant_slides.length > 0 ? 'warning-text' : ''}
                style={{ color: redundant_slides.length > 0 ? '#FACC15' : '' }}
            />
            <div className="subtext">{redundant_slides.length > 0 ? 'Redundant slide pairs found' : 'No significant redundancy'}</div>
          </NlpMetricBox>
          <StyledDivider />
          <NlpMetricBox>
            <h4>🚩 Buzzword Analysis</h4>
            <AnimatedCounter endValue={buzzword_count} />
            <div className="subtext">Buzzwords detected</div>
          </NlpMetricBox>
        </NlpGrid>
      </Card>
      
      <Card delay="0.3s">
        <h3>🏷️ Business Intelligence</h3>
        <InsightsContainer>
          {Object.entries(ner_insights).map(([key, value]) => {
            const details = {
              team_summary: { icon: '👥', title: 'Team & Leadership' },
              competition_summary: { icon: '🏢', title: 'Competitive Landscape' },
              financial_summary: { icon: '💰', title: 'Financial Context' },
              market_geography: { icon: '🌍', title: 'Market Geography' },
            }[key];
            
            return (
              <InsightItem key={key}>
                <InsightIcon>{details.icon}</InsightIcon>
                <InsightContent status={getInsightStatus(value)}>
                  <h4>{details.title}</h4>
                  <p>{value || 'No data available.'}</p>
                </InsightContent>
              </InsightItem>
            );
          })}
        </InsightsContainer>
      </Card>
    </TabGrid>
  );
}

// You'll need to install react-intersection-observer and styled-components
// npm install react-intersection-observer styled-components
// or
// yarn add react-intersection-observer styled-components

export default TechnicalTab;