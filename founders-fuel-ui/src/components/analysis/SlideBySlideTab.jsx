// // src/components/analysis/SlideBySlideTab.jsx

// import React, { useState } from 'react';
// import styled from 'styled-components';

// // --- Styled Components ---

// const AccordionContainer = styled.div`
//   width: 100%;
//   display: flex;
//   flex-direction: column;
//   gap: 1rem;
// `;

// const AccordionItem = styled.div`
//   background-color: #1E293B;
//   border: 1px solid #334155;
//   border-radius: 12px;
//   overflow: hidden;
//   transition: all 0.3s ease-in-out;
// `;

// const AccordionHeader = styled.button`
//   width: 100%;
//   padding: 1.25rem 1.5rem;
//   display: flex;
//   justify-content: space-between;
//   align-items: center;
//   background-color: #1E293B;
//   border: none;
//   cursor: pointer;
//   text-align: left;
//   font-size: 1.1rem;
//   font-weight: 600;
//   color: #e2e8f0;

//   &:hover {
//     background-color: #334155;
//   }
// `;

// const AccordionContent = styled.div`
//   max-height: ${({ isOpen }) => (isOpen ? '1000px' : '0')};
//   overflow: hidden;
//   transition: max-height 0.4s ease-in-out;
//   padding: ${({ isOpen }) => (isOpen ? '0 1.5rem 1.5rem 1.5rem' : '0 1.5rem')};
// `;

// const SentimentBadge = styled.span`
//   padding: 0.3rem 0.8rem;
//   border-radius: 999px;
//   font-size: 0.9rem;
//   font-weight: 700;
//   color: #0F172A;
//   background-color: ${({ sentiment }) => {
//     if (sentiment === 'POSITIVE') return '#34d399'; // Green
//     if (sentiment === 'NEGATIVE') return '#f87171'; // Red
//     return '#64748b'; // Gray for Neutral/Other
//   }};
// `;

// // Inner Tabs for Content/Feedback/Metrics
// const InnerTabContainer = styled.div`
//   display: flex;
//   border-bottom: 1px solid #334155;
//   margin-bottom: 1rem;
// `;

// const InnerTabButton = styled.button`
//   padding: 0.75rem 1.25rem;
//   border: none;
//   background: none;
//   color: ${({ active }) => (active ? '#14f1d9' : '#94a3b8')};
//   font-size: 0.9rem;
//   font-weight: 600;
//   cursor: pointer;
//   border-bottom: 2px solid ${({ active }) => (active ? '#14f1d9' : 'transparent')};
  
//   &:hover { color: #14f1d9; }
// `;

// const InnerTabContent = styled.div`
//   padding: 1rem;
//   background-color: #0F172A;
//   border-radius: 8px;
//   min-height: 200px;
// `;

// const SlideContentText = styled.pre`
//   white-space: pre-wrap; /* Allows text to wrap */
//   word-wrap: break-word;
//   font-family: inherit;
//   font-size: 0.95rem;
//   line-height: 1.6;
//   color: #cbd5e1;
// `;

// const FeedbackSection = styled.div`
//   h5 { 
//     color: #14f1d9; 
//     margin-top: 0;
//     margin-bottom: 0.5rem;
//   }
//   p {
//     margin: 0 0 1rem 0;
//     color: #cbd5e1;
//   }
// `;

// const MetricsGrid = styled.div`
//   display: grid;
//   grid-template-columns: repeat(3, 1fr);
//   gap: 1rem;
//   text-align: center;
// `;

// const MetricBox = styled.div`
//   background-color: #1E293B;
//   padding: 1rem;
//   border-radius: 8px;
//   .label { 
//     font-size: 0.9rem;
//     color: #94a3b8;
//     margin-bottom: 0.5rem;
//   }
//   .value {
//     font-size: 1.5rem;
//     font-weight: 700;
//     color: #fff;
//   }
// `;


// // --- Single Slide Component (to manage its own state) ---
// const SlideItem = ({ slideData, index }) => {
//   const [isOpen, setIsOpen] = useState(false);
//   const [activeInnerTab, setActiveInnerTab] = useState('feedback');

//   const { slide_texts, slide_by_slide_feedback, local_metrics } = slideData;
//   const slideText = slide_texts[index] || "No content extracted for this slide.";
//   const aiFeedback = slide_by_slide_feedback[index] || {};
//   const metrics = local_metrics[index] || {};
//   const sentiment = metrics.sentiment || {};

//   return (
//     <AccordionItem>
//       <AccordionHeader onClick={() => setIsOpen(!isOpen)}>
//         <span>📄 Slide {index + 1}</span>
//         <SentimentBadge sentiment={sentiment.label}>{sentiment.label}</SentimentBadge>
//       </AccordionHeader>
//       <AccordionContent isOpen={isOpen}>
//         <InnerTabContainer>
//           <InnerTabButton active={activeInnerTab === 'feedback'} onClick={() => setActiveInnerTab('feedback')}>AI Feedback</InnerTabButton>
//           <InnerTabButton active={activeInnerTab === 'metrics'} onClick={() => setActiveInnerTab('metrics')}>Metrics</InnerTabButton>
//           <InnerTabButton active={activeInnerTab === 'content'} onClick={() => setActiveInnerTab('content')}>Content</InnerTabButton>
//         </InnerTabContainer>

//         {activeInnerTab === 'content' && (
//           <InnerTabContent>
//             <SlideContentText>{slideText}</SlideContentText>
//           </InnerTabContent>
//         )}
//         {activeInnerTab === 'feedback' && (
//           <InnerTabContent>
//             <FeedbackSection>
//               <h5>Clarity</h5>
//               <p>{aiFeedback.clarity_feedback || 'N/A'}</p>
//               <h5>Tone</h5>
//               <p>{aiFeedback.tone_feedback || 'N/A'}</p>
//               <h5>Intent</h5>
//               <p>{aiFeedback.intent_fulfillment_feedback || 'N/A'}</p>
//             </FeedbackSection>
//           </InnerTabContent>
//         )}
//         {activeInnerTab === 'metrics' && (
//           <InnerTabContent>
//             <MetricsGrid>
//               <MetricBox>
//                 <div className="label">Readability</div>
//                 <div className="value">{metrics.readability_interpretation || 'N/A'}</div>
//               </MetricBox>
//               <MetricBox>
//                 <div className="label">Data Points</div>
//                 <div className="value">{metrics.quantitative_count || 0}</div>
//               </MetricBox>
//               <MetricBox>
//                 <div className="label">Sentiment</div>
//                 <div className="value">{sentiment.label} ({(sentiment.score * 100).toFixed(0)}%)</div>
//               </MetricBox>
//             </MetricsGrid>
//             {/* We can add Detected Entities here later if needed */}
//           </InnerTabContent>
//         )}
//       </AccordionContent>
//     </AccordionItem>
//   );
// };


// // --- Main Component ---
// function SlideBySlideTab({ data }) {
//   // We need to loop based on the number of slides, which we can get from slide_texts
//   const slideCount = data.slide_texts.length;
  
//   return (
//     <AccordionContainer>
//       {[...Array(slideCount)].map((_, index) => (
//         <SlideItem key={index} index={index} slideData={data} />
//       ))}
//     </AccordionContainer>
//   );
// }

// export default SlideBySlideTab;

import React, { useState, useRef, useEffect } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

// Animated Number Counter Component
const AnimatedCounter = ({ endValue, duration = 1500 }) => {
  const [count, setCount] = useState(0);
  const [hasAnimated, setHasAnimated] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated) {
          setHasAnimated(true);
          const isString = typeof endValue === 'string' && isNaN(parseInt(endValue, 10));
          
          if (!isString) {
            let start = 0;
            const end = parseInt(endValue, 10);
            if (start === end) { setCount(end); return; }

            let startTime = null;
            const animate = (currentTime) => {
              if (!startTime) startTime = currentTime;
              const progress = Math.min((currentTime - startTime) / duration, 1);
              const currentNum = Math.floor(progress * (end - start) + start);
              setCount(currentNum);
              if (progress < 1) requestAnimationFrame(animate);
            };
            requestAnimationFrame(animate);
          } else {
            setCount(endValue);
          }
        }
      },
      { threshold: 0.5 }
    );

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [endValue, duration, hasAnimated]);

  return (
    <div ref={ref} className="metric-value">
      {count}{String(endValue).includes('%') ? '%' : ''}
    </div>
  );
};

// Helper function to get readability color class
const getReadabilityColor = (readability) => {
  const lower = String(readability).toLowerCase();
  if (lower.includes('easy') || lower.includes('good')) return 'readability-easy';
  if (lower.includes('moderate') || lower.includes('medium')) return 'readability-moderate';
  if (lower.includes('hard') || lower.includes('difficult')) return 'readability-hard';
  return 'readability-neutral';
};

// Helper function to get sentiment color class
const getSentimentColor = (sentiment) => {
  const lower = String(sentiment).toLowerCase();
  if (lower.includes('positive')) return 'sentiment-positive';
  if (lower.includes('negative')) return 'sentiment-negative';
  return 'sentiment-neutral';
};

// SlideItem Component
const SlideItem = ({ slideData, index }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeInnerTab, setActiveInnerTab] = useState("feedback");
  const contentRef = useRef(null);
  const [contentHeight, setContentHeight] = useState(0);
  
  const tabsContainerRef = useRef(null);
  const [indicatorPosition, setIndicatorPosition] = useState({ left: 0, width: 0 });

  const { slide_texts = [], slide_by_slide_feedback = [], local_metrics = [] } = slideData;
  const slideText = slide_texts[index] || "No content extracted for this slide.";
  const aiFeedback = slide_by_slide_feedback[index] || {};
  const metrics = local_metrics[index] || {};
  const sentiment = metrics.sentiment || {};
  
  const TABS = ["feedback", "metrics", "content"];

  useEffect(() => {
    if (contentRef.current) {
      setContentHeight(contentRef.current.scrollHeight + 50);
    }
  }, [isOpen, activeInnerTab, slideText]);

  useEffect(() => {
    if (tabsContainerRef.current) {
      const activeTabIndex = TABS.indexOf(activeInnerTab);
      const buttons = Array.from(tabsContainerRef.current.querySelectorAll('.inner-tab-button'));
      const activeButton = buttons[activeTabIndex];
      if (activeButton) {
        setIndicatorPosition({
          left: activeButton.offsetLeft,
          width: activeButton.offsetWidth,
        });
      }
    }
  }, [activeInnerTab]);

  return (
    <div 
      className={`accordion-item ${isOpen ? 'is-open' : ''}`}
      style={{ animationDelay: `${index * 0.07}s` }}
    >
      <button className="accordion-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="header-title">
          <span>📄 Slide {index + 1}</span>
        </div>
        <div className="header-title">
          <span className={`sentiment-badge sentiment-${(sentiment.label || 'NEUTRAL').toLowerCase()}`}>
            {sentiment.label || "NEUTRAL"}
          </span>
          <div className={`chevron-wrapper ${isOpen ? 'is-open' : ''}`}>
            {isOpen ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>
        </div>
      </button>

      <div 
        className={`collapsible ${isOpen ? 'is-open' : ''}`}
        style={{ maxHeight: isOpen ? `${contentHeight}px` : '0' }}
      >
        <div ref={contentRef}>
          <div className="inner-tab-container" ref={tabsContainerRef}>
            <button 
              className={`inner-tab-button ${activeInnerTab === "feedback" ? 'active' : ''}`}
              onClick={() => setActiveInnerTab("feedback")}
            >
              AI Feedback
            </button>
            <button 
              className={`inner-tab-button ${activeInnerTab === "metrics" ? 'active' : ''}`}
              onClick={() => setActiveInnerTab("metrics")}
            >
              Metrics
            </button>
            <button 
              className={`inner-tab-button ${activeInnerTab === "content" ? 'active' : ''}`}
              onClick={() => setActiveInnerTab("content")}
            >
              Content
            </button>
            <div 
              className="active-tab-indicator"
              style={{ 
                left: `${indicatorPosition.left}px`, 
                width: `${indicatorPosition.width}px` 
              }}
            />
          </div>
          
          <div key={activeInnerTab} className="inner-tab-content"> 
            {activeInnerTab === "content" && (
              <div className="slide-content-wrapper">
                <pre className="slide-content-text">{slideText}</pre>
              </div>
            )}

            {activeInnerTab === "feedback" && (
              <div className="feedback-section">
                <div className="feedback-item">
                  <div className="feedback-icon clarity">💡</div>
                  <div className="feedback-content">
                    <h5 className="clarity-text">Clarity</h5>
                    <p>{aiFeedback.clarity_feedback || "AI clarity feedback not available."}</p>
                  </div>
                </div>
                <div className="feedback-item">
                  <div className="feedback-icon tone">🗣️</div>
                  <div className="feedback-content">
                    <h5 className="tone-text">Tone</h5>
                    <p>{aiFeedback.tone_feedback || "AI tone feedback not available."}</p>
                  </div>
                </div>
                <div className="feedback-item">
                  <div className="feedback-icon intent">🎯</div>
                  <div className="feedback-content">
                    <h5 className="intent-text">Intent</h5>
                    <p>{aiFeedback.intent_fulfillment_feedback || "AI intent feedback not available."}</p>
                  </div>
                </div>
              </div>
            )}

            {activeInnerTab === "metrics" && (
              <div className="metrics-grid">
                <div className={`metric-box ${getReadabilityColor(metrics.readability_interpretation)}`}>
                  <div className="metric-label">Readability</div>
                  <AnimatedCounter endValue={metrics.readability_interpretation || "N/A"} />
                </div>
                <div className="metric-box data-points">
                  <div className="metric-label">Data Points</div>
                  <AnimatedCounter endValue={metrics.quantitative_count ?? 0} />
                </div>
                <div className={`metric-box ${getSentimentColor(sentiment.label)}`}>
                  <div className="metric-label">Sentiment</div>
                  <div className="metric-value">
                    {sentiment.label || "NEUTRAL"}
                    {sentiment.score ? ` (${(sentiment.score * 100).toFixed(0)}%)` : ""}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function SlideBySlideTab({ data }) {
  const slides = data?.slide_texts || [];
  
  if (!slides.length) {
    return (
      <div className="accordion-container">
        <p style={{ color: '#cbd5e1', padding: '1rem', textAlign: 'center' }}>
          No slides available.
        </p>
      </div>
    );
  }

  return (
    <>
      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes contentFadeIn {
          from { opacity: 0; transform: scale(0.98); }
          to { opacity: 1; transform: scale(1); }
        }

        .accordion-container {
          width: 100%;
          display: flex;
          flex-direction: column;
          gap: 1rem;
          padding: 1rem;
        }

        .accordion-item {
          background: rgba(45, 55, 72, 0.65);
          border: 1px solid rgba(148, 163, 184, 0.25);
          border-radius: 16px;
          backdrop-filter: blur(14px) saturate(180%);
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
          overflow: hidden;
          transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s ease;
          opacity: 0;
          transform: translateY(20px);
          animation: fadeInUp 0.5s cubic-bezier(0.25, 0.1, 0.25, 1) forwards;
        }

        .accordion-item:hover {
          transform: translateY(-4px);
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }

        .accordion-item.is-open {
          border-color: rgba(34, 211, 238, 0.5);
          box-shadow: 0 10px 25px rgba(34, 211, 238, 0.12);
        }

        .accordion-header {
          width: 100%;
          padding: 1.25rem 1.5rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: transparent;
          border: none;
          cursor: pointer;
          color: #f8fafc;
          font-size: 1.05rem;
          font-weight: 600;
        }

        .accordion-header:hover {
          background: rgba(255, 255, 255, 0.04);
        }

        .header-title {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .chevron-wrapper {
          transition: transform 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
          color: #cbd5e1;
          display: flex;
          align-items: center;
          transform: rotate(0deg);
        }

        .chevron-wrapper.is-open {
          color: #22d3ee;
          transform: rotate(180deg);
        }

        .sentiment-badge {
          padding: 0.25rem 0.7rem;
          border-radius: 999px;
          font-size: 0.8rem;
          font-weight: 700;
          color: #0f172a;
        }

        .sentiment-badge.sentiment-positive {
          background: #34d399;
        }

        .sentiment-badge.sentiment-negative {
          background: #f87171;
        }

        .sentiment-badge.sentiment-neutral {
          background: #94a3b8;
        }

        .collapsible {
          border-top: 1px solid rgba(148, 163, 184, 0.1);
          background: rgba(15, 23, 42, 0.7);
          overflow: hidden;
          transition: max-height 0.4s cubic-bezier(0.25, 0.1, 0.25, 1), padding 0.4s cubic-bezier(0.25, 0.1, 0.25, 1);
          padding: 0 1.5rem;
          max-height: 0;
        }

        .collapsible.is-open {
          padding: 1.5rem;
        }

        .inner-tab-container {
          position: relative;
          display: flex;
          background-color: rgba(30, 41, 59, 0.6);
          border-radius: 99px;
          padding: 0.25rem;
          margin-bottom: 1.5rem;
          width: fit-content;
        }

        .inner-tab-button {
          padding: 0.5rem 1.25rem;
          border: none;
          background: none;
          color: #cbd5e1;
          font-size: 0.9rem;
          font-weight: 600;
          cursor: pointer;
          transition: color 0.3s ease;
          z-index: 2;
          border-radius: 99px;
        }

        .inner-tab-button.active {
          color: #f8fafc;
        }

        .inner-tab-button:hover {
          color: #f8fafc;
        }

        .active-tab-indicator {
          position: absolute;
          top: 0.25rem;
          bottom: 0.25rem;
          background: rgba(51, 65, 85, 0.8);
          border-radius: 99px;
          transition: left 0.3s cubic-bezier(0.25, 0.1, 0.25, 1), width 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
          z-index: 1;
        }

        .inner-tab-content {
          border-radius: 10px;
          min-height: 100px;
          animation: contentFadeIn 0.5s ease forwards;
          transform-origin: top;
        }

        .slide-content-text {
          white-space: pre-wrap;
          font-family: "Inter", sans-serif;
          font-size: 1rem;
          line-height: 1.6;
          color: #f8fafc;
          margin: 0;
          padding: 1rem;
          background: rgba(20, 25, 35, 0.8);
          border: 1px solid rgba(148, 163, 184, 0.12);
          border-radius: 10px;
        }

        .feedback-section {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          padding: 1rem;
          background: rgba(20, 25, 35, 0.8);
          border: 1px solid rgba(148, 163, 184, 0.12);
          border-radius: 10px;
        }

        .feedback-item {
          display: flex;
          align-items: flex-start;
          gap: 1rem;
        }

        .feedback-icon {
          flex-shrink: 0;
          width: 36px;
          height: 36px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1rem;
        }

        .feedback-icon.clarity {
          background: rgba(34, 211, 238, 0.15);
          color: #22d3ee;
        }

        .feedback-icon.tone {
          background: rgba(168, 85, 247, 0.15);
          color: #a855f7;
        }

        .feedback-icon.intent {
          background: rgba(16, 185, 129, 0.15);
          color: #10b981;
        }

        .feedback-content h5 {
          margin: 0 0 0.25rem 0;
          font-size: 1rem;
          font-weight: 600;
        }

        .feedback-content p {
          margin: 0;
          color: #cbd5e1;
          line-height: 1.6;
        }

        .clarity-text {
          background: linear-gradient(90deg, #22d3ee, #3b82f6);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .tone-text {
          background: linear-gradient(90deg, #a855f7, #ec4899);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .intent-text {
          background: linear-gradient(90deg, #10b981, #34d399);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1.5rem;
        }

        .metric-box {
          border-radius: 12px;
          padding: 1.25rem;
          border: 1px solid rgba(148, 163, 184, 0.25);
          text-align: center;
          position: relative;
          overflow: hidden;
          box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.05);
          transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        }

        .metric-box:hover {
          transform: translateY(-5px);
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }

        .metric-label {
          color: #cbd5e1;
          font-size: 0.8rem;
          text-transform: uppercase;
          margin-bottom: 0.75rem;
        }

        .metric-value {
          color: #f8fafc;
          font-size: 1.75rem;
          font-weight: 700;
          line-height: 1;
        }

        /* Readability Color Themes */
        .metric-box.readability-easy {
          background: linear-gradient(160deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.3));
          border-color: rgba(16, 185, 129, 0.4);
        }

        .metric-box.readability-easy:hover {
          border-color: #10b981;
          box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3), 0 0 15px rgba(16, 185, 129, 0.3);
        }

        .metric-box.readability-easy .metric-value {
          background: linear-gradient(135deg, #10b981, #34d399);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .metric-box.readability-moderate {
          background: linear-gradient(160deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.3));
          border-color: rgba(251, 191, 36, 0.4);
        }

        .metric-box.readability-moderate:hover {
          border-color: #fbbf24;
          box-shadow: 0 8px 25px rgba(251, 191, 36, 0.3), 0 0 15px rgba(251, 191, 36, 0.3);
        }

        .metric-box.readability-moderate .metric-value {
          background: linear-gradient(135deg, #fbbf24, #f59e0b);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .metric-box.readability-hard {
          background: linear-gradient(160deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.3));
          border-color: rgba(239, 68, 68, 0.4);
        }

        .metric-box.readability-hard:hover {
          border-color: #ef4444;
          box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3), 0 0 15px rgba(239, 68, 68, 0.3);
        }

        .metric-box.readability-hard .metric-value {
          background: linear-gradient(135deg, #ef4444, #dc2626);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .metric-box.readability-neutral {
          background: linear-gradient(160deg, rgba(148, 163, 184, 0.2), rgba(100, 116, 139, 0.3));
          border-color: rgba(148, 163, 184, 0.4);
        }

        .metric-box.readability-neutral:hover {
          border-color: #94a3b8;
          box-shadow: 0 8px 25px rgba(148, 163, 184, 0.3), 0 0 15px rgba(148, 163, 184, 0.3);
        }

        /* Sentiment Color Themes */
        .metric-box.sentiment-positive {
          background: linear-gradient(160deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.3));
          border-color: rgba(59, 130, 246, 0.4);
        }

        .metric-box.sentiment-positive:hover {
          border-color: #3b82f6;
          box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3), 0 0 15px rgba(59, 130, 246, 0.3);
        }

        .metric-box.sentiment-positive .metric-value {
          background: linear-gradient(135deg, #3b82f6, #2563eb);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .metric-box.sentiment-negative {
          background: linear-gradient(160deg, rgba(236, 72, 153, 0.2), rgba(219, 39, 119, 0.3));
          border-color: rgba(236, 72, 153, 0.4);
        }

        .metric-box.sentiment-negative:hover {
          border-color: #ec4899;
          box-shadow: 0 8px 25px rgba(236, 72, 153, 0.3), 0 0 15px rgba(236, 72, 153, 0.3);
        }

        .metric-box.sentiment-negative .metric-value {
          background: linear-gradient(135deg, #ec4899, #db2777);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .metric-box.sentiment-neutral {
          background: linear-gradient(160deg, rgba(139, 92, 246, 0.2), rgba(124, 58, 237, 0.3));
          border-color: rgba(139, 92, 246, 0.4);
        }

        .metric-box.sentiment-neutral:hover {
          border-color: #8b5cf6;
          box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3), 0 0 15px rgba(139, 92, 246, 0.3);
        }

        .metric-box.sentiment-neutral .metric-value {
          background: linear-gradient(135deg, #8b5cf6, #7c3aed);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        /* Data Points - Unique Theme */
        .metric-box.data-points {
          background: linear-gradient(160deg, rgba(34, 211, 238, 0.2), rgba(6, 182, 212, 0.3));
          border-color: rgba(34, 211, 238, 0.4);
        }

        .metric-box.data-points:hover {
          border-color: #22d3ee;
          box-shadow: 0 8px 25px rgba(34, 211, 238, 0.3), 0 0 15px rgba(34, 211, 238, 0.3);
        }

        .metric-box.data-points .metric-value {
          background: linear-gradient(135deg, #22d3ee, #06b6d4);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
      `}</style>

      <div className="accordion-container">
        {slides.map((_, i) => (
          <SlideItem key={i} index={i} slideData={data} />
        ))}
      </div>
    </>
  );
}