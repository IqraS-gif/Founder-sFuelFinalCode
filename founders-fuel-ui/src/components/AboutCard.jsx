import React from 'react';
import styled from 'styled-components';

// A map of features to their corresponding SVG icons (unchanged)
const features = [
  { 
    icon: <path d="M3 13.335V18a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1v-4.665c0-1.205-.826-2.264-2-2.583V3a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v7.752c-1.174.319-2 1.378-2 2.583zM12 4h5v7h-5V4zM5 4h6v7H5V4zm14 9H5v3h14v-3z"/>,
    text: "Refine your pitch deck with data-driven AI analysis."
  },
  { 
    icon: <path d="M17.5 2.5a2.5 2.5 0 0 0-5 0A2.5 2.5 0 0 0 17.5 2.5zM20 17.5a2.5 2.5 0 0 0-5 0A2.5 2.5 0 0 0 20 17.5zM4 17.5a2.5 2.5 0 0 0-2.5 2.5A2.5 2.5 0 0 0 4 22.5a2.5 2.5 0 0 0 2.5-2.5A2.5 2.5 0 0 0 4 17.5zM12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm-7.5 5a7.5 7.5 0 1 1 15 0 7.5 7.5 0 0 1-15 0z"/>,
    text: "Get intelligent recommendations for investors and accelerators."
  },
  {
    icon: <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/>,
    text: "Get data-driven startup evaluations with an AI-powered RAG chatbot that analyzes 7 core domains + live web seraching."
  },
  {
    icon: <path d="M7 2a1 1 0 0 0-1 1v1a1 1 0 0 0 1 1h1V2H7zM6 7v10c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6zm7-3h-2V2h2v2z"/>,
    text: "Practice for pitch interviews with an AI and receive instant feedback."
  },
   {
    icon: <><path d="M12 2a10 10 0 1 0 10 10A10.011 10.011 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8.009 8.009 0 0 1-8 8z" /><path d="M12 10a2 2 0 1 0 2 2a2.002 2.002 0 0 0-2-2z" /></>,
    text: "Predicts the success probability of a startup pitch using ML trained model in real campaign data."
  },
];


const AboutCard = () => {
  return (
    <StyledWrapper>
      <div className="card">
        <div className="card__border" />

        <div className="card__header">
          <h2 className="card__title">About Founder's Fuel</h2>
          <p className="card__mission">
            We empower founders with cutting-edge AI tools to turn visionary ideas into impactful, successful businesses. Our goal is to provide clarity, strategy, and precision for every entrepreneur.
          </p>
        </div>
        
        <div className="card__body">
          <ul className="features__list">
            {features.map((feature, index) => (
              <li key={index} className="feature__item">
                <div className="feature__icon">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                    {feature.icon}
                  </svg>
                </div>
                <p className="feature__text">{feature.text}</p>
              </li>
            ))}
          </ul>
        </div>

      </div>
    </StyledWrapper>
  );
}

// --- STYLES ---
const StyledWrapper = styled.div`
  .card {
    --white: hsl(0, 0%, 100%);
    --black: hsl(240, 15%, 9%);
    --paragraph: hsl(0, 0%, 83%);
    --line: hsl(240, 9%, 17%);
    --primary: hsl(189, 92%, 58%);
    --primary-glow: hsla(189, 92%, 58%, 0.4);

    position: relative;
    display: flex;
    flex-direction: column;
    gap: 2rem;
    
    width: 60rem;
    max-width: 90vw;
    padding: 3rem;
    
    background-color: hsla(240, 15%, 9%, 1);
    background-image: radial-gradient(at 0% 64%, hsl(189, 99%, 16%) 0px, transparent 85%),
                      radial-gradient(at 100% 99%, hsl(188, 94%, 13%) 0px, transparent 85%);
    border-radius: 1.5rem;
    box-shadow: 0px -16px 24px 0px rgba(0, 0, 0, 0.5) inset;
    
    /* --- CHANGE #1: The hover effect has been removed from here --- */
  }

  .card .card__border {
    overflow: hidden; pointer-events: none; position: absolute;
    z-index: -10; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: calc(100% + 2px); height: calc(100% + 2px);
    background-image: linear-gradient(0deg, hsl(0, 0%, 100%) -50%, hsl(0, 0%, 40%) 100%);
    border-radius: 1.5rem;
  }

  .card .card__border::before {
    content: ""; pointer-events: none; position: fixed;
    z-index: 200; top: 50%; left: 50%;
    transform: translate(-50%, -50%); transform-origin: center;
    width: 200%; height: 10rem;
    background-image: linear-gradient(0deg, hsla(0, 0%, 100%, 0) 0%, hsl(189, 100%, 50%) 40%, hsl(189, 100%, 50%) 60%, hsla(0, 0%, 40%, 0) 100%);
    
    /* --- CHANGE #3: ADDED the new pulse-glow animation here. --- */
    /* The border now rotates AND pulses at the same time. */
    animation: 
      rotate 8s linear infinite,
      pulse-glow 4s ease-in-out infinite;
  }

  @keyframes rotate {
    to { transform: translate(-50%, -50%) rotate(360deg); }
  }

  /* --- CHANGE #2: A NEW keyframe animation for the "breathing" glow --- */
  @keyframes pulse-glow {
    0% { opacity: 0.5; }
    50% { opacity: 1; }
    100% { opacity: 0.5; }
  }
  
  /* All styles below this line are unchanged */
  .card__header {
    text-align: center;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--line);
  }
  
  .card__title {
    font-size: 2.25rem;
    font-weight: 700;
    color: var(--white);
    margin: 0 0 1rem 0;
    text-shadow: 0 0 10px var(--primary-glow);
  }

  .card__mission {
    font-size: 1.1rem;
    color: var(--paragraph);
    line-height: 1.6;
    max-width: 80ch;
    margin: 0 auto;
  }

  .features__list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
  }

  .feature__item {
    display: flex;
    align-items: center;
    gap: 1rem;
    background-color: rgba(0,0,0,0.2);
    padding: 1rem;
    border-radius: 0.75rem;
    border: 1px solid var(--line);
  }
  
  .feature__icon {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background-color: hsla(189, 99%, 16%, 0.5);
    border: 1px solid var(--primary);
    
    svg {
      width: 24px;
      height: 24px;
      color: var(--primary);
    }
  }

  .feature__text {
    font-size: 1rem;
    line-height: 1.5;
    color: var(--paragraph);
    margin: 0;
  }
`;

export default AboutCard;