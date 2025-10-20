// src/App.jsx

import React from 'react';
// 1. IMPORT createGlobalStyle
import styled, { createGlobalStyle } from 'styled-components'; 
import Navbar from '../components/Navbar';
import AnimatedGauge from '../components/AnimatedGauge';
import CardCarousel from '../components/CardCarousel';
import AboutCard from '../components/AboutCard'; 
import bgVideo from '../assets/background.mp4'; 


// 2. DEFINE THE GLOBAL STYLE TO HIDE THE SCROLLBAR
const GlobalStyle = createGlobalStyle`
  /* Hides the scrollbar for Chrome, Safari, and Edge */
  body::-webkit-scrollbar {
    display: none; 
  }

  /* Hides the scrollbar for Firefox */
  html {
    scrollbar-width: none;
  }
`;

// This is the container for the ENTIRE scrollable page
const AppContainer = styled.div`
  background: transparent;
  color: white;
  overflow-x: hidden;
`;

const VideoBackground = styled.video`
  position: fixed;
  right: 0;
  bottom: 0;
  min-width: 100%;
  min-height: 100%;
  z-index: -1;
  opacity: 0.5;
  object-fit: cover;
`;

// This is now the container for ONLY the top hero section
const HeroSection = styled.main`
  display: flex;
  align-items: center;
  justify-content: center;
  padding-top: 80px; 
  box-sizing: border-box;
`;

const LeftSide = styled.div`
  flex-basis: 50%;
  padding: 2rem 4rem;
  box-sizing: border-box;
`;

const RightSide = styled.div`
  flex-basis: 50%;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
`;
    
const Quote = styled.h1`
  font-family: 'Inter', sans-serif;
  font-size: clamp(2.5rem, 5vw, 4.5rem);
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -2px;
  color: #e2e8f0;
  text-shadow: 0 0 20px rgba(20, 184, 166, 0.4);
`;

// A generic container for the new sections
const ContentSection = styled.section`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 4rem 2rem;
  box-sizing: border-box;
  position: relative;
`;

// A title for our new sections
const SectionTitle = styled.h2`
  font-family: 'Inter', sans-serif;
  font-size: 3rem;
  font-weight: 700;
  color: #fff;
  text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
  margin-bottom: 3rem; // Increased margin for better spacing
  text-align: center;
`;

function LandingPage() {
  return (
    <AppContainer>
      {/* 3. ADD THE GlobalStyle COMPONENT TO APPLY THE FIX */}
      <GlobalStyle />

      <VideoBackground autoPlay loop muted>
        <source src={bgVideo} type="video/mp4" />
      </VideoBackground>
      
      <Navbar />

      {/* --- Section 1: The Hero (Your existing layout) --- */}
      <HeroSection>
        <LeftSide>
          <Quote>Fueling the next generation of founders.</Quote>
        </LeftSide>
        <RightSide>
          <AnimatedGauge percentage={65} />
        </RightSide>
      </HeroSection>

      {/* --- Section 2: The Card Carousel --- */}
      <ContentSection>
        <SectionTitle>Explore Our Features</SectionTitle>
        <CardCarousel />
      </ContentSection>

      {/* --- 2. ADD the new About Us section --- */}
      <ContentSection>
        <SectionTitle>About Us</SectionTitle>
        <AboutCard />
      </ContentSection>

    </AppContainer>
  );
}

export default LandingPage;