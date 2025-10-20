import React from 'react';
import styled from 'styled-components';

const AnimatedGauge = () => {
  return (
    <StyledWrapper>
      <div className="full-progress">
        <div className="content">
          <div className="text">
            <span>FOUNDER'S</span>
            <span>FUEL</span>
          </div>
          <div className="progresses">
            <div className="main-prog">
              <div className="separ" />
              <div className="separ" />
              <div className="separ" />
              <div className="separ" />
              <div className="separ" />
            </div>
            <div className="second-circle" />
            <div className="third-circle" />
            <div className="base-circle">
              <div className="separ2" />
              <div className="separ2" />
              <div className="separ2" />
              <div className="separ2" />
              <div className="separ2" />
              <div className="separ2" />
              <div className="separ2" />
              <div className="separ2" />
            </div>
            <div className="last-progress" />
          </div>
        </div>
      </div>
    </StyledWrapper>
  );
}

const StyledWrapper = styled.div`
  /* --- THE CORRECTED CODE TO MOVE THE COMPONENT UP --- */
  /* This moves the element visually AND its space in the layout, which removes the unwanted gap. */
  transform: translateY(-150px);
  /* ---------------------------------------------------- */

  @keyframes move-second-circle {
    50% { bottom: -42%; }
    100% { bottom: -50%; }
  }
  @keyframes move-third-circle {
    50% { bottom: -66%; }
    100% { bottom: -75%; }
  }
  @keyframes move-last-progress {
    50% { bottom: -64%; }
    100% { bottom: -70%; }
  }
  @keyframes move-base {
    50% { bottom: -94%; }
    100% { bottom: -100%; }
  }

  .main-prog {
    margin: auto; 
    width: 300px; 
    height: 300px; 
    border-radius: 300px;
    position: relative; 
    background-color: #000814;
  }
  .main-prog::after {
    content: ""; 
    display: block; 
    position: absolute; 
    left: -40px; 
    top: -40px;
    right: -40px; 
    bottom: -40px; 
    background-color: transparent;
    background-image: conic-gradient(#0ff, #0ff 65%, #001233 65%);
    z-index: -1; 
    border-radius: 300px;
  }
  .main-prog::before {
    content: ""; 
    display: block; 
    position: absolute; 
    border: 15px solid #001233;
    height: 430px; 
    width: 430px; 
    left: 50%; 
    top: 50%;
    transform: translate(-50%, -50%); 
    border-radius: 50%;
  }
  .main-prog > .separ {
    height: 200px; 
    width: 10px; 
    background: #000814;
    position: absolute; 
    top: 50%; 
    left: 50%;
    transform: translate(-50%, -100%); 
    transform-origin: bottom center;
  }
  .main-prog > .separ:nth-child(1) { transform: translate(-50%, -100%) rotate(0deg); }
  .main-prog > .separ:nth-child(2) { transform: translate(-50%, -100%) rotate(72deg); }
  .main-prog > .separ:nth-child(3) { transform: translate(-50%, -100%) rotate(144deg); }
  .separ:nth-child(3)::before {
    content: "40"; 
    color: #0ff; 
    position: absolute;
    transform: translate(-50%, -36px) rotate(180deg);
    font-size: 22px; 
    font-family: sans-serif; 
    font-weight: bold;
  }
  .main-prog > .separ:nth-child(4) { transform: translate(-50%, -100%) rotate(216deg); }
  .separ:nth-child(4)::before {
    content: "60"; 
    color: #0ff; 
    position: absolute;
    transform: translate(-50%, -36px) rotate(180deg);
    font-size: 22px; 
    font-family: sans-serif; 
    font-weight: bold;
  }
  .main-prog > .separ:nth-child(5) { transform: translate(-50%, -100%) rotate(288deg); }
  
  .content { 
    perspective: 500px; 
    width: 300px; 
    height: 300px; 
  }
  .progresses { 
    transform: rotateX(50deg) scaleY(0.6); 
    position: relative; 
  }

  .second-circle {
    border: 6px solid transparent; 
    border-top-color: rgba(0, 255, 255, 0.2);
    border-left-color: rgba(0, 255, 255, 0.2); 
    border-right-color: rgba(0, 255, 255, 0.2);
    border-bottom-color: #001233; 
    height: 250px; 
    width: 250px;
    position: absolute; 
    left: 50%; 
    bottom: -50%; 
    transform: translateX(-50%);
    border-radius: 50%; 
    z-index: 2; 
    animation: move-second-circle 3s ease-in-out infinite;
  }
  .third-circle {
    border: 3px solid #0ff; 
    height: 130px; 
    width: 130px; 
    position: absolute;
    left: 50%; 
    bottom: -75%; 
    transform: translateX(-50%); 
    border-radius: 50%;
    z-index: 1; 
    animation: move-third-circle 3s ease-in-out infinite; 
    animation-delay: 300ms;
  }
  .last-progress {
    position: absolute; 
    margin: auto; 
    left: 50%; 
    bottom: -70%;
    transform: translateX(-50%); 
    border-radius: 50%; 
    z-index: 0;
    overflow: visible; 
    animation: move-last-progress 3s ease-in-out infinite; 
    animation-delay: 600ms;
  }
  .last-progress::after {
    content: ""; 
    display: block; 
    position: absolute; 
    left: 50%; 
    top: 50%;
    transform: translate(-50%, -50%); 
    margin: auto; 
    background: #000814;
    height: 58px; 
    width: 58px; 
    border-radius: 50%;
  }
  .last-progress::before {
    content: ""; 
    display: block; 
    position: absolute; 
    left: -40px; 
    top: -40px;
    right: -40px; 
    bottom: -40px; 
    background-color: transparent;
    background-image: conic-gradient(#0ff, #0ff 65%, #001233 65%);
    z-index: 0; 
    border-radius: 300px;
  }
  .base-circle {
    border: 40px solid rgba(0, 255, 255, 0.05); 
    height: 180px; 
    width: 180px;
    position: absolute; 
    left: 50%; 
    bottom: -100%; 
    transform: translateX(-50%);
    border-radius: 50%; 
    z-index: -2; 
    animation: move-base 3s ease-in-out infinite; 
    animation-delay: 600ms;
  }
  .base-circle > .separ2 {
    height: 100px; 
    width: 3px; 
    background-color: #000814;
    position: absolute; 
    top: 50%; 
    left: 50%;
    transform: translate(-50%, -100%); 
    transform-origin: bottom center;
  }
  .separ2:nth-child(2) { transform: translate(-50%, -100%) rotate(45deg); }
  .separ2:nth-child(3) { transform: translate(-50%, -100%) rotate(90deg); }
  .separ2:nth-child(4) { transform: translate(-50%, -100%) rotate(135deg); }
  .separ2:nth-child(5) { transform: translate(-50%, -100%) rotate(180deg); }
  .separ2:nth-child(6) { transform: translate(-50%, -100%) rotate(225deg); }
  .separ2:nth-child(7) { transform: translate(-50%, -100%) rotate(270deg); }
  .separ2:nth-child(8) { transform: translate(-50%, -100%) rotate(315deg); }

  .text {
    position: absolute; 
    top: 35%; 
    left: 50%; 
    color: #0ff; 
    z-index: 4;
    transform: translate(-50%, -50%); 
    font-family: 'Inter', sans-serif;
    display: flex; 
    flex-direction: column; 
    align-items: center;
    text-transform: uppercase;
    letter-spacing: 3px;
  }
  .text span:first-child {
    font-size: 42px; 
    font-weight: 900; 
    line-height: 1;
    margin-bottom: 5px;
  }
  .text span:last-child {
    font-size: 62px;
    font-weight: 900;
    background: linear-gradient(90deg, #0ff, #0af);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(0 0 5px rgba(0, 255, 255, 0.7));
  }
  .full-progress {
    width: 550px;
    display: flex;
    justify-content: center;
    background: transparent;
    transform: scale(1.2);
  }
`;

export default AnimatedGauge;