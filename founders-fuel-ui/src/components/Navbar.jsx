// src/components/Navbar.jsx
import React from "react";
import styled, { keyframes } from "styled-components";

// --- ANIMATIONS ---
const fadeInDown = keyframes`
  from { 
    opacity: 0; 
    transform: translateY(-20px); 
  } 
  to { 
    opacity: 1; 
    transform: translateY(0); 
  }
`;

const underlineGrow = keyframes`
  from { transform: scaleX(0); } 
  to { transform: scaleX(1); }
`;

// --- STYLED COMPONENTS ---
const NavContainer = styled.nav`
  padding: 1rem 3rem;
  background: rgba(15, 23, 42, 0.5);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  z-index: 100;
  box-sizing: border-box;
  animation: ${fadeInDown} 0.8s ease-out;
`;

const LogoContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;

  svg {
    width: 28px;
    height: 28px;
    color: #14b8a6;
  }

  span {
    font-family: "Inter", sans-serif;
    font-size: 2.5rem;
    font-weight: 1000;
    color: #e2e8f0;

    /* Reduced Glow System */
    text-shadow: 
      0 0 5px rgba(20, 184, 166, 0.5),
      0 0 10px rgba(20, 184, 166, 0.3);

    /* Subtle text outline */
    -webkit-text-stroke: 0.5px rgba(20, 184, 166, 0.2);

    /* Keep the animation */
    animation: ${fadeInDown} 0.8s ease-out forwards;
    opacity: 0;
  }
`;

const NavLinks = styled.div`
  display: flex;
  align-items: center;
  gap: 2.5rem;
`;

const StyledLink = styled.a`
  font-family: "Inter", sans-serif;
  color: #cbd5e1;
  text-decoration: none;
  font-weight: 800;
  position: relative;
  padding-bottom: 0.5rem;
  transition: color 0.2s ease-in-out;
  font-size: 1.5rem;

  &::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background-color: #14b8a6;
    transform: scaleX(0);
    transform-origin: center;
    transition: transform 0.3s ease-out;
  }

  &:hover {
    color: #ffffff;
    &::after {
      transform: scaleX(1);
    }
  }
`;

const CtaButton = styled.button`
  font-family: "Inter", sans-serif;
  font-size: 1rem;
  font-weight: 600;
  color: #ffffff;
  background: transparent;
  border: 2px solid #14b8a6;
  border-radius: 8px;
  padding: 0.6rem 1.2rem;
  cursor: pointer;
  transition: background-color 0.3s ease, box-shadow 0.3s ease,
    transform 0.2s ease;

  &:hover {
    background-color: #14b8a6;
    box-shadow: 0 0 20px rgba(20, 184, 166, 0.6);
    transform: translateY(-2px);
  }

  &:active {
    transform: translateY(0);
  }
`;

// --- THE COMPONENT ---
const Navbar = () => {
  return (
    <NavContainer>
      <LogoContainer>
        {/* Simple flame icon */}
        <svg
          fill="currentColor"
          viewBox="0 0 20 20"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            fillRule="evenodd"
            d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414l-4 4a1 1 0 11-1.414-1.414l4-4a1 1 0 011.414 0zm8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414z"
            clipRule="evenodd"
          />
        </svg>
        <span>Founders Fuel</span>
      </LogoContainer>

      <NavLinks>
        <StyledLink href="https://founder-s-fuel-final-code-1y38.vercel.app/">Home</StyledLink>
        <StyledLink href="#">About</StyledLink>
        <StyledLink href="#">Features</StyledLink>
        <CtaButton>Get Started</CtaButton>
      </NavLinks>
    </NavContainer>
  );
};

export default Navbar;
