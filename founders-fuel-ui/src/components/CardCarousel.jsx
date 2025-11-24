// // // src/components/CardCarousel.jsx

// import React, { useState } from 'react'
// import styled from 'styled-components'

// // --- Card Data: Updated with your new data ---
// const cardData = [
//   {
//     id: 1,
//     title: 'Pitch Deck Analysis',
//     description: 'Analyze your pitch deck with AI to get insights, clarity scores, and improvement suggestions.',
//     color: '142, 249, 252',
//     link: 'http://localhost:8501/?page=pitch_deck_analysis',
//   },
//   {
//     id: 2,
//     title: 'Investors & Accelerators Recommendations',
//     description: 'Get AI-driven suggestions for the most suitable investors and accelerators for your startup.',
//     color: '142, 252, 157',
//     link: 'http://localhost:8502/?page=investor_recommendations',
//   },
//   {
//     id: 3,
//     title: 'Success Prediction',
//     description: 'Predict your startup’s success probability using AI models trained on real-world startup data.',
//     color: '252, 252, 142',
//     link: 'http://localhost:8504/?page=success_prediction',
//   },
//   {
//     id: 4,
//     title: 'Startup Analysis Chatbot',
//     description: 'Data-driven startup evaluations with an AI-powered RAG chatbot.',
//     color: '252, 142, 142',
//     link: 'http://localhost:8503/?page=startup_analysis_chatbot',
//   },
//   {
//     id: 5,
//     title: 'Live Competitors & Benchmark',
//     description: 'Compare your startup against live competitors and benchmark performance using AI analytics.',
//     color: '204, 142, 252',
//     link: '',
//   },
// ]

// // --- The Carousel Component (No changes needed here) ---
// const CardCarousel = () => {
//   const [currentIndex, setCurrentIndex] = useState(0)
//   const totalCards = cardData.length

//   const handlePrev = () => {
//     setCurrentIndex(prevIndex => (prevIndex === 0 ? totalCards - 1 : prevIndex - 1))
//   }

//   const handleNext = () => {
//     setCurrentIndex(prevIndex => (prevIndex === totalCards - 1 ? 0 : prevIndex + 1))
//   }

//   const rotationAngle = -currentIndex * (360 / totalCards)

//   return (
//     <CarouselWrapper>
//       <div className="wrapper">
//         <div
//           className="inner"
//           style={{
//             '--quantity': totalCards,
//             '--rotation-angle': `${rotationAngle}deg`,
//           }}
//         >
//           {cardData.map((card, index) => (
//             <div
//               key={card.id}
//               className="card"
//               style={{ '--index': index, '--color-card': card.color }}
//             >
//               <div className="card-content">
//                 <h3>{card.title}</h3>
//                 <p>{card.description}</p>
//                 <a href={card.link} className="try-me-button">
//                   Try Me
//                 </a>
//               </div>
//               <div className="img" />
//             </div>
//           ))}
//         </div>
//       </div>
//       <Controls>
//         <ControlButton onClick={handlePrev}>
//           <svg
//             xmlns="http://www.w3.org/2000/svg"
//             fill="none"
//             viewBox="0 0 24 24"
//             strokeWidth={1.5}
//             stroke="currentColor"
//           >
//             <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
//           </svg>
//         </ControlButton>
//         <ControlButton onClick={handleNext}>
//           <svg
//             xmlns="http://www.w3.org/2000/svg"
//             fill="none"
//             viewBox="0 0 24 24"
//             strokeWidth={1.5}
//             stroke="currentColor"
//           >
//             <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
//           </svg>
//         </ControlButton>
//       </Controls>
//     </CarouselWrapper>
//   )
// }

// // --- STYLES ---
// const CarouselWrapper = styled.div`
//   width: 100%;
//   height: 650px;
//   position: relative;
//   display: flex;
//   justify-content: center;
//   align-items: center;

//   .wrapper {
//     width: 100%;
//     height: 100%;
//     position: relative;
//     display: flex;
//     align-items: center;
//     justify-content: center;
//   }

//   .inner {
//     /* --- CSS FIX #1: Adjusting card dimensions and carousel radius --- */
//     --w: 250px; /* Increased card width to fit longer text */
//     --h: 340px; /* Increased card height slightly */
//     --translateZ: 480px; /* Increased radius to prevent wider cards from overlapping */
//     --rotateX: -10deg;
//     --perspective: 1500px;

//     position: absolute;
//     width: var(--w);
//     height: var(--h);
//     z-index: 2;
//     transform-style: preserve-3d;
//     transform: perspective(var(--perspective)) rotateX(var(--rotateX))
//       rotateY(var(--rotation-angle, 0deg));
//     transition: transform 0.8s cubic-bezier(0.25, 1, 0.5, 1);
//   }

//   .card {
//     position: absolute;
//     border: 2px solid rgba(var(--color-card), 0.7);
//     border-radius: 16px;
//     overflow: hidden;
//     inset: 0;
//     transform: rotateY(calc((360deg / var(--quantity)) * var(--index)))
//       translateZ(var(--translateZ));
//     box-shadow: 0 0 20px rgba(var(--color-card), 0.2);
//   }

//   .img {
//     position: absolute;
//     inset: 0;
//     z-index: -1;
//     background: radial-gradient(
//       circle at 50% 50%,
//       rgba(var(--color-card), 0.25) 0%,
//       rgba(15, 23, 42, 0.6) 80%
//     );
//   }

//   .card-content {
//     padding: 1rem;
//     height: 100%;
//     display: flex;
//     flex-direction: column;
//     justify-content: center;
//     align-items: center;
//     gap: 1rem;
//     color: #fff;
//     text-shadow: 0 1px 3px #000;
//     text-align: center; /* Added for better centering of multi-line text */

//     h3 {
//       /* --- CSS FIX #2: Reduced font size for longer titles --- */
//       font-size: 1.5rem; /* Reduced from 1.75rem */
//       font-weight: 750;
//       color: rgba(var(--color-card), 1);
//       margin: 0;
//     }
//     p {
//       font-size: 1rem;
//       color: #e2e8f0;
//       line-height: 1.5;
//       margin: 0;
//     }
//     .try-me-button {
//       margin-top: 1rem;
//       padding: 0.75rem 1.5rem;
//       background-color: rgba(var(--color-card), 0.9);
//       color: #0f172a;
//       border: none;
//       border-radius: 8px;
//       font-weight: 600;
//       cursor: pointer;
//       text-decoration: none;
//       transition: all 0.2s ease;

//       &:hover {
//         background-color: rgba(var(--color-card), 1);
//         transform: scale(1.05);
//         box-shadow: 0 0 25px rgba(var(--color-card), 0.5);
//       }
//     }
//   }
// `

// const Controls = styled.div`
//   position: absolute;
//   bottom: 60px;
//   display: flex;
//   gap: 1.5rem;
//   z-index: 10;
// `

// const ControlButton = styled.button`
//   width: 50px;
//   height: 50px;
//   border-radius: 50%;
//   background: rgba(255, 255, 255, 0.1);
//   border: 1px solid rgba(255, 255, 255, 0.2);
//   color: #fff;
//   cursor: pointer;
//   display: flex;
//   justify-content: center;
//   align-items: center;
//   transition: all 0.2s ease;

//   &:hover {
//     background: rgba(0, 255, 255, 0.2);
//     border-color: #0ff;
//   }

//   svg {
//     width: 24px;
//     height: 24px;
//   }
// `

// export default CardCarousel



// // src/components/CardCarousel.jsx
// // import React, { useState } from 'react';
// // import styled from 'styled-components';

// // // --- Card Data ---
// // const cardData = [
// //   {
// //     id: 1,
// //     title: "Pitch Deck Analysis",
// //     description: "Get insights, clarity scores, and suggestions.",
// //     color: "142, 249, 252",
// //     link: "http://localhost:8501/?page=ai_analysis"
// //   },
// //   {
// //     id: 2,
// //     title: "Data Visualization",
// //     description: "Transform complex datasets into beautiful, interactive charts.",
// //     color: "142, 252, 157",
// //     link: "http://localhost:8501/?page=data_visualization"
// //   },
// //   {
// //     id: 3,
// //     title: "Secure Platform",
// //     description: "Built on a foundation of enterprise-grade security and compliance.",
// //     color: "252, 252, 142",
// //     link: "http://localhost:8501/?page=security"
// //   },
// //   {
// //     id: 4,
// //     title: "Global Reach",
// //     description: "Connect with founders and investors from around the world, instantly.",
// //     color: "252, 142, 142",
// //     link: "/features/global-reach"
// //   },
// //   {
// //     id: 5,
// //     title: "Instant Funding",
// //     description: "A streamlined process to get your vision funded faster than ever.",
// //     color: "204, 142, 252",
// //     link: "/features/instant-funding"
// //   }
// // ];

// // // --- Carousel Component ---
// // const CardCarousel = () => {
// //   const [currentIndex, setCurrentIndex] = useState(0);
// //   const totalCards = cardData.length;

// //   const handlePrev = () => {
// //     setCurrentIndex(prevIndex => (prevIndex === 0 ? totalCards - 1 : prevIndex - 1));
// //   };

// //   const handleNext = () => {
// //     setCurrentIndex(prevIndex => (prevIndex === totalCards - 1 ? 0 : prevIndex + 1));
// //   };

// //   const rotationAngle = -currentIndex * (360 / totalCards);

// //   return (
// //     <CarouselWrapper>
// //       <div className="wrapper">
// //         <div
// //           className="inner"
// //           style={{
// //             '--quantity': totalCards,
// //             '--rotation-angle': `${rotationAngle}deg`
// //           }}
// //         >
// //           {cardData.map((card, index) => (
// //             <div
// //               key={card.id}
// //               className="card"
// //               style={{ '--index': index, '--color-card': card.color }}
// //             >
// //               <div className="card-content">
// //                 <h3>{card.title}</h3>
// //                 <p>{card.description}</p>
// //                 <a href={card.link} className="try-me-button">Try Me</a>
// //               </div>
// //               <div className="img" />
// //             </div>
// //           ))}
// //         </div>
// //       </div>

// //       <Controls>
// //         <ControlButton onClick={handlePrev}>
// //           <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
// //             <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
// //           </svg>
// //         </ControlButton>
// //         <ControlButton onClick={handleNext}>
// //           <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
// //             <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
// //           </svg>
// //         </ControlButton>
// //       </Controls>
// //     </CarouselWrapper>
// //   );
// // };

// // // --- Styled Components ---
// // const CarouselWrapper = styled.div`
// //   width: 100%;
// //   height: 650px;
// //   position: relative;
// //   display: flex;
// //   justify-content: center;
// //   align-items: center;

// //   .wrapper {
// //     width: 100%;
// //     height: 100%;
// //     position: relative;
// //     display: flex;
// //     align-items: center;
// //     justify-content: center;
// //   }

// //   .inner {
// //     --w: 220px;
// //     --h: 320px;
// //     --translateZ: 350px;
// //     --rotateX: -10deg;
// //     --perspective: 1500px;
// //     position: absolute;
// //     width: var(--w);
// //     height: var(--h);
// //     z-index: 2;
// //     transform-style: preserve-3d;
// //     transform: perspective(var(--perspective)) rotateX(var(--rotateX)) rotateY(var(--rotation-angle, 0deg));
// //     transition: transform 0.8s cubic-bezier(0.25, 1, 0.5, 1);
// //   }

// //   .card {
// //     position: absolute;
// //     border: 2px solid rgba(var(--color-card), 0.7);
// //     border-radius: 16px;
// //     overflow: hidden;
// //     inset: 0;
// //     transform: rotateY(calc((360deg / var(--quantity)) * var(--index))) translateZ(var(--translateZ));
// //     box-shadow: 0 0 20px rgba(var(--color-card), 0.2);
// //   }

// //   .img {
// //     position: absolute;
// //     inset: 0;
// //     z-index: -1;
// //     background: radial-gradient(circle at 50% 50%, rgba(var(--color-card), 0.25) 0%, rgba(15, 23, 42, 0.6) 80%);
// //   }

// //   .card-content {
// //     padding: 1.5rem;
// //     height: 100%;
// //     display: flex;
// //     flex-direction: column;
// //     justify-content: center;
// //     align-items: center;
// //     gap: 1rem;
// //     color: #FFF;
// //     text-shadow: 0 1px 3px #000;

// //     h3 {
// //       font-size: 1.75rem;
// //       font-weight: 700;
// //       color: rgba(var(--color-card), 1);
// //       margin: 0;
// //     }

// //     p {
// //       font-size: 1rem;
// //       color: #E2E8F0;
// //       line-height: 1.5;
// //       margin: 0;
// //     }

// //     .try-me-button {
// //       margin-top: 1rem;
// //       padding: 0.75rem 1.5rem;
// //       background-color: rgba(var(--color-card), 0.9);
// //       color: #0F172A;
// //       border: none;
// //       border-radius: 8px;
// //       font-weight: 600;
// //       cursor: pointer;
// //       text-decoration: none;
// //       transition: all 0.2s ease;

// //       &:hover {
// //         background-color: rgba(var(--color-card), 1);
// //         transform: scale(1.05);
// //         box-shadow: 0 0 25px rgba(var(--color-card), 0.5);
// //       }
// //     }
// //   }
// // `;

// // const Controls = styled.div`
// //   position: absolute;
// //   bottom: 60px;
// //   display: flex;
// //   gap: 1.5rem;
// //   z-index: 10;
// // `;

// // const ControlButton = styled.button`
// //   width: 50px;
// //   height: 50px;
// //   border-radius: 50%;
// //   background: rgba(255, 255, 255, 0.1);
// //   border: 1px solid rgba(255, 255, 255, 0.2);
// //   color: #FFF;
// //   cursor: pointer;
// //   display: flex;
// //   justify-content: center;
// //   align-items: center;
// //   transition: all 0.2s ease;

// //   &:hover {
// //     background: rgba(0, 255, 255, 0.2);
// //     border-color: #0ff;
// //   }

// //   svg {
// //     width: 24px;
// //     height: 24px;
// //   }
// // `;

// // export default CardCarousel;



import React, { useState } from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom'; // --- NEW: Import the Link component

// --- Card Data: Updated with the new evaluation link ---
const cardData = [
  {
    id: 1,
    title: 'Pitch Deck Analysis',
    description: 'Analyze your pitch deck with AI to get insights, clarity scores, and improvement suggestions.',
    color: '142, 249, 252',
    link: '/copilot', // CHANGED
  },
  {
    id: 2,
    title: 'Investors & Accelerators Recommendations',
    description: 'Get AI-driven suggestions for the most suitable investors and accelerators for your startup.',
    color: '142, 252, 157',
    link: '/investor-search', // CHANGED
  },
  {
    id: 3,
    title: 'Success Prediction',
    description: 'Predict your startup’s success probability using AI models trained on real-world startup data.',
    color: '252, 252, 142',
    link: '/predictor', // CHANGED
  },
  {
    id: 4,
    title: 'Startup Analysis Chatbot',
    description: 'Data-driven startup evaluations with an AI-powered RAG chatbot.',
    color: '252, 142, 142',
    link: 'https://foundersfuel.streamlit.app/', // CHANGED
  },
  {
    id: 5,
    title: '7 domain analysis',
    description: 'Analyzes 7 core domains + live web searching for startup evaluation.',
    color: '204, 142, 252',
    link: 'https://foundersfuel.streamlit.app/', // CHANGED
  },
];

// --- The Carousel Component (No changes needed here) ---
const CardCarousel = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const totalCards = cardData.length;

  const handlePrev = () => {
    setCurrentIndex(prevIndex => (prevIndex === 0 ? totalCards - 1 : prevIndex - 1));
  };

  const handleNext = () => {
    setCurrentIndex(prevIndex => (prevIndex === totalCards - 1 ? 0 : prevIndex + 1));
  };

  const rotationAngle = -currentIndex * (360 / totalCards);

  return (
    <CarouselWrapper>
      <div className="wrapper">
        <div
          className="inner"
          style={{
            '--quantity': totalCards,
            '--rotation-angle': `${rotationAngle}deg`,
          }}
        >
          {cardData.map((card, index) => (
            <div
              key={card.id}
              className="card"
              style={{ '--index': index, '--color-card': card.color }}
            >
              <div className="card-content">
                <h3>{card.title}</h3>
                <p>{card.description}</p>
                {/* --- CHANGED: Use Link component instead of <a> tag --- */}
                <Link to={card.link} className="try-me-button">
                  Try Me
                </Link>
              </div>
              <div className="img" />
            </div>
          ))}
        </div>
      </div>
      <Controls>
        <ControlButton onClick={handlePrev}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
          </svg>
        </ControlButton>
        <ControlButton onClick={handleNext}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
          </svg>
        </ControlButton>
      </Controls>
    </CarouselWrapper>
  );
};

// --- STYLES (No changes needed here) ---
const CarouselWrapper = styled.div`
  width: 100%;
  height: 650px;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;

  .wrapper {
    width: 100%;
    height: 100%;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .inner {
    /* --- CSS FIX #1: Adjusting card dimensions and carousel radius --- */
    --w: 250px; /* Increased card width to fit longer text */
    --h: 340px; /* Increased card height slightly */
    --translateZ: 480px; /* Increased radius to prevent wider cards from overlapping */
    --rotateX: -10deg;
    --perspective: 1500px;

    position: absolute;
    width: var(--w);
    height: var(--h);
    z-index: 2;
    transform-style: preserve-3d;
    transform: perspective(var(--perspective)) rotateX(var(--rotateX))
      rotateY(var(--rotation-angle, 0deg));
    transition: transform 0.8s cubic-bezier(0.25, 1, 0.5, 1);
  }

  .card {
    position: absolute;
    border: 2px solid rgba(var(--color-card), 0.7);
    border-radius: 16px;
    overflow: hidden;
    inset: 0;
    transform: rotateY(calc((360deg / var(--quantity)) * var(--index)))
      translateZ(var(--translateZ));
    box-shadow: 0 0 20px rgba(var(--color-card), 0.2);
  }

  .img {
    position: absolute;
    inset: 0;
    z-index: -1;
    background: radial-gradient(
      circle at 50% 50%,
      rgba(var(--color-card), 0.25) 0%,
      rgba(15, 23, 42, 0.6) 80%
    );
  }

  .card-content {
    padding: 1rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    color: #fff;
    text-shadow: 0 1px 3px #000;
    text-align: center; /* Added for better centering of multi-line text */

    h3 {
      /* --- CSS FIX #2: Reduced font size for longer titles --- */
      font-size: 1.5rem; /* Reduced from 1.75rem */
      font-weight: 750;
      color: rgba(var(--color-card), 1);
      margin: 0;
    }
    p {
      font-size: 1rem;
      color: #e2e8f0;
      line-height: 1.5;
      margin: 0;
    }
    .try-me-button {
      margin-top: 1rem;
      padding: 0.75rem 1.5rem;
      background-color: rgba(var(--color-card), 0.9);
      color: #0f172a;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      transition: all 0.2s ease;

      &:hover {
        background-color: rgba(var(--color-card), 1);
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(var(--color-card), 0.5);
      }
    }
  }
`;

const Controls = styled.div`
  position: absolute;
  bottom: 60px;
  display: flex;
  gap: 1.5rem;
  z-index: 10;
`;

const ControlButton = styled.button`
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: #fff;
  cursor: pointer;
  display: flex;
  justify-content: center;
  align-items: center;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(0, 255, 255, 0.2);
    border-color: #0ff;
  }

  svg {
    width: 24px;
    height: 24px;
  }
`;

export default CardCarousel;