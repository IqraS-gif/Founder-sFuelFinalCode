// // // // // // src/App.jsx (NEW ROUTER VERSION)

// // // // // import React from 'react';
// // // // // import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// // // // // import styled, { createGlobalStyle } from 'styled-components'; 

// // // // // // --- Global Components & Styles ---
// // // // // import Navbar from './components/Navbar'; // Your existing Navbar


// // // // // // --- Page Components ---
// // // // // import LandingPage from './pages/LandingPage'; // The component we just created
// // // // // import CoPilotPage from './pages/CoPilotPage';
// // // // // import ResultsPage from './pages/ResultsPage';

// // // // // // This GlobalStyle applies to ALL pages, which is what we want.
// // // // // const GlobalStyle = createGlobalStyle`
// // // // //   body::-webkit-scrollbar { display: none; }
// // // // //   html { scrollbar-width: none; }
// // // // // `;



// // // // // function App() {
// // // // //   return (
// // // // //     <Router>
// // // // //       {/* These components are OUTSIDE <Routes>, so they appear on every page */}
// // // // //       <GlobalStyle />
   
// // // // //       <Navbar />

// // // // //       {/* The <Routes> component will switch between your pages */}
// // // // //       <Routes>
// // // // //         <Route path="/" element={<LandingPage />} />
// // // // //         <Route path="/copilot" element={<CoPilotPage />} />
// // // // //         <Route path="/results/:taskId" element={<ResultsPage />} />
// // // // //       </Routes>
// // // // //     </Router>
// // // // //   );
// // // // // }

// // // // // export default App;

// // // // // src/App.jsx (UPDATED ROUTER VERSION)

// // // // import React from 'react';
// // // // import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// // // // import styled, { createGlobalStyle } from 'styled-components'; 

// // // // // --- Global Components & Styles ---
// // // // import Navbar from './components/Navbar'; // Your existing Navbar

// // // // // --- Page Components ---
// // // // import LandingPage from './pages/LandingPage';
// // // // import CoPilotPage from './pages/CoPilotPage';
// // // // import ResultsPage from './pages/ResultsPage';

// // // // // --- NEW: Import the pages for the investor matching feature ---
// // // // import InvestorSearchPage from './pages/InvestorSearchPage';
// // // // import InvestorResultsPage from './pages/InvestorResultsPage';

// // // // // This GlobalStyle applies to ALL pages, which is what we want.
// // // // const GlobalStyle = createGlobalStyle`
// // // //   body::-webkit-scrollbar { display: none; }
// // // //   html { scrollbar-width: none; }
// // // // `;

// // // // function App() {
// // // //   return (
// // // //     <Router>
// // // //       {/* These components are OUTSIDE <Routes>, so they appear on every page */}
// // // //       <GlobalStyle />
// // // //       <Navbar />

// // // //       {/* The <Routes> component will switch between your pages */}
// // // //       <Routes>
// // // //         {/* --- Your existing routes remain unchanged --- */}
// // // //         <Route path="/" element={<LandingPage />} />
// // // //         <Route path="/copilot" element={<CoPilotPage />} />
// // // //         <Route path="/results/:taskId" element={<ResultsPage />} />
        
// // // //         {/* --- NEW: Add routes for the investor matching feature --- */}
// // // //         <Route path="/investor-search" element={<InvestorSearchPage />} />
// // // //         <Route path="/investor-results" element={<InvestorResultsPage />} />
// // // //       </Routes>
// // // //     </Router>
// // // //   );
// // // // }

// // // // export default App;

// // // // src/App.jsx (MODIFIED ROUTER VERSION)

// // // import React from 'react';
// // // import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// // // import styled, { createGlobalStyle } from 'styled-components'; 

// // // // --- Global Components & Styles ---
// // // import Navbar from './components/Navbar'; // Your existing Navbar

// // // // --- Page Components ---
// // // import LandingPage from './pages/LandingPage';
// // // import CoPilotPage from './pages/CoPilotPage';
// // // import ResultsPage from './pages/ResultsPage';

// // // // --- EXISTING: Investor Matching Pages ---
// // // import InvestorSearchPage from './pages/InvestorSearchPage';
// // // import InvestorResultsPage from './pages/InvestorResultsPage';

// // // // --- NEW: Evaluation Pages ---
// // // import EvaluationInputPage from './pages/EvaluationInputPage'; // <--- NEW INPUT PAGE
// // // import EvaluationResultsPage from './pages/EvaluationResultsPage'; // <--- NEW RESULTS PAGE

// // // // This GlobalStyle applies to ALL pages, which is what we want.
// // // const GlobalStyle = createGlobalStyle`
// // //     body::-webkit-scrollbar { display: none; }
// // //     html { scrollbar-width: none; }
// // // `;

// // // function App() {
// // //     return (
// // //         <Router>
// // //             {/* These components are OUTSIDE <Routes>, so they appear on every page */}
// // //             <GlobalStyle />
// // //             <Navbar />

// // //             {/* The <Routes> component will switch between your pages */}
// // //             <Routes>
// // //                 {/* --- Existing Analysis/Landing Routes --- */}
// // //                 <Route path="/" element={<LandingPage />} />
// // //                 <Route path="/copilot" element={<CoPilotPage />} />
// // //                 <Route path="/results/:taskId" element={<ResultsPage />} />
                
// // //                 {/* --- Existing Investor Matching Routes --- */}
// // //                 <Route path="/investor-search" element={<InvestorSearchPage />} />
// // //                 <Route path="/investor-results" element={<InvestorResultsPage />} />

               
// // //             </Routes>
// // //         </Router>
// // //     );
// // // }

// // // export default App;



// import React from 'react';
// import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// import styled, { createGlobalStyle } from 'styled-components';

// // --- Global Components & Styles ---
// import Navbar from './components/Navbar'; // Your existing Navbar

// // --- Page Components ---
// import LandingPage from './pages/LandingPage';
// import CoPilotPage from './pages/CoPilotPage';
// import ResultsPage from './pages/ResultsPage';
// import InvestorSearchPage from './pages/InvestorSearchPage';
// import InvestorResultsPage from './pages/InvestorResultsPage';


// // --- NEW: Import the page for the 7-domain evaluation feature ---
// import EvaluationPage from './pages/EvaluationPage';


// // This GlobalStyle applies to ALL pages, which is what we want.
// const GlobalStyle = createGlobalStyle`
//   body::-webkit-scrollbar { display: none; }
//   html { scrollbar-width: none; }
// `;

// function App() {
//   return (
//     <Router>
//       {/* These components are OUTSIDE <Routes>, so they appear on every page */}
//       <GlobalStyle />
//       <Navbar />

//       {/* The <Routes> component will switch between your pages */}
//       <Routes>
//         {/* --- Your existing routes remain unchanged --- */}
//         <Route path="/" element={<LandingPage />} />
//         <Route path="/copilot" element={<CoPilotPage />} />
//         <Route path="/results/:taskId" element={<ResultsPage />} />
        
//         {/* --- Routes for the investor matching feature --- */}
//         <Route path="/investor-search" element={<InvestorSearchPage />} />
//         <Route path="/investor-results" element={<InvestorResultsPage />} />

//         {/* --- NEW: Add the route for the 7-domain evaluation feature --- */}
//         <Route path="/evaluate" element={<EvaluationPage />} />
         
//       </Routes>
//     </Router>
//   );
// }

// export default App;



import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import styled, { createGlobalStyle } from 'styled-components';

// --- Global Components & Styles ---
import Navbar from './components/Navbar'; 

// --- Page Components ---
import LandingPage from './pages/LandingPage';
import CoPilotPage from './pages/CoPilotPage';
import ResultsPage from './pages/ResultsPage';
import InvestorSearchPage from './pages/InvestorSearchPage';
import InvestorResultsPage from './pages/InvestorResultsPage';
import EvaluationPage from './pages/EvaluationPage';

// --- NEW: Import the Success Predictor Page ---
import SuccessPredictorPage from './pages/SuccessPredictorPage'; 


// This GlobalStyle applies to ALL pages, which is what we want.
const GlobalStyle = createGlobalStyle`
  body::-webkit-scrollbar { display: none; }
  html { scrollbar-width: none; }
`;

function App() {
  return (
    <Router>
      {/* These components are OUTSIDE <Routes>, so they appear on every page */}
      <GlobalStyle />
      <Navbar />

      {/* The <Routes> component will switch between your pages */}
      <Routes>
        {/* --- Existing Analysis/Landing Routes --- */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/copilot" element={<CoPilotPage />} />
        <Route path="/results/:taskId" element={<ResultsPage />} />
        
        {/* --- Investor Matching Routes --- */}
        <Route path="/investor-search" element={<InvestorSearchPage />} />
        <Route path="/investor-results" element={<InvestorResultsPage />} />

        {/* --- 7-Domain Evaluation Route --- */}
        <Route path="/evaluate" element={<EvaluationPage />} />
         
        {/* --- NEW: Success Predictor Route --- */}
        <Route path="/predictor" element={<SuccessPredictorPage />} />

      </Routes>
    </Router>
  );
}

export default App;