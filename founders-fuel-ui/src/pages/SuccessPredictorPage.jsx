// FoundersFuelFrontend/founders-fuel-ui/src/pages/SuccessPredictorPage.jsx

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styled, { createGlobalStyle } from 'styled-components';
import { predictSuccess, explainSuccess } from '../api/apiClient';

// --- UPDATED GLOBAL STYLES (MATCHES SuccessForm.css) ---
const SuccessPredictorGlobalStyle = createGlobalStyle`
  /* Note: Added the body and root selector to ensure the dark background covers the whole page */
  body, #root { 
    background-color: #0B1120; /* Deep dark background */
    min-height: 100vh;
  }
    
  .page-wrapper {
    background-color: #0B1120; /* Deep dark background */
    min-height: 100vh;
    padding: 40px;
    color: #f0f0f0;
    font-family: 'Segoe UI', sans-serif;
  }

  .header {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  .header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #ffffff;
  }

  .header h1 span {
    color: #41f1d9; /* Teal accent */
    text-decoration: underline;
  }

  .header p {
    font-size: 1.1rem;
    color: #cccccc;
    max-width: 600px;
    margin: 0 auto;
  }

  .form-container {
    max-width: 600px;
    margin: auto;
    padding: 30px;
    background-color: #131B2A; 
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    color: #E0E7FF;
    transition: all 0.3s ease;
    border: 2px dashed #41f1d9; /* Teal border */
    border-radius: 15px;
  }

  .form-container label {
    display: block;
    margin-top: 15px;
    font-weight: 600;
    color: #ddd;
  }

  .form-container input[type="range"],
  .form-container textarea {
    width: 100%;
    margin-top: 5px;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid #555;
    background-color: #131B2A;
    color: #f0f0f0;
    /* Special styling for range input thumb */
    -webkit-appearance: none;
    height: 6px;
  }
  
  .form-container input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: #41f1d9;
    cursor: pointer;
    margin-top: -4px; /* Fix vertical alignment */
  }


  .form-container textarea {
    resize: vertical;
    min-height: 80px;
  }

  .form-container button {
    margin-top: 20px;
    padding: 12px 24px;
    background-color: #41f1d9;
    color: #1a1a1a;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: bold;
    transition: background-color 0.3s ease;
  }

  .form-container button:hover {
    background-color: #2dd6c3;
  }

  .result-box {
    margin-top: 20px;
    padding: 15px;
    border-radius: 8px;
    font-weight: bold;
    font-size: 1.2em; /* Added for readability */
    text-align: center;
  }
    /* Add these rules inside the SuccessPredictorGlobalStyle block */
  .narrative ul {
    list-style: none; /* Remove default bullets */
    padding-left: 0;
    margin-top: 10px;
  }

  .narrative li {
    margin-bottom: 10px;
    padding-left: 20px;
    position: relative;
    font-size: 1rem; /* Added for clarity */
    line-height: 1.4;
  }

  .result-box.success {
    background-color: #e6f4ea;
    color: #2e7d32;
  }

  .result-box.fail {
    background-color: #fdecea;
    color: #c62828;
  }

  .narrative {
    margin-top: 30px;
    background-color: #131B2A;
    padding: 20px;
    border-left: 4px solid #41f1d9;
    border-radius: 8px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #e0e0e0;
  }
`;

function SuccessPredictorPage() {
  const [formData, setFormData] = useState({
    goalINR: 5000,
    backers: 100,
    usdPledgedINR: 4500,
    pitchText: 'A smart watch for kids'
  });

  const [usdRate, setUsdRate] = useState(0);
  const [result, setResult] = useState(null);
  const [summary, setSummary] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Fetch live exchange rate on component mount
  useEffect(() => {
    // Note: Using a hardcoded key from the open-source project
    axios.get('https://v6.exchangerate-api.com/v6/5c26cf14ce2a6e87b5a86aa1/latest/INR')
      .then(res => setUsdRate(res.data.conversion_rates.USD))
      .catch(err => {
        console.error('Exchange rate error, defaulting to 0.012:', err);
        setUsdRate(0.012); // Default approximate rate if API fails
      });
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSlider = (e) => {
    setFormData({ ...formData, [e.target.name]: Number(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResult(null);
    setSummary('');

    // Convert INR to USD using the fetched rate for the backend payload
    const payload = {
      // Keys MUST match the Python SuccessInput model
      goal: formData.goalINR * usdRate,
      backers: formData.backers,
      usd_pledged: formData.usdPledgedINR * usdRate,
      text: formData.pitchText
    };

    try {
      // 1. Predict using the new API client function
      const res = await predictSuccess(payload);
      const successProbability = res.data.success_probability;
      setResult(successProbability);

      // 2. Explain using the new API client function
      const explainRes = await explainSuccess(payload);
      setSummary(explainRes.data.narrative);

    } catch (err) {
      console.error('Prediction error:', err.response?.data?.detail || err.message);
      setSummary(`Error: ${err.response?.data?.detail || 'An unexpected error occurred during prediction.'}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <SuccessPredictorGlobalStyle /> {/* Apply the styles */}
      {/* Note: The main content container needs to be wrapped, matching the original CSS selector */}
      <div className="page-wrapper">
        <div className="header">
          <h1>
            Startup <span>Success Predictor</span>
          </h1>
          <p>
            Estimate your crowdfunding success based on pitch quality, funding goals, and backer momentum.
          </p>
        </div>

        <div className="form-container">
          <form onSubmit={handleSubmit}>
            <label>Goal Amount (INR): {formData.goalINR.toLocaleString()}</label>
            <input type="range" name="goalINR" min="100" max="1000000" step="100" value={formData.goalINR} onChange={handleSlider} />

            <label>Number of Backers: {formData.backers}</label>
            <input type="range" name="backers" min="0" max="5000" step="10" value={formData.backers} onChange={handleSlider} />

            <label>Pledged Amount (INR): {formData.usdPledgedINR.toLocaleString()}</label>
            <input type="range" name="usdPledgedINR" min="0" max="1000000" step="100" value={formData.usdPledgedINR} onChange={handleSlider} />

            <label>Pitch Text:</label>
            <textarea name="pitchText" value={formData.pitchText} onChange={handleChange} required />

            <button type="submit" disabled={isLoading}>
              {isLoading ? 'Predicting...' : 'Predict Success Probability'}
            </button>
          </form>

          {result !== null && (
            <div className={`result-box ${result < 30 ? 'fail' : 'success'}`}>
              Estimated Success Probability: {result}%
            </div>
          )}

          {summary && (
            <div className="narrative" dangerouslySetInnerHTML={{ __html: summary }} />
          )}

          <p style={{ marginTop: '40px', fontStyle: 'italic', textAlign: 'center', color: '#888' }}>
            Live INR to USD rate: ₹1 = ${usdRate.toFixed(6)} 
          </p>
        </div>
      </div>
    </>
  );
}

export default SuccessPredictorPage;