// src/pages/ResultsPage.jsx
// CORRECTED AND FINAL VERSION

import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import styled, { keyframes } from 'styled-components';
import { getTaskStatus, getAnalysisResults } from '../api/apiClient'; // Make sure getAnalysisResults is imported
import OverviewTab from '../components/analysis/OverviewTab';
import SlideBySlideTab from '../components/analysis/SlideBySlideTab';
import TechnicalTab from '../components/analysis/TechnicalTab';
import AiInsightsTab from '../components/analysis/AiInsightsTab';
import HistoryTab from '../components/analysis/HistoryTab';



// --- Styled Components ---
const PageContainer = styled.div`
  padding: 100px 2rem 2rem 2rem;
  box-sizing: border-box;
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
`;

const StatusBox = styled.div`
  text-align: center;
  padding: 3rem;
  border: 1px solid rgba(0, 255, 255, 0.2);
  border-radius: 12px;
  background-color: rgba(15, 23, 42, 0.8);
  backdrop-filter: blur(10px);
  max-width: 600px;
  
  h2 { color : rgba(2, 255, 255, 1) ; font-size: 2.5rem; margin-bottom: 1rem; text-shadow: 0 0 19px rgba(14, 171, 255, 1); }
  p { font-size: 1.2rem; color: #94a3b8; min-height: 30px; }
`;

const ErrorMessage = styled.p`
  color: #ff6b6b;
  font-weight: 500;
`;

const spin = keyframes` to { transform: rotate(360deg); }`;
const Spinner = styled.div`
  border: 4px solid rgba(0, 255, 255, 0.2);
  border-top: 4px solid #00f2fe;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: ${spin} 1s linear infinite;
  margin: 2rem auto;
`;

const ResultsHeader = styled.header`
  margin-bottom: 2rem;
  h1 { font-size: 2.2rem; color: #fff; }
  p { color: #94a3b8; }
`;

const TabContainer = styled.div`
  display: flex;
  border-bottom: 1px solid #334155;
  margin-bottom: 2rem;
`;

const TabButton = styled.button`
  padding: 1rem 1.5rem;
  border: none;
  background: none;
  color: ${({ active }) => (active ? '#14f1d9' : '#94a3b8')};
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  border-bottom: 3px solid ${({ active }) => (active ? '#14f1d9' : 'transparent')};
  transition: all 0.2s ease-in-out;
  &:hover { color: #14f1d9; }
`;

const TabContent = styled.div`
  /* Content for the active tab will be rendered here */
`;


function ResultsPage() {
    const { taskId } = useParams();
    
    // --- State Management ---
    const [message, setMessage] = useState('Initiating analysis...');
    const [error, setError] = useState('');
    
    // *** BUG FIX: THIS LINE WAS MISSING ***
    const [analysisData, setAnalysisData] = useState(null); 
    
    const [activeTab, setActiveTab] = useState('overview');

    // --- Data Fetching Effect ---
    useEffect(() => {
        const pollStatus = async () => {
            try {
                const response = await getTaskStatus(taskId);
                const { status: newStatus, message: newMessage, analysis_id: newAnalysisId } = response.data;
                
                if (newStatus === 'complete') {
                    clearInterval(intervalId);
                    setMessage('Analysis Complete! Fetching results...');
                    
                    const resultsResponse = await getAnalysisResults(newAnalysisId);
                    setAnalysisData(resultsResponse.data);

                } else if (newStatus === 'failed') {
                    clearInterval(intervalId);
                    setError(newMessage || 'Analysis failed. Please try again.');
                } else {
                    setMessage(newMessage);
                }
            } catch (err) {
                clearInterval(intervalId);
                setError('Could not get status. Is the backend running?');
            }
        };

        const intervalId = setInterval(pollStatus, 3000);
        pollStatus();
        return () => clearInterval(intervalId);
    }, [taskId]);


    // --- Render Logic ---

    // 1. While analysisData is null, show the loading/error screen
    if (!analysisData) {
        return (
            <PageContainer style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <StatusBox>
                    {error ? (
                        <><h2>Error</h2><ErrorMessage>{error}</ErrorMessage></>
                    ) : (
                        <>
                          <h2>🧠 Analyzing Your Deck...</h2>
                          <Spinner />
                          <p>{message}</p>
                        </>
                    )}
                </StatusBox>
            </PageContainer>
        );
    }
    
    // 2. Once data is fetched, render the dashboard
    const TABS = {
     'overview': <OverviewTab data={analysisData} />,
        'slides': <SlideBySlideTab data={analysisData} />,
        'technical': <TechnicalTab data={analysisData} />,
        'ai_insights': <AiInsightsTab data={analysisData} />,
        'history': <HistoryTab data={analysisData} />,
    };

    return (
        <PageContainer>
            <ResultsHeader>
                <h1>Analysis Results</h1>
                <p>Analyzed <span style={{ color: '#14f1d9', fontWeight: 'bold' }}>{analysisData.filename}</span> as a <span style={{ color: '#14f1d9', fontWeight: 'bold' }}>{analysisData.analyzed_as_persona}</span></p>
            </ResultsHeader>

            <TabContainer>
                {Object.keys(TABS).map(tabName => (
                    <TabButton 
                        key={tabName} 
                        active={activeTab === tabName}
                        onClick={() => setActiveTab(tabName)}
                    >
                        {tabName.replace('_', ' ').toUpperCase()}
                    </TabButton>
                ))}
            </TabContainer>

            <TabContent>
                {TABS[activeTab]}
            </TabContent>
        </PageContainer>
    );
}

export default ResultsPage;


