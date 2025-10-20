import React, { useState } from 'react';
import styled from 'styled-components';
import { evaluateStartup } from '../api/apiClient'; // Make sure this path is correct

// --- Styled Components for a better UI ---
const PageContainer = styled.div`
  padding: 2rem 4rem;
  max-width: 900px;
  margin: 2rem auto;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
`;

const EvaluationForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  background: #f9f9f9;
  padding: 2rem;
  border-radius: 12px;
  border: 1px solid #e0e0e0;
`;

const FormInput = styled.input`
  padding: 0.8rem 1rem;
  border-radius: 8px;
  border: 1px solid #ccc;
  font-size: 1rem;
`;

const FormTextarea = styled.textarea`
  padding: 0.8rem 1rem;
  border-radius: 8px;
  border: 1px solid #ccc;
  font-size: 1rem;
  min-height: 120px;
  resize: vertical;
`;

const SubmitButton = styled.button`
  padding: 1rem;
  font-size: 1.1rem;
  font-weight: bold;
  color: white;
  background-color: #007bff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background-color: #0056b3;
  }

  &:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }
`;

const ErrorMessage = styled.p`
  color: #d9534f;
  background-color: #f2dede;
  padding: 1rem;
  border: 1px solid #ebccd1;
  border-radius: 8px;
`;

const ResultsContainer = styled.div`
  margin-top: 2rem;
`;

const Section = styled.div`
  margin-bottom: 1.5rem;
  padding: 1.5rem;
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
`;

const SectionTitle = styled.h3`
  margin-top: 0;
  border-bottom: 2px solid #007bff;
  padding-bottom: 0.5rem;
`;

const EvaluationPage = () => {
  const [startupName, setStartupName] = useState('');
  const [startupIdea, setStartupIdea] = useState('');
  const [evaluation, setEvaluation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setEvaluation(null);

    // Basic frontend validation to match the backend
    if (startupIdea.trim().length < 20) {
      setError('Please provide a more detailed startup idea (at least 20 characters).');
      setIsLoading(false);
      return;
    }

    try {
      // The keys here (`startup_idea`, `startup_name`) MUST match the backend Pydantic model
      const payload = {
        startup_idea: startupIdea,
        startup_name: startupName,
      };
      
      const response = await evaluateStartup(payload);
      setEvaluation(response.data.evaluation);
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 'An unexpected error occurred while evaluating the startup.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <PageContainer>
      <h1>7-Domain Startup Evaluation</h1>
      <p>Get a comprehensive, AI-powered analysis of your startup idea, enhanced with real-time market data.</p>
      
      <EvaluationForm onSubmit={handleSubmit}>
        <FormInput
          type="text"
          value={startupName}
          onChange={(e) => setStartupName(e.target.value)}
          placeholder="Startup Name (Optional)"
        />
        <FormTextarea
          value={startupIdea}
          onChange={(e) => setStartupIdea(e.target.value)}
          placeholder="Describe your startup idea in detail. What problem do you solve? Who is your target market?"
          required
        />
        <SubmitButton type="submit" disabled={isLoading}>
          {isLoading ? 'Analyzing...' : 'Generate Evaluation'}
        </SubmitButton>
      </EvaluationForm>
      
      {error && <ErrorMessage>{error}</ErrorMessage>}

      {isLoading && <p>Please wait, gathering intelligence and generating your report...</p>}

      {evaluation && (
        <ResultsContainer>
          <h2>Evaluation Report for {startupName || 'Your Startup'}</h2>
          {Object.entries(evaluation).map(([key, value]) => (
            value && (
              <Section key={key}>
                <SectionTitle>{key.replace(/_/g, ' ').toUpperCase()}</SectionTitle>
                <div style={{ whiteSpace: 'pre-wrap' }}>{value}</div>
              </Section>
            )
          ))}
        </ResultsContainer>
      )}
    </PageContainer>
  );
};

export default EvaluationPage;


