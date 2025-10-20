// src/pages/CoPilotPage.jsx
// NEW AND IMPROVED VERSION

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { startAnalysis } from '../api/apiClient';

// --- Styled Components (Matching Your New Design) ---

const PageContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120px 2rem 2rem 2rem;
  box-sizing: border-box;
  width: 100%;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 2.5rem;
  h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #fff;
    span {
      color: #14f1d9; /* Teal accent color */
      text-decoration: underline;
    }
  }
  p {
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 600px;
  }
`;

const MainBox = styled.div`
  width: 100%;
  max-width: 700px;
  padding: 2rem;
  border: 1px solid #14f1d940;
  border-radius: 16px;
  background-color: #1E293B90;
  backdrop-filter: blur(10px);
  box-shadow: 0 0 30px #14f1d920;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
`;

const UploaderBox = styled.div`
  /* Uploader styles */
  text-align: center;
  padding: 2rem;
  border: 2px dashed #334155;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  
  &:hover {
    border-color: #14f1d9;
    background-color: #33415550;
  }

  h2 { margin-top: 1rem; margin-bottom: 0.5rem; color: #fff; }
  p { margin: 0; color: #94a3b8; }
`;

const UploadIcon = styled.div`
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background-color: #14f1d930;
  color: #14f1d9;
  display: grid;
  place-items: center;
  font-size: 2rem;
  margin: 0 auto;
`;

const FileTypes = styled.div`
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 1.5rem;
  span {
    padding: 0.3rem 0.8rem;
    background-color: #334155;
    border-radius: 6px;
    font-size: 0.8rem;
  }
`;

const UploadedFileCard = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  background-color: #14f1d920;
  padding: 1rem;
  border-radius: 12px;
  border-left: 4px solid #14f1d9;
  
  .icon { font-size: 1.5rem; color: #14f1d9; }
  .details { flex-grow: 1; }
  .filename { font-weight: 600; color: #fff; }
  .filesize { font-size: 0.9rem; color: #94a3b8; }
  .remove-btn {
    background: none; border: none; color: #94a3b8;
    font-size: 1.5rem; cursor: pointer;
    &:hover { color: #ff6b6b; }
  }
`;

const ConfigBox = styled(MainBox)`
  margin-top: 0;
  h3 {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0 0 1.5rem 0;
    font-size: 1.2rem;
    color: #fff;
  }
`;

const ConfigGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  label { font-weight: 500; color: #cbd5e1; margin-bottom: 0.5rem; display: block; }
  select {
    width: 100%;
    padding: 0.8rem;
    border-radius: 8px;
    border: 1px solid #334155;
    background-color: #0F172A;
    color: #e2e8f0;
    font-size: 1rem;
    &:focus { outline: 2px solid #14f1d9; }
  }
`;

const AnalyzeButton = styled.button`
  width: 100%;
  max-width: 700px;
  padding: 1rem 1.5rem;
  font-size: 1.1rem;
  font-weight: 600;
  color: #020617;
  background: #14f1d9;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  box-shadow: 0 0 20px #14f1d950;
  &:hover { transform: translateY(-2px); box-shadow: 0 0 30px #14f1d980; }
  &:disabled { /* ... styles for disabled ... */ }
`;

// --- Configuration Data and Component Logic ---

const CATEGORIES = ["Fintech", "Healthcare", "SaaS", "AI/ML", "E-commerce", "Deep Tech", "Consumer", "Enterprise"];

// *** BUG FIX: The VC_PERSONAS object is now correctly populated ***
const VC_PERSONAS = {
    "The Friendly Mentor": "a friendly, encouraging mentor who focuses on nurturing potential",
    "The Data-Driven Analyst": "a sharp, data-driven analyst who demands concrete metrics and evidence",
    "The 'Big Picture' Strategist": "a 'big picture' strategist who evaluates market timing and scalability",
    "The Devil's Advocate": "a devil's advocate who challenges assumptions and identifies risks"
};

function CoPilotPage() {
    const navigate = useNavigate();
    const [file, setFile] = useState(null);
    const [category, setCategory] = useState(CATEGORIES[0]);
    const [persona, setPersona] = useState(Object.keys(VC_PERSONAS)[0]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    
    // Hidden file input ref
    const fileInputRef = React.useRef(null);

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
        }
    };
    
    const handleSubmit = async () => {
        if (!file) return;
        setIsLoading(true);
        setError('');
        try {
            const response = await startAnalysis(file, category, persona);
            navigate(`/results/${response.data.task_id}`);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to start analysis.');
            setIsLoading(false);
        }
    };

    return (
        <PageContainer>
            <Header>
                <h1>VC Pitch Deck <span>Co-Pilot</span></h1>
                <p>Get comprehensive AI-powered analysis of your pitch deck from different VC perspectives</p>
            </Header>

            {/* Hidden file input */}
            <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".pdf,.ppt,.pptx" style={{ display: 'none' }} />

            {/* --- CONDITIONAL RENDERING --- */}
            
            {!file ? (
                // --- VIEW 1: BEFORE UPLOAD ---
                <MainBox onClick={() => fileInputRef.current.click()}>
                    <UploaderBox>
                        <UploadIcon>⬆️</UploadIcon>
                        <h2>Upload Your Pitch Deck</h2>
                        <p>Drag and drop your presentation file or click to browse</p>
                        <FileTypes>
                            <span>📄 PDF</span>
                            <span>📊 PowerPoint</span>
                            <span>💻 PPTX</span>
                        </FileTypes>
                    </UploaderBox>
                </MainBox>
            ) : (
                // --- VIEW 2: AFTER UPLOAD ---
                <>
                    <MainBox>
                        <UploadedFileCard>
                            <div className="icon">📄</div>
                            <div className="details">
                                <div className="filename">{file.name}</div>
                                <div className="filesize">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                            </div>
                            <button className="remove-btn" onClick={() => setFile(null)}>×</button>
                        </UploadedFileCard>
                    </MainBox>
                    <ConfigBox>
                        <h3>⚙️ Analysis Configuration</h3>
                        <ConfigGrid>
                            <div>
                                <label htmlFor="category-select">Business Category</label>
                                <select id="category-select" value={category} onChange={(e) => setCategory(e.target.value)}>
                                    {CATEGORIES.map(cat => <option key={cat} value={cat}>{cat}</option>)}
                                </select>
                            </div>
                            <div>
                                <label htmlFor="persona-select">VC Evaluation Style</label>
                                <select id="persona-select" value={persona} onChange={(e) => setPersona(e.target.value)}>
                                    {Object.keys(VC_PERSONAS).map(p => <option key={p} value={p}>{p}</option>)}
                                </select>
                            </div>
                        </ConfigGrid>
                    </ConfigBox>
                    <AnalyzeButton onClick={handleSubmit} disabled={isLoading}>
                        {isLoading ? 'Processing...' : 'Analyze Pitch Deck'}
                    </AnalyzeButton>
                </>
            )}

        </PageContainer>
    );
}

export default CoPilotPage;