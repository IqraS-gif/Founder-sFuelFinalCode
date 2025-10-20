// // src/components/analysis/HistoryTab.jsx

// import React, { useState, useEffect } from 'react';
// import styled from 'styled-components';
// import { getHistory } from '../../api/apiClient';
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// // --- Styled Components ---
// const Card = styled.div`
//   background-color: #1E293B;
//   border: 1px solid #334155;
//   border-radius: 12px;
//   padding: 1.5rem;
//   margin-bottom: 1.5rem;
// `;

// const InfoBox = styled(Card)`
//   text-align: center;
//   color: #94a3b8;
// `;

// const Table = styled.table`
//   width: 100%;
//   border-collapse: collapse;
//   th, td {
//     padding: 1rem;
//     text-align: left;
//     border-bottom: 1px solid #334155;
//   }
//   th {
//     color: #94a3b8;
//     font-size: 0.9rem;
//     text-transform: uppercase;
//   }
//   td {
//     color: #e2e8f0;
//   }
// `;

// // --- The Component ---
// function HistoryTab({ data }) {
//   const [historyData, setHistoryData] = useState(null);
//   const [error, setError] = useState('');

//   useEffect(() => {
//     const fetchHistory = async () => {
//       try {
//         const response = await getHistory(data.filename);
//         // Sort data from oldest to newest for the chart
//         const sortedData = response.data.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
//         setHistoryData(sortedData);
//       } catch (err) {
//         setError('No previous analysis history found for this file.');
//       }
//     };
//     fetchHistory();
//   }, [data.filename]);

//   if (error) {
//     return <InfoBox>{error}</InfoBox>;
//   }

//   if (!historyData) {
//     return <InfoBox>Loading history...</InfoBox>;
//   }
  
//   if (historyData.length < 2) {
//       return <InfoBox>Analyze this deck again to see improvement trends over time.</InfoBox>
//   }

//   // Format data for the chart
//   const chartData = historyData.map(item => ({
//     date: new Date(item.timestamp).toLocaleDateString(),
//     'Readiness': item.investment_readiness_score,
//     'Moat': item.competitive_moat_score,
//     'GTM': item.gtm_strategy_score,
//   }));

//   return (
//     <div>
//       <Card>
//         <h3>📈 Improvement Trend Over Time</h3>
//         <ResponsiveContainer width="100%" height={400}>
//           <LineChart data={chartData}>
//             <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
//             <XAxis dataKey="date" stroke="#94a3b8" />
//             <YAxis domain={[0, 10]} stroke="#94a3b8" />
//             <Tooltip contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #334155' }} />
//             <Legend />
//             <Line type="monotone" dataKey="Readiness" stroke="#14f1d9" strokeWidth={2} />
//             <Line type="monotone" dataKey="Moat" stroke="#818cf8" strokeWidth={2} />
//             <Line type="monotone" dataKey="GTM" stroke="#f472b6" strokeWidth={2} />
//           </LineChart>
//         </ResponsiveContainer>
//       </Card>
//       <Card>
//         <h3>📋 Analysis History</h3>
//         <Table>
//           <thead>
//             <tr>
//               <th>Date</th>
//               <th>Persona</th>
//               <th>Readiness Score</th>
//               <th>Moat Score</th>
//               <th>GTM Score</th>
//             </tr>
//           </thead>
//           <tbody>
//             {[...historyData].reverse().map(item => ( // Show newest first in table
//               <tr key={item.id}>
//                 <td>{new Date(item.timestamp).toLocaleString()}</td>
//                 <td>{item.persona}</td>
//                 <td>{item.investment_readiness_score.toFixed(1)} / 10</td>
//                 <td>{item.competitive_moat_score} / 10</td>
//                 <td>{item.gtm_strategy_score} / 10</td>
//               </tr>
//             ))}
//           </tbody>
//         </Table>
//       </Card>
//     </div>
//   );
// }

// export default HistoryTab;


// src/components/analysis/HistoryTab.jsx

import React, { useState, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';
import { getHistory } from '../../api/apiClient';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Loader, AlertTriangle, TrendingUp, Clipboard } from 'lucide-react';

// --- Keyframes for Animations ---
const fadeInUp = keyframes`
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
`;

const spin = keyframes`
  to { transform: rotate(360deg); }
`;

// --- Styled Components ---

const HistoryContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  padding: 1rem;
`;

const Card = styled.div`
  background: rgba(45, 55, 72, 0.65);
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 16px;
  padding: 1.5rem 2rem;
  backdrop-filter: blur(14px) saturate(180%);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  animation: ${fadeInUp} 0.6s cubic-bezier(0.25, 0.1, 0.25, 1) forwards;
  animation-delay: ${({ delay }) => delay || '0s'};
  opacity: 0;
`;

const CardHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: #f8fafc;
  margin-bottom: 1.5rem;

  h3 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
  }
  
  svg {
    color: #94a3b8;
  }
`;

const InfoBox = styled(Card)`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #94a3b8;
  min-height: 200px;
  gap: 1rem;
  font-size: 1.1rem;

  svg.spinner {
    animation: ${spin} 1.5s linear infinite;
  }
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;

  th, td {
    padding: 1rem 1.25rem;
    text-align: left;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
  }

  th {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  td {
    color: #e2e8f0;
    font-size: 0.95rem;
  }

  tbody tr {
    transition: background-color 0.2s ease;
    &:hover {
      background-color: rgba(255, 255, 255, 0.04);
    }
  }
`;

const ScoreBadge = styled.span`
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  color: #fff;
  background-color: ${({ color }) => color};
`;

const CustomTooltipContainer = styled.div`
  background: rgba(15, 23, 42, 0.85);
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(10px);
  
  .label {
    color: #f8fafc;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .intro {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #e2e8f0;
    font-size: 0.9rem;
  }
`;


// --- Helper Functions & Custom Components ---

const getScoreColor = (score) => {
  if (score >= 8) return '#10b981'; // Green
  if (score >= 5) return '#f59e0b'; // Amber
  return '#ef4444'; // Red
};

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <CustomTooltipContainer>
        <p className="label">{`Analysis: ${label}`}</p>
        {payload.map(pld => (
          <div className="intro" key={pld.dataKey} style={{ color: pld.color }}>
            {`${pld.dataKey}: ${pld.value.toFixed(1)}`}
          </div>
        ))}
      </CustomTooltipContainer>
    );
  }
  return null;
};

// --- The Main Component ---
function HistoryTab({ data }) {
  const [historyData, setHistoryData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchHistory = async () => {
      if (!data.filename) {
          setError('Filename is missing.');
          return;
      }
      try {
        const response = await getHistory(data.filename);
        const sortedData = response.data.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        setHistoryData(sortedData);
      } catch (err) {
        setError('No previous analysis history found for this file.');
      }
    };
    fetchHistory();
  }, [data.filename]);

  if (error) {
    return <HistoryContainer><InfoBox><AlertTriangle size={32} color="#f87171" />{error}</InfoBox></HistoryContainer>;
  }

  if (!historyData) {
    return <HistoryContainer><InfoBox><Loader size={32} className="spinner" />Loading history...</InfoBox></HistoryContainer>;
  }
  
  if (historyData.length < 2) {
      return <HistoryContainer><InfoBox><TrendingUp size={32} />Analyze this deck again to see improvement trends over time.</InfoBox></HistoryContainer>;
  }

  const chartData = historyData.map(item => ({
    date: new Date(item.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    'Readiness': item.investment_readiness_score,
    'Moat': item.competitive_moat_score,
    'GTM': item.gtm_strategy_score,
  }));

  return (
    <HistoryContainer>
      <Card delay="0.1s">
        <CardHeader>
          <TrendingUp size={24} />
          <h3>Improvement Trend Over Time</h3>
        </CardHeader>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
             <defs>
              <linearGradient id="colorReadiness" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#14f1d9" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#14f1d9" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorMoat" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#818cf8" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#818cf8" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorGTM" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f472b6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#f472b6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
            <XAxis dataKey="date" stroke="#94a3b8" />
            <YAxis domain={[0, 10]} stroke="#94a3b8" />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Area type="monotone" dataKey="Readiness" stroke="#14f1d9" strokeWidth={2} fillOpacity={1} fill="url(#colorReadiness)" activeDot={{ r: 8 }} />
            <Area type="monotone" dataKey="Moat" stroke="#818cf8" strokeWidth={2} fillOpacity={1} fill="url(#colorMoat)" activeDot={{ r: 8 }} />
            <Area type="monotone" dataKey="GTM" stroke="#f472b6" strokeWidth={2} fillOpacity={1} fill="url(#colorGTM)" activeDot={{ r: 8 }} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>
      
      <Card delay="0.2s">
        <CardHeader>
            <Clipboard size={24} />
            <h3>Analysis History</h3>
        </CardHeader>
        <Table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Persona</th>
              <th>Readiness Score</th>
              <th>Moat Score</th>
              <th>GTM Score</th>
            </tr>
          </thead>
          <tbody>
            {[...historyData].reverse().map(item => (
              <tr key={item.id}>
                <td>{new Date(item.timestamp).toLocaleString()}</td>
                <td>{item.persona}</td>
                <td><ScoreBadge color={getScoreColor(item.investment_readiness_score)}>{item.investment_readiness_score.toFixed(1)} / 10</ScoreBadge></td>
                <td><ScoreBadge color={getScoreColor(item.competitive_moat_score)}>{item.competitive_moat_score} / 10</ScoreBadge></td>
                <td><ScoreBadge color={getScoreColor(item.gtm_strategy_score)}>{item.gtm_strategy_score} / 10</ScoreBadge></td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </HistoryContainer>
  );
}

export default HistoryTab;