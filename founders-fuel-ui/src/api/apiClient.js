// // // // src/api/apiClient.js
// // // // CORRECTED VERSION

// // // import axios from 'axios';

// // // // The base URL of our FastAPI backend
// // // const API_BASE_URL = 'http://127.0.0.1:8000/api';

// // // const apiClient = axios.create({
// // //     baseURL: API_BASE_URL,
// // //     headers: {
// // //         'Content-Type': 'application/json',
// // //     },
// // // });

// // // /**
// // //  * Uploads the pitch deck and configuration to start the analysis.
// // //  * @param {File} file - The pitch deck file.
// // //  * @param {string} category - The industry category.
// // //  * @param {string} personaName - The selected VC persona.
// // //  * @returns {Promise<object>} The response from the server (e.g., { task_id }).
// // //  */
// // // export const startAnalysis = (file, category, personaName) => {
// // //     const formData = new FormData();
// // //     formData.append('file', file);
// // //     formData.append('category', category);
// // //     formData.append('persona_name', personaName);

// // //     return apiClient.post('/analyze', formData, {
// // //         headers: {
// // //             'Content-Type': 'multipart/form-data',
// // //         },
// // //     });
// // // };

// // // /**
// // //  * Fetches the current status of an analysis task.
// // //  * @param {string} taskId - The ID of the task.
// // //  * @returns {Promise<object>} The status response from the server.
// // //  */
// // // export const getTaskStatus = (taskId) => {
// // //     return apiClient.get(`/status/${taskId}`);
// // // };
// // // /**
// // //  * Fetches the analysis history for a specific filename.
// // //  * @param {string} filename - The name of the file to get history for.
// // //  * @returns {Promise<object>} A list of historical analysis summaries.
// // //  */
// // // export const getHistory = (filename) => {
// // //     return apiClient.get(`/history/${filename}`);
// // // };

// // // /**
// // //  * Fetches the full analysis results from the database.
// // //  * @param {number} analysisId - The final ID of the completed analysis.
// // //  * @returns {Promise<object>} The full analysis JSON object.
// // //  */
// // // export const getAnalysisResults = (analysisId) => {
// // //     return apiClient.get(`/results/${analysisId}`);
// // // };

// // // // This is the correct export. We are exporting the functions individually.A


// // // src/api/apiClient.js
// // // CORRECTED AND UNIFIED VERSION

// // import axios from 'axios';

// // // The base URL of our FastAPI backend
// // const API_BASE_URL = 'http://127.0.0.1:8000/api';

// // const apiClient = axios.create({
// //     baseURL: API_BASE_URL,
// //     headers: {
// //         'Content-Type': 'application/json',
// //     },
// // });

// // // --- Pitch Deck Analysis Functions (Existing) ---

// // /**
// //  * Uploads the pitch deck and configuration to start the analysis.
// //  * @param {File} file - The pitch deck file.
// //  * @param {string} category - The industry category.
// //  * @param {string} personaName - The selected VC persona.
// //  * @returns {Promise<object>} The response from the server (e.g., { task_id }).
// //  */
// // export const startAnalysis = (file, category, personaName) => {
// //     const formData = new FormData();
// //     formData.append('file', file);
// //     formData.append('category', category);
// //     formData.append('persona_name', personaName);

// //     return apiClient.post('/analyze', formData, {
// //         headers: {
// //             'Content-Type': 'multipart/form-data',
// //         },
// //     });
// // };

// // /**
// //  * Fetches the current status of an analysis task.
// //  * @param {string} taskId - The ID of the task.
// //  * @returns {Promise<object>} The status response from the server.
// //  */
// // export const getTaskStatus = (taskId) => {
// //     return apiClient.get(`/status/${taskId}`);
// // };

// // /**
// //  * Fetches the analysis history for a specific filename.
// //  * @param {string} filename - The name of the file to get history for.
// //  * @returns {Promise<object>} A list of historical analysis summaries.
// //  */
// // export const getHistory = (filename) => {
// //     // We encode the filename to handle special characters safely in the URL
// //     return apiClient.get(`/history/${encodeURIComponent(filename)}`);
// // };

// // /**
// //  * Fetches the full analysis results from the database.
// //  * @param {number} analysisId - The final ID of the completed analysis.
// //  * @returns {Promise<object>} The full analysis JSON object.
// //  */
// // export const getAnalysisResults = (analysisId) => {
// //     return apiClient.get(`/results/${analysisId}`);
// // };


// // // --- Investor Matching Functions (New) ---

// // /**
// //  * Finds top investor matches based on a startup description.
// //  * @param {string} description - The startup's description.
// //  * @param {string} stage - The desired funding stage.
// //  * @param {string} investorType - The desired type of investor.
// //  * @returns {Promise<object>} The match results from the server.
// //  */
// // export const findMatches = (description, stage, investorType) => {
// //     const payload = {
// //         description: description,
// //         stage: stage,
// //         investor_type: investorType, // Matches the FastAPI Pydantic model
// //     };
// //     // This call uses the default 'application/json' header
// //     return apiClient.post('/find-matches', payload);
// // };


// // src/api/apiClient.js
// // CORRECTED AND UNIFIED VERSION

// import axios from 'axios';

// // The base URL of our FastAPI backend
// const API_BASE_URL = 'http://127.0.0.1:8000/api';

// const apiClient = axios.create({
//     baseURL: API_BASE_URL,
//     headers: {
//         'Content-Type': 'application/json',
//     },
// });

// // --- Pitch Deck Analysis Functions (Existing) ---

// /**
//  * Uploads the pitch deck and configuration to start the analysis.
//  * @param {File} file - The pitch deck file.
//  * @param {string} category - The industry category.
//  * @param {string} personaName - The selected VC persona.
//  * @returns {Promise<object>} The response from the server (e.g., { task_id }).
//  */
// export const startAnalysis = (file, category, personaName) => {
//     const formData = new FormData();
//     formData.append('file', file);
//     formData.append('category', category);
//     formData.append('persona_name', personaName);

//     return apiClient.post('/analyze', formData, {
//         headers: {
//             'Content-Type': 'multipart/form-data',
//         },
//     });
// };

// /**
//  * Fetches the current status of an analysis task.
//  * @param {string} taskId - The ID of the task.
//  * @returns {Promise<object>} The status response from the server.
//  */
// export const getTaskStatus = (taskId) => {
//     return apiClient.get(`/status/${taskId}`);
// };

// /**
//  * Fetches the analysis history for a specific filename.
//  * @param {string} filename - The name of the file to get history for.
//  * @returns {Promise<object>} A list of historical analysis summaries.
//  */
// export const getHistory = (filename) => {
//     // We encode the filename to handle special characters safely in the URL
//     return apiClient.get(`/history/${encodeURIComponent(filename)}`);
// };

// /**
//  * Fetches the full analysis results from the database.
//  * @param {number} analysisId - The final ID of the completed analysis.
//  * @returns {Promise<object>} The full analysis JSON object.
//  */
// export const getAnalysisResults = (analysisId) => {
//     return apiClient.get(`/results/${analysisId}`);
// };


// // --- Investor Matching Functions (Existing) ---

// /**
//  * Finds top investor matches based on a startup description.
//  * @param {string} description - The startup's description.
//  * @param {string} stage - The desired funding stage.
//  * @param {string} investorType - The desired type of investor.
//  * @returns {Promise<object>} The match results from the server.
//  */
// export const findMatches = (description, stage, investorType) => {
//     const payload = {
//         description: description,
//         stage: stage,
//         investor_type: investorType, // Matches the FastAPI Pydantic model
//     };
//     // This call uses the default 'application/json' header
//     return apiClient.post('/find-matches', payload);
// };

// // --- NEW: 7-Domain Evaluation Function ---

// /**
//  * Runs the comprehensive 7-Domain startup evaluation with files and web context.
//  * @param {string} startupName - The name of the startup.
//  * @param {string} startupIdea - The detailed startup idea/description.
//  * @param {File[]} files - An array of pitch deck/document files.
//  * @returns {Promise<object>} The structured ComprehensiveEvaluationResponse.
//  */
// export const runEvaluation = (startupName, startupIdea, files = []) => {
//     const formData = new FormData();
    
//     // Append fields to match the FastAPI @router.post("/evaluate") arguments
//     formData.append('startup_idea', startupIdea);
//     if (startupName) {
//         formData.append('startup_name', startupName);
//     }
    
//     // Append multiple files, using the field name 'files'
//     files.forEach(file => {
//         formData.append('files', file); 
//     });

//     return apiClient.post('/evaluate', formData, {
//         headers: {
//             // Must use 'multipart/form-data' for file uploads
//             'Content-Type': 'multipart/form-data', 
//         },
//     });
// };


// src/api/apiClient.js
// CORRECTED AND UNIFIED VERSION (with new evaluation function)

import axios from 'axios';

// The base URL of our FastAPI backend
const API_BASE_URL = 'https://iqrasayedhassan-founders-fuel-backend.hf.space/api';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// --- Pitch Deck Analysis Functions (Existing) ---

/**
 * Uploads the pitch deck and configuration to start the analysis.
 * @param {File} file - The pitch deck file.
 * @param {string} category - The industry category.
 * @param {string} personaName - The selected VC persona.
 * @returns {Promise<object>} The response from the server (e.g., { task_id }).
 */
export const startAnalysis = (file, category, personaName) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', category);
    formData.append('persona_name', personaName);

    return apiClient.post('/analyze', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
};

/**
 * Fetches the current status of an analysis task.
 * @param {string} taskId - The ID of the task.
 * @returns {Promise<object>} The status response from the server.
 */
export const getTaskStatus = (taskId) => {
    return apiClient.get(`/status/${taskId}`);
};

/**
 * Fetches the analysis history for a specific filename.
 * @param {string} filename - The name of the file to get history for.
 * @returns {Promise<object>} A list of historical analysis summaries.
 */
export const getHistory = (filename) => {
    // We encode the filename to handle special characters safely in the URL
    return apiClient.get(`/history/${encodeURIComponent(filename)}`);
};

/**
 * Fetches the full analysis results from the database.
 * @param {number} analysisId - The final ID of the completed analysis.
 * @returns {Promise<object>} The full analysis JSON object.
 */
export const getAnalysisResults = (analysisId) => {
    return apiClient.get(`/results/${analysisId}`);
};


// --- Investor Matching Functions (Existing) ---

/**
 * Finds top investor matches based on a startup description.
 * @param {string} description - The startup's description.
 * @param {string} stage - The desired funding stage.
 * @param {string} investorType - The desired type of investor.
 * @returns {Promise<object>} The match results from the server.
 */
export const findMatches = (description, stage, investorType) => {
    const payload = {
        description: description,
        stage: stage,
        investor_type: investorType, // Matches the FastAPI Pydantic model
    };
    // This call uses the default 'application/json' header
    return apiClient.post('/find-matches', payload);
};


// --- 7-Domain Startup Evaluation Functions (New) ---

/**
 * Sends a startup idea to the backend for a comprehensive 7-domain evaluation.
 * @param {object} evaluationData - The data for the evaluation.
 * @param {string} evaluationData.startup_idea - The core idea and description of the startup.
 * @param {string} [evaluationData.startup_name] - The optional name of the startup.
 * @param {string} [evaluationData.uploaded_content] - Optional text content from uploaded documents.
 * @returns {Promise<object>} The full 7-domain evaluation from the server.
 */
export const evaluateStartup = (evaluationData) => {
    // The keys in the evaluationData object (e.g., startup_idea) must match
    // the Pydantic model (EvaluationRequest) in the FastAPI backend.
    return apiClient.post('/evaluate', evaluationData);
};

/**
 * Predicts the success probability of a crowdfunding campaign.
 * @param {object} payload - The form data (goal, backers, usd_pledged, text).
 * @returns {Promise<object>} An object with the success_probability percentage.
 */
export const predictSuccess = (payload) => {
    // Note: This matches the /predict endpoint added to endpoints.py
    return apiClient.post('/predict', payload);
}

/**
 * Generates a narrative explanation for the prediction.
 * @param {object} payload - The form data (goal, backers, usd_pledged, text).
 * @returns {Promise<object>} An object with the narrative summary.
 */
export const explainSuccess = (payload) => {
    // Note: This matches the /explain endpoint added to endpoints.py
    return apiClient.post('/explain', payload);
};