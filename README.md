<div align="center">
  <h1 style="font-family: 'Poppins', sans-serif; font-size: 42px; font-weight: 700; color:purple">
    🚀 Founders Fuel – AI Powered Startup Analyst For Both Investors And Founders
  </h1>
  <h3>
    <a href="https://founder-s-fuel-final-code-1y38.vercel.app/" target="_blank">
      🔗 Visit The Website
    </a><br>
    <a href="https://youtu.be/O3U3rKZdfro" target="_blank">
      🎥See The Demo Video
    </a>
  </h3>
</div>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">

---

## 🌟 Overview

**Founders Fuel** bridges the gap between ambitious founders and venture capital.
Using advanced **RAG**, **LLMs**, and **Machine Learning**, this platform acts as a 24/7 intelligent consultant.

From pitch deck roasting to investor matching and startup validation — Founders Fuel has you covered.

---

## ✨ Key Features

| Feature                         | Description                                                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **📊 AI Pitch Deck Analyzer**   | Upload PDF decks and receive deep analysis from **4 unique VC personas** with scoring on clarity, market, product, and fundraising strength.                  |
| **🤝 Smart Investor Matching**  | Match with the right investors based on sector, stage, geography, and thesis — with actionable outreach insights.                                             |
| **🧠 7-Domain Idea Evaluation** | A structured RAG-powered framework analyzing feasibility, market size, competitors, execution difficulty, defensibility, monetization, and user pain.         |
| **💬 RAG Chatbot (Standalone)** | A fully separate interactive chatbot for market research, competitor insights, idea refinement, and evidence-backed recommendations. Runs as a Streamlit app. |
| **🔮 Success Predictor**        | ML model with SHAP explainability to estimate your fundraising success probability and show why the model predicts that outcome.                              |

---

<img width="600" height="600" alt="Screenshot 2025-11-25 213722" src="https://github.com/user-attachments/assets/dc120f48-5d4c-4124-9155-0289d7b97b1e" />


## 🛠️ Tech Stack

### **Frontend**

* React (Vite)
* Styled Components
* Recharts

### **Backend**

* Python
* FastAPI
* Google Gemini API
* HuggingFace Transformers (BERT, NER, Sentiment)
* SQLite
* Scikit-Learn + SHAP

### **RAG Module**

* Streamlit
* Pinecone 
* Sentence Transformers
* PDF/Text ingestion
  
---

## 📂 Project Structure

```
founder-sfuel/
├── founders-fuel-ui/          # React Client
│   ├── src/
│   │   ├── components/        # UI Components
│   │   ├── pages/             # App Screens (CoPilot, Results)
│   │   └── api/               # Axios Interceptors
│   └── package.json
│
├── vc-copilot-backend/        # FastAPI Server
│   ├── app/
│   │   ├── core/              # ML & Model Config
│   │   ├── services/          # Matching, Pitch Processing
│   │   └── api/               # FastAPI Routes
│   ├── main.py
│   ├── requirements.txt
│   └── pitchmodel.pkl
│
└── finalcodes/                # Standalone Tools
    └── ragfinalone.py         # Streamlit RAG Chatbot
```

---

# 🚀 Project Setup Guide


## 🔐 Environment Variables (Required)

Create a `.env` file in `vc-copilot-backend/`:

```
GEMINI_API_KEY=

SERPAPI_API_KEY=

PINECONE_API_KEY=

```
## 🖥️ Frontend Setup

```
cd founders-fuel-ui
npm install
npm run dev
```

---

## ⚙️ Backend Setup (FastAPI)

```
cd vc-copilot-backend
uvicorn main:app --reload --port 8000
```

---

## 🧠 RAG Chatbot Setup (Standalone)

```
cd finalcodes
streamlit run ragfinalone.py
```

---

<div align="center">

Made with ❤️ by the **Iqra Sayed Hassan**

</div>

---
