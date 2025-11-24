# # # # # vc-copilot-backend/app/api/endpoints.py
# # # # # CORRECTED AND FINALIZED VERSION

# # # # import uuid
# # # # import json
# # # # import re
# # # # import os
# # # # import logging
# # # # from datetime import datetime
# # # # from uuid import UUID
# # # # from typing import Dict, List, Any
# # # # from urllib.parse import unquote

# # # # # Third-party imports
# # # # import textstat
# # # # import numpy as np
# # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # import google.generativeai as genai
# # # # from dotenv import load_dotenv
# # # # from dataclasses import dataclass, asdict

# # # # # FastAPI imports
# # # # from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException

# # # # # Local app imports
# # # # from app.core.ml import ml_models
# # # # from app.db.database import DatabaseManager, NumpyEncoder # <-- Make sure NumpyEncoder is imported
# # # # from app.services.file_processor import FileProcessor
# # # # from app.models.schemas import AnalyzeResponse, StatusResponse, TaskStatus

# # # # # --- Basic Setup ---
# # # # logging.basicConfig(level=logging.INFO)
# # # # logger = logging.getLogger(__name__)
# # # # load_dotenv()
# # # # try:
# # # #     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# # # # except Exception as e:
# # # #     logger.error(f"AI model configuration failed: {e}")


# # # # # --- DATA CLASSES AND CONFIG ---
# # # # @dataclass
# # # # class SlideAnalysis:
# # # #     readability_interpretation: str; quantitative_count: int; sentiment: Dict[str, Any]; named_entities: List[Dict[str, str]]

# # # # @dataclass
# # # # class LocalAnalysisResult:
# # # #     redundant_slides: List[Dict[str, Any]]
# # # #     local_metrics: List[Any]
# # # #     prob_sol_alignment: float
# # # #     checklist: Dict[str, bool]
# # # #     buzzword_analysis: Dict[str, Any]
# # # #     ner_insights: Dict[str, Any]

# # # # @dataclass
# # # # class AIAnalysisResult:
# # # #     dimensional_scores: Dict[str, float]; vc_verdict: str; investment_readiness_score: float; competitive_moat: Dict[str, Any]; gtm_strategy: Dict[str, Any]; team_market_fit: Dict[str, Any]; slide_by_slide_feedback: List[Dict[str, str]]; key_strengths: List[str]; key_improvement_suggestions: List[str]

# # # # class Config:
# # # #     VC_PERSONAS = {
# # # #         "The Friendly Mentor": "a friendly, encouraging mentor who focuses on nurturing potential",
# # # #         "The Data-Driven Analyst": "a sharp, data-driven analyst who demands concrete metrics and evidence",
# # # #         "The 'Big Picture' Strategist": "a 'big picture' strategist who evaluates market timing and scalability",
# # # #         "The Devil's Advocate": "a devil's advocate who challenges assumptions and identifies risks"
# # # #     }
# # # #     BUZZWORDS = ['synergy', 'disrupt', 'paradigm shift', 'game-changer', 'revolutionize', 'innovative', 'cutting-edge', 'breakthrough', 'revolutionary']
# # # #     CHECKLIST_PATTERNS = {
# # # #         "TAM/SAM/SOM": r'tam|sam|som|total addressable market|market size',
# # # #         "Go-to-Market Strategy": r'go-to-market|gtm|target audience|distribution channels|marketing plan|customer acquisition',
# # # #         "Funding Ask": r'(\$[0-9,.]+m? seed|\$[0-9,.]+m? series [a-z]|raising|seeking|funding)',
# # # #         "Traction Metrics (KPIs)": r'mrr|arr|cac|ltv|mau|daily active users|kpi|revenue|growth|metrics',
# # # #         "Milestones / Timeline": r'q[1-4] 202[4-9]|roadmap|timeline|launch|scale|milestones',
# # # #         "Team Credentials": r'ex-google|ex-meta|phd|serial entrepreneur|[0-9]+\+ years of experience|founder|cto|ceo',
# # # #         "Tech Stack / IP": r'ai|blockchain|patent|proprietary algorithm|ml model|technology|infrastructure',
# # # #         "Regulatory / Compliance": r'hipaa|gdpr|fda approval|iso certified|compliance|regulation',
# # # #         "Risk Disclosure": r'risk|challenge|headwind|burn rate|scalability issues|threats',
# # # #         "Financial Projections": r'revenue projection|financial forecast|p&l|unit economics|burn rate'
# # # #     }

# # # # # --- ANALYSIS ENGINE CLASS (Cleaned up) ---
# # # # class AnalysisEngine:
# # # #     def __init__(self, models: Dict[str, Any]):
# # # #         self.models = models

# # # #     def perform_local_analysis(self, slides: List[str]) -> LocalAnalysisResult:
# # # #         if not slides:
# # # #             raise ValueError("No slides available for analysis.")
        
# # # #         embeddings = self.models['bert'].encode(slides)
# # # #         sim_matrix = cosine_similarity(embeddings)
        
# # # #         redundant_pairs = []
# # # #         for i in range(len(slides)):
# # # #             for j in range(i + 1, len(slides)):
# # # #                 if sim_matrix[i][j] > 0.85:
# # # #                     redundant_pairs.append({'slide1': i + 1, 'slide2': j + 1, 'similarity': float(sim_matrix[i][j])})

# # # #         prob_sol_alignment = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]) if len(slides) > 1 else 0.0
# # # #         local_metrics = [self._analyze_single_slide(text) for text in slides if text.strip()]
# # # #         combined_text = " ".join(slides).lower()
# # # #         checklist = self._perform_checklist_analysis(combined_text)
# # # #         buzzword_count = self._count_buzzwords(combined_text)
# # # #         buzzword_analysis = {"count": buzzword_count}
# # # #         ner_insights = self._perform_ner_analysis(slides)

# # # #         return LocalAnalysisResult(
# # # #             redundant_slides=redundant_pairs,
# # # #             local_metrics=[asdict(m) for m in local_metrics],
# # # #             prob_sol_alignment=prob_sol_alignment,
# # # #             checklist=checklist,
# # # #             buzzword_analysis=buzzword_analysis,
# # # #             ner_insights=ner_insights
# # # #         )

# # # #     def _analyze_single_slide(self, slide_text: str) -> SlideAnalysis:
# # # #         flesch_score = textstat.flesch_reading_ease(slide_text)
# # # #         if flesch_score > 80: readability = "Very Easy"
# # # #         elif flesch_score > 60: readability = "Easy"
# # # #         elif flesch_score > 40: readability = "Standard"
# # # #         else: readability = "Difficult"

# # # #         quant_patterns = [r'\$\d[\d,.]*[kmb]?', r'\d[\d,.]*%', r'\d[\d,.]+\s?(?:million|billion|m|b)', r'\d+x', r'[0-9]+\+\s*(?:users|customers|clients)']
# # # #         quantitative_count = sum(len(re.findall(p, slide_text, re.I)) for p in quant_patterns)

# # # #         try:
# # # #             sentiment_scores = self.models['sentiment'](slide_text[:512])[0]
# # # #             sentiment = max(sentiment_scores, key=lambda x: x['score'])
# # # #             sentiment['label'] = sentiment['label'].replace('LABEL_', '').upper()
# # # #         except Exception as e:
# # # #             logger.warning(f"Sentiment analysis failed: {e}")
# # # #             sentiment = {'label': 'NEUTRAL', 'score': 0.0}

# # # #         try:
# # # #             entities = self.models['ner'](slide_text[:512])
# # # #             named_entities = [{'entity': e['word'], 'type': e.get('entity_group', 'UNKNOWN')} for e in entities if e.get('score', 0) > 0.8]
# # # #         except Exception:
# # # #             named_entities = []

# # # #         return SlideAnalysis(readability, quantitative_count, sentiment, named_entities)

# # # #     def _perform_checklist_analysis(self, text: str) -> Dict[str, bool]:
# # # #         return {item: bool(re.search(pattern, text, re.I)) for item, pattern in Config.CHECKLIST_PATTERNS.items()}

# # # #     def _count_buzzwords(self, text: str) -> int:
# # # #         return sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text, re.I)) for word in Config.BUZZWORDS)

# # # #     def _perform_ner_analysis(self, slides: List[str]) -> Dict[str, str]:
# # # #             """Enhanced NER analysis for key business insights."""
# # # #             insights = {}
# # # #             all_entities = []
            
# # # #             # Collect all entities from all slides
# # # #             for slide in slides:
# # # #                 if slide:
# # # #                     try:
# # # #                         entities = self.models['ner'](slide[:512])
# # # #                         for entity in entities:
# # # #                             if entity.get('score', 0) > 0.8:  # High confidence only
# # # #                                 all_entities.append({
# # # #                                     'text': entity['word'].replace('##', ''),  # Clean subword tokens
# # # #                                     'type': entity.get('entity_group', 'UNKNOWN'),
# # # #                                     'confidence': entity.get('score', 0)
# # # #                                 })
# # # #                     except Exception:
# # # #                         continue
            
# # # #             # Analyze team slide specifically
# # # #             team_slide = self._find_slide_by_keywords(slides, ['team', 'founder', 'ceo', 'cto', 'management', 'leadership'])
# # # #             if team_slide:
# # # #                 try:
# # # #                     team_entities = self.models['ner'](team_slide[:512])
# # # #                     people = [e for e in team_entities if e.get('entity_group') == 'PER' and e.get('score', 0) > 0.8]
# # # #                     orgs = [e for e in team_entities if e.get('entity_group') == 'ORG' and e.get('score', 0) > 0.8]
                    
# # # #                     # Extract previous company names (valuable for credibility)
# # # #                     previous_companies = []
# # # #                     for org in orgs:
# # # #                         org_name = org['word'].replace('##', '')
# # # #                         if any(keyword in team_slide.lower() for keyword in ['ex-', 'former', 'previously', 'worked at']):
# # # #                             previous_companies.append(org_name)
                    
# # # #                     insights['team_summary'] = f"Found {len(people)} team members"
# # # #                     if previous_companies:
# # # #                         insights['team_summary'] += f" with experience at: {', '.join(previous_companies[:3])}"
# # # #                     if orgs and not previous_companies:
# # # #                         insights['team_summary'] += f" associated with {len(orgs)} organizations"
                        
# # # #                 except Exception:
# # # #                     insights['team_summary'] = "Team slide found but analysis failed"
# # # #             else:
# # # #                 insights['team_summary'] = "❌ No team slide detected - major red flag for investors"

# # # #             # Analyze competition/market slide
# # # #             comp_slide = self._find_slide_by_keywords(slides, ['competition', 'competitor', 'competitive', 'market', 'landscape'])
# # # #             if comp_slide:
# # # #                 try:
# # # #                     comp_entities = self.models['ner'](comp_slide[:512])
# # # #                     competitors = [e for e in comp_entities if e.get('entity_group') == 'ORG' and e.get('score', 0) > 0.8]
# # # #                     competitor_names = [e['word'].replace('##', '') for e in competitors]
                    
# # # #                     insights['competition_summary'] = f"Identified {len(competitors)} competitors: {', '.join(competitor_names[:5])}"
# # # #                     if len(competitor_names) > 5:
# # # #                         insights['competition_summary'] += f" and {len(competitor_names) - 5} more"
                        
# # # #                 except Exception:
# # # #                     insights['competition_summary'] = "Competition slide found but analysis failed"
# # # #             else:
# # # #                 insights['competition_summary'] = "⚠️ No competitive analysis slide - investors need to see market awareness"

# # # #             # Financial entities analysis (revenue, funding, etc.)
# # # #             financial_entities = []
# # # #             money_pattern = r'\$[\d,.]+(M|B|K|million|billion|thousand)?'
# # # #             for slide in slides:
# # # #                 if slide:
# # # #                     money_mentions = re.findall(money_pattern, slide, re.IGNORECASE)
# # # #                     financial_entities.extend(money_mentions)
            
# # # #             insights['financial_summary'] = f"Found {len(financial_entities)} financial figures across all slides"
            
# # # #             # Location analysis (market focus)
# # # #             locations = [e for e in all_entities if e['type'] == 'LOC']
# # # #             if locations:
# # # #                 location_names = list(set([e['text'] for e in locations]))
# # # #                 insights['market_geography'] = f"Geographic focus: {', '.join(location_names[:3])}"
# # # #             else:
# # # #                 insights['market_geography'] = "No clear geographic market focus identified"
                
# # # #             return insights
            
# # # #     def _find_slide_by_keywords(self, slides: List[str], keywords: List[str]) -> str:
# # # #         # ... (This method is fine)
# # # #         for slide in slides:
# # # #             if slide and any(keyword in slide.lower() for keyword in keywords):
# # # #                 return slide
# # # #         return ""


# # # # # --- AI ANALYZER CLASS (Cleaned up) ---
# # # # class AIAnalyzer:
# # # #     def __init__(self):
# # # #         self.ai_model = None
# # # #         self.setup_ai_model() # <-- BUG FIX #2: Added this call

# # # #     def setup_ai_model(self):
# # # #         try:
# # # #             if os.getenv("GEMINI_API_KEY"):
# # # #                 self.ai_model = genai.GenerativeModel("models/gemini-2.5-flash")
# # # #                 logger.info("Successfully configured AI model")
# # # #             else:
# # # #                 logger.error("GEMINI_API_KEY not found in environment variables")
# # # #         except Exception as e:
# # # #             logger.error(f"Failed to configure AI model: {e}")

# # # #     def perform_ai_analysis(self, slides: List[str], category: str, persona_desc: str, local_context: LocalAnalysisResult) -> AIAnalysisResult:
# # # #         if not self.ai_model:
# # # #             raise ValueError("AI model is not available for analysis.")

# # # #         # <-- BUG FIX #3: Use the NumpyEncoder
# # # #         context_json = json.dumps(asdict(local_context), cls=NumpyEncoder)
# # # #         prompt = self._build_analysis_prompt(slides, category, persona_desc, context_json)
        
# # # #         response = self.ai_model.generate_content(prompt)
# # # #         return self._parse_ai_response(response.text)

# # # #     def _build_analysis_prompt(self, slides: List[str], category: str, 
# # # #                                 persona_desc: str, context_json: str) -> str:
# # # #             """Build comprehensive analysis prompt."""
# # # #             return f"""
# # # #             As a venture capitalist acting as {persona_desc}, analyze this '{category}' pitch deck.
            
# # # #             CONTEXT FROM PRE-ANALYSIS: {context_json}
            
# # # #             FULL DECK TEXT: {json.dumps(slides[:10])}  # Limit to avoid token limits
            
# # # #             Provide your analysis in a single, valid JSON object with this exact structure:
# # # #             {{
# # # #                 "dimensional_scores": {{
# # # #                     "clarity": 0,
# # # #                     "completeness": 0,
# # # #                     "persuasiveness": 0,
# # # #                     "data_driven": 0
# # # #                 }},
# # # #                 "vc_verdict": "Your final, conclusive statement as this VC persona",
# # # #                 "investment_readiness_score": 0,
# # # #                 "competitive_moat": {{
# # # #                     "score": 0,
# # # #                     "feedback": "Detailed assessment of competitive advantages"
# # # #                 }},
# # # #                 "gtm_strategy": {{
# # # #                     "score": 0,
# # # #                     "feedback": "Analysis of go-to-market approach"
# # # #                 }},
# # # #                 "team_market_fit": {{
# # # #                     "score": 0,
# # # #                     "feedback": "Evaluation of team capabilities vs market needs"
# # # #                 }},
# # # #                 "slide_by_slide_feedback": [
# # # #                     {{
# # # #                         "slide_number": 1,
# # # #                         "clarity_feedback": "...",
# # # #                         "tone_feedback": "...",
# # # #                         "intent_fulfillment_feedback": "..."
# # # #                     }}
# # # #                 ],
# # # #                 "key_strengths": ["strength1", "strength2"],
# # # #                 "key_improvement_suggestions": ["improvement1", "improvement2"]
# # # #             }}
            
# # # #              CRITICAL REQUIREMENTS:
# # # #         - The "vc_verdict" MUST be a single, conversational paragraph of approximately 80-85 words.
# # # #         - The "key_strengths" and "key_improvement_suggestions" MUST be arrays of simple strings.
# # # #         - DO NOT include any Markdown (like ** **, #,  ' ' , ` ` or *) inside any string values in the JSON.
# # # #         - All scores MUST be integers from 0-10.
# # # #         - Respond ONLY with the valid JSON object and nothing else.
# # # #         """

# # # #     def _parse_ai_response(self, response_text: str) -> AIAnalysisResult:
# # # #         # (This method is fine)
# # # #         try:
# # # #             json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
# # # #             json_str = json_match.group(1) if json_match else response_text[response_text.find('{'):response_text.rfind('}')+1]
# # # #             parsed = json.loads(json_str)
# # # #             # You might want to add validation here for required fields
# # # #             return AIAnalysisResult(**parsed)
# # # #         except (json.JSONDecodeError, TypeError) as e:
# # # #             logger.error(f"Failed to parse AI response: {e}\nResponse: {response_text[:300]}")
# # # #             raise ValueError("Invalid JSON response from AI.")


# # # # # --- FastAPI Task Management & Endpoints ---
# # # # api_router = APIRouter()
# # # # task_statuses: Dict[UUID, Dict[str, Any]] = {}
# # # # db_manager = DatabaseManager()

# # # # def run_real_analysis(task_id: UUID, file_content: bytes, filename: str, file_type: str, category: str, persona_name: str):
# # # #     try:
# # # #         db_manager.init_db()
# # # #         clean_filename = unquote(filename)

# # # #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 1/4: Extracting slides..."}
# # # #         slides = FileProcessor.extract_slides(file_content, file_type)
        
# # # #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 2/4: Performing local ML analysis..."}
# # # #         engine = AnalysisEngine(models=ml_models)
# # # #         local_analysis = engine.perform_local_analysis(slides)

# # # #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 3/4: Getting insights from AI..."}
# # # #         persona_desc = Config.VC_PERSONAS.get(persona_name, "a standard venture capitalist")
# # # #         ai_analyzer = AIAnalyzer()
# # # #         ai_analysis = ai_analyzer.perform_ai_analysis(slides, category, persona_desc, local_analysis)

# # # #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 4/4: Saving analysis..."}
# # # #         full_analysis = {
# # # #             **asdict(ai_analysis),
# # # #             **asdict(local_analysis),
# # # #             "slide_texts": slides,
# # # #             "timestamp": datetime.now().isoformat(),
# # # #             "analyzed_as_persona": persona_name,
# # # #             "category": category,
# # # #             "filename": clean_filename
# # # #         }

# # # #         analysis_id = db_manager.save_analysis(clean_filename, category, persona_name, full_analysis)
# # # #         task_statuses[task_id] = {"status": TaskStatus.COMPLETE, "analysis_id": analysis_id}
# # # #         logger.info(f"Analysis complete for task {task_id}. Saved as ID {analysis_id}")

# # # #     except Exception as e:
# # # #         logger.error(f"Analysis failed for task {task_id}: {e}", exc_info=True)
# # # #         task_statuses[task_id] = {"status": TaskStatus.FAILED, "message": "An internal error occurred. Please check server logs."}


# # # # @api_router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
# # # # async def analyze_deck(background_tasks: BackgroundTasks, file: UploadFile = File(...), category: str = Form(...), persona_name: str = Form(...)):
# # # #     task_id = uuid.uuid4()
# # # #     task_statuses[task_id] = {"status": TaskStatus.PENDING}
# # # #     file_content = await file.read()
# # # #     background_tasks.add_task(run_real_analysis, task_id, file_content, file.filename, file.content_type, category, persona_name)
# # # #     return AnalyzeResponse(task_id=task_id)

# # # # @api_router.get("/status/{task_id}", response_model=StatusResponse)
# # # # async def get_analysis_status(task_id: UUID):
# # # #     status_info = task_statuses.get(task_id)
# # # #     if not status_info:
# # # #         raise HTTPException(status_code=404, detail="Task not found")
# # # #     return StatusResponse(task_id=task_id, **status_info)

# # # # @api_router.get("/results/{analysis_id}")
# # # # async def get_analysis_results(analysis_id: int):
# # # #     results = db_manager.load_analysis(analysis_id)
# # # #     if not results:
# # # #         raise HTTPException(status_code=404, detail="Analysis results not found")
# # # #     return results

# # # # @api_router.get("/history/{filename}")
# # # # async def get_analysis_history(filename: str):
# # # #     """
# # # #     Fetches the historical analysis data for a given filename.
# # # #     """
# # # #     logger.info(f"Fetching history for decoded filename: {filename}")
# # # #     history = db_manager.get_history(filename)
    
# # # #     if not history:
# # # #         logger.warning(f"No history found in DB for filename: {filename}")
# # # #         raise HTTPException(status_code=444, detail="No history found for this file")
    
# # # #     processed_history = []
# # # #     for record_row in history:
# # # #         try:
# # # #             # --- THE FIX IS HERE ---
# # # #             # Convert the sqlite3.Row object to a standard dictionary
# # # #             record = dict(record_row)

# # # #             analysis = json.loads(record['full_analysis'])
            
# # # #             processed_history.append({
# # # #                 "id": record['id'],
# # # #                 "timestamp": record['timestamp'],
# # # #                 # Now we can safely use the .get() method on the dictionary
# # # #                 "persona": record.get('persona', 'Unknown'),
# # # #                 "investment_readiness_score": analysis.get('investment_readiness_score', 0),
# # # #                 "competitive_moat_score": analysis.get('competitive_moat', {}).get('score', 0),
# # # #                 "gtm_strategy_score": analysis.get('gtm_strategy', {}).get('score', 0),
# # # #             })
# # # #         except (json.JSONDecodeError, KeyError) as e:
# # # #             logger.error(f"Could not process history record {record.get('id', 'N/A')}: {e}")
# # # #             continue
            
# # # #     logger.info(f"Found {len(processed_history)} history records for {filename}")
# # # #     return processed_history


# # # # vc-copilot-backend/app/api/endpoints.py
# # # # CORRECTED AND UNIFIED VERSION

# # # import uuid
# # # import json
# # # import re
# # # import os
# # # import logging
# # # from datetime import datetime
# # # from uuid import UUID
# # # from typing import Dict, List, Any
# # # from urllib.parse import unquote

# # # # Third-party imports
# # # import textstat
# # # import numpy as np
# # # from sklearn.metrics.pairwise import cosine_similarity
# # # import google.generativeai as genai
# # # from dotenv import load_dotenv
# # # from dataclasses import dataclass, asdict

# # # # FastAPI imports
# # # from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException

# # # # Local app imports
# # # from app.core.ml import ml_models
# # # from app.db.database import DatabaseManager, NumpyEncoder
# # # from app.services.file_processor import FileProcessor
# # # # --- ADDED: Import for the new matching service ---
# # # from app.services.matching_service import MatchingService
# # # # --- MODIFIED: Import all necessary schemas ---
# # # from app.models.schemas import (
# # #     AnalyzeResponse, 
# # #     StatusResponse, 
# # #     TaskStatus,
# # #     MatchRequest,
# # #     MatchResponse,
# # #     Investor
# # # )


# # # # --- Basic Setup ---
# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)
# # # load_dotenv()
# # # try:
# # #     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# # # except Exception as e:
# # #     logger.error(f"AI model configuration failed: {e}")

# # # # --- DATA CLASSES AND CONFIG (Existing - Unchanged) ---
# # # @dataclass
# # # class SlideAnalysis:
# # #     readability_interpretation: str; quantitative_count: int; sentiment: Dict[str, Any]; named_entities: List[Dict[str, str]]

# # # @dataclass
# # # class LocalAnalysisResult:
# # #     redundant_slides: List[Dict[str, Any]]; local_metrics: List[Any]; prob_sol_alignment: float; checklist: Dict[str, bool]; buzzword_analysis: Dict[str, Any]; ner_insights: Dict[str, Any]

# # # @dataclass
# # # class AIAnalysisResult:
# # #     dimensional_scores: Dict[str, float]; vc_verdict: str; investment_readiness_score: float; competitive_moat: Dict[str, Any]; gtm_strategy: Dict[str, Any]; team_market_fit: Dict[str, Any]; slide_by_slide_feedback: List[Dict[str, str]]; key_strengths: List[str]; key_improvement_suggestions: List[str]

# # # class Config:
# # #     VC_PERSONAS = {"The Friendly Mentor": "a friendly, encouraging mentor who focuses on nurturing potential", "The Data-Driven Analyst": "a sharp, data-driven analyst who demands concrete metrics and evidence", "The 'Big Picture' Strategist": "a 'big picture' strategist who evaluates market timing and scalability", "The Devil's Advocate": "a devil's advocate who challenges assumptions and identifies risks"}
# # #     BUZZWORDS = ['synergy', 'disrupt', 'paradigm shift', 'game-changer', 'revolutionize', 'innovative', 'cutting-edge', 'breakthrough', 'revolutionary']
# # #     CHECKLIST_PATTERNS = {"TAM/SAM/SOM": r'tam|sam|som|total addressable market|market size', "Go-to-Market Strategy": r'go-to-market|gtm|target audience|distribution channels|marketing plan|customer acquisition', "Funding Ask": r'(\$[0-9,.]+m? seed|\$[0-9,.]+m? series [a-z]|raising|seeking|funding)', "Traction Metrics (KPIs)": r'mrr|arr|cac|ltv|mau|daily active users|kpi|revenue|growth|metrics', "Milestones / Timeline": r'q[1-4] 202[4-9]|roadmap|timeline|launch|scale|milestones', "Team Credentials": r'ex-google|ex-meta|phd|serial entrepreneur|[0-9]+\+ years of experience|founder|cto|ceo', "Tech Stack / IP": r'ai|blockchain|patent|proprietary algorithm|ml model|technology|infrastructure', "Regulatory / Compliance": r'hipaa|gdpr|fda approval|iso certified|compliance|regulation', "Risk Disclosure": r'risk|challenge|headwind|burn rate|scalability issues|threats', "Financial Projections": r'revenue projection|financial forecast|p&l|unit economics|burn rate'}

# # # # --- ANALYSIS ENGINE & AI ANALYZER CLASSES (Existing - Unchanged) ---
# # # class AnalysisEngine:
# # #     # ... (Your existing AnalysisEngine class code is unchanged)
# # #     def __init__(self, models: Dict[str, Any]):
# # #         self.models = models
# # #     def perform_local_analysis(self, slides: List[str]) -> LocalAnalysisResult:
# # #         if not slides: raise ValueError("No slides available for analysis.")
# # #         embeddings = self.models['bert'].encode(slides)
# # #         sim_matrix = cosine_similarity(embeddings)
# # #         redundant_pairs = []
# # #         for i in range(len(slides)):
# # #             for j in range(i + 1, len(slides)):
# # #                 if sim_matrix[i][j] > 0.85: redundant_pairs.append({'slide1': i + 1, 'slide2': j + 1, 'similarity': float(sim_matrix[i][j])})
# # #         prob_sol_alignment = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]) if len(slides) > 1 else 0.0
# # #         local_metrics = [self._analyze_single_slide(text) for text in slides if text.strip()]
# # #         combined_text = " ".join(slides).lower()
# # #         checklist = self._perform_checklist_analysis(combined_text)
# # #         buzzword_count = self._count_buzzwords(combined_text)
# # #         buzzword_analysis = {"count": buzzword_count}
# # #         ner_insights = self._perform_ner_analysis(slides)
# # #         return LocalAnalysisResult(redundant_slides=redundant_pairs, local_metrics=[asdict(m) for m in local_metrics], prob_sol_alignment=prob_sol_alignment, checklist=checklist, buzzword_analysis=buzzword_analysis, ner_insights=ner_insights)
# # #     def _analyze_single_slide(self, slide_text: str) -> SlideAnalysis:
# # #         flesch_score = textstat.flesch_reading_ease(slide_text)
# # #         if flesch_score > 80: readability = "Very Easy"
# # #         elif flesch_score > 60: readability = "Easy"
# # #         elif flesch_score > 40: readability = "Standard"
# # #         else: readability = "Difficult"
# # #         quant_patterns = [r'\$\d[\d,.]*[kmb]?', r'\d[\d,.]*%', r'\d[\d,.]+\s?(?:million|billion|m|b)', r'\d+x', r'[0-9]+\+\s*(?:users|customers|clients)']
# # #         quantitative_count = sum(len(re.findall(p, slide_text, re.I)) for p in quant_patterns)
# # #         try:
# # #             sentiment_scores = self.models['sentiment'](slide_text[:512])[0]
# # #             sentiment = max(sentiment_scores, key=lambda x: x['score'])
# # #             sentiment['label'] = sentiment['label'].replace('LABEL_', '').upper()
# # #         except Exception as e:
# # #             logger.warning(f"Sentiment analysis failed: {e}")
# # #             sentiment = {'label': 'NEUTRAL', 'score': 0.0}
# # #         try:
# # #             entities = self.models['ner'](slide_text[:512])
# # #             named_entities = [{'entity': e['word'], 'type': e.get('entity_group', 'UNKNOWN')} for e in entities if e.get('score', 0) > 0.8]
# # #         except Exception: named_entities = []
# # #         return SlideAnalysis(readability, quantitative_count, sentiment, named_entities)
# # #     def _perform_checklist_analysis(self, text: str) -> Dict[str, bool]: return {item: bool(re.search(pattern, text, re.I)) for item, pattern in Config.CHECKLIST_PATTERNS.items()}
# # #     def _count_buzzwords(self, text: str) -> int: return sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text, re.I)) for word in Config.BUZZWORDS)
# # #     def _perform_ner_analysis(self, slides: List[str]) -> Dict[str, str]:
# # #         insights = {}
# # #         all_entities = []
# # #         for slide in slides:
# # #             if slide:
# # #                 try:
# # #                     entities = self.models['ner'](slide[:512])
# # #                     for entity in entities:
# # #                         if entity.get('score', 0) > 0.8: all_entities.append({'text': entity['word'].replace('##', ''), 'type': entity.get('entity_group', 'UNKNOWN'), 'confidence': entity.get('score', 0)})
# # #                 except Exception: continue
# # #         team_slide = self._find_slide_by_keywords(slides, ['team', 'founder', 'ceo', 'cto', 'management', 'leadership'])
# # #         if team_slide:
# # #             try:
# # #                 team_entities = self.models['ner'](team_slide[:512])
# # #                 people = [e for e in team_entities if e.get('entity_group') == 'PER' and e.get('score', 0) > 0.8]
# # #                 orgs = [e for e in team_entities if e.get('entity_group') == 'ORG' and e.get('score', 0) > 0.8]
# # #                 previous_companies = []
# # #                 for org in orgs:
# # #                     org_name = org['word'].replace('##', '')
# # #                     if any(keyword in team_slide.lower() for keyword in ['ex-', 'former', 'previously', 'worked at']): previous_companies.append(org_name)
# # #                 insights['team_summary'] = f"Found {len(people)} team members"
# # #                 if previous_companies: insights['team_summary'] += f" with experience at: {', '.join(previous_companies[:3])}"
# # #                 if orgs and not previous_companies: insights['team_summary'] += f" associated with {len(orgs)} organizations"
# # #             except Exception: insights['team_summary'] = "Team slide found but analysis failed"
# # #         else: insights['team_summary'] = "❌ No team slide detected - major red flag for investors"
# # #         comp_slide = self._find_slide_by_keywords(slides, ['competition', 'competitor', 'competitive', 'market', 'landscape'])
# # #         if comp_slide:
# # #             try:
# # #                 comp_entities = self.models['ner'](comp_slide[:512])
# # #                 competitors = [e for e in comp_entities if e.get('entity_group') == 'ORG' and e.get('score', 0) > 0.8]
# # #                 competitor_names = [e['word'].replace('##', '') for e in competitors]
# # #                 insights['competition_summary'] = f"Identified {len(competitors)} competitors: {', '.join(competitor_names[:5])}"
# # #                 if len(competitor_names) > 5: insights['competition_summary'] += f" and {len(competitor_names) - 5} more"
# # #             except Exception: insights['competition_summary'] = "Competition slide found but analysis failed"
# # #         else: insights['competition_summary'] = "⚠️ No competitive analysis slide - investors need to see market awareness"
# # #         financial_entities = []
# # #         money_pattern = r'\$[\d,.]+(M|B|K|million|billion|thousand)?'
# # #         for slide in slides:
# # #             if slide:
# # #                 money_mentions = re.findall(money_pattern, slide, re.IGNORECASE)
# # #                 financial_entities.extend(money_mentions)
# # #         insights['financial_summary'] = f"Found {len(financial_entities)} financial figures across all slides"
# # #         locations = [e for e in all_entities if e['type'] == 'LOC']
# # #         if locations:
# # #             location_names = list(set([e['text'] for e in locations]))
# # #             insights['market_geography'] = f"Geographic focus: {', '.join(location_names[:3])}"
# # #         else: insights['market_geography'] = "No clear geographic market focus identified"
# # #         return insights
# # #     def _find_slide_by_keywords(self, slides: List[str], keywords: List[str]) -> str:
# # #         for slide in slides:
# # #             if slide and any(keyword in slide.lower() for keyword in keywords): return slide
# # #         return ""

# # # class AIAnalyzer:
# # #     # ... (Your existing AIAnalyzer class code is unchanged)
# # #     def __init__(self):
# # #         self.ai_model = None
# # #         self.setup_ai_model()
# # #     def setup_ai_model(self):
# # #         try:
# # #             if os.getenv("GEMINI_API_KEY"):
# # #                 self.ai_model = genai.GenerativeModel("models/gemini-1.5-flash") # Corrected model name
# # #                 logger.info("Successfully configured AI model")
# # #             else: logger.error("GEMINI_API_KEY not found in environment variables")
# # #         except Exception as e: logger.error(f"Failed to configure AI model: {e}")
# # #     def perform_ai_analysis(self, slides: List[str], category: str, persona_desc: str, local_context: LocalAnalysisResult) -> AIAnalysisResult:
# # #         if not self.ai_model: raise ValueError("AI model is not available for analysis.")
# # #         context_json = json.dumps(asdict(local_context), cls=NumpyEncoder)
# # #         prompt = self._build_analysis_prompt(slides, category, persona_desc, context_json)
# # #         response = self.ai_model.generate_content(prompt)
# # #         return self._parse_ai_response(response.text)
# # #     def _build_analysis_prompt(self, slides: List[str], category: str, persona_desc: str, context_json: str) -> str:
# # #         return f"""As a venture capitalist acting as {persona_desc}, analyze this '{category}' pitch deck. CONTEXT FROM PRE-ANALYSIS: {context_json} FULL DECK TEXT: {json.dumps(slides[:10])} Provide your analysis in a single, valid JSON object with this exact structure: {{"dimensional_scores": {{"clarity": 0, "completeness": 0, "persuasiveness": 0, "data_driven": 0}}, "vc_verdict": "Your final, conclusive statement as this VC persona", "investment_readiness_score": 0, "competitive_moat": {{"score": 0, "feedback": "Detailed assessment of competitive advantages"}}, "gtm_strategy": {{"score": 0, "feedback": "Analysis of go-to-market approach"}}, "team_market_fit": {{"score": 0, "feedback": "Evaluation of team capabilities vs market needs"}}, "slide_by_slide_feedback": [{{"slide_number": 1, "clarity_feedback": "...", "tone_feedback": "...", "intent_fulfillment_feedback": "..."}}], "key_strengths": ["strength1", "strength2"], "key_improvement_suggestions": ["improvement1", "improvement2"]}} CRITICAL REQUIREMENTS: - The "vc_verdict" MUST be a single, conversational paragraph of approximately 80-85 words. - The "key_strengths" and "key_improvement_suggestions" MUST be arrays of simple strings. - DO NOT include any Markdown (like ** **, #,  ' ' , ` ` or *) inside any string values in the JSON. - All scores MUST be integers from 0-10. - Respond ONLY with the valid JSON object and nothing else."""
# # #     def _parse_ai_response(self, response_text: str) -> AIAnalysisResult:
# # #         try:
# # #             json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
# # #             json_str = json_match.group(1) if json_match else response_text[response_text.find('{'):response_text.rfind('}')+1]
# # #             parsed = json.loads(json_str)
# # #             return AIAnalysisResult(**parsed)
# # #         except (json.JSONDecodeError, TypeError) as e:
# # #             logger.error(f"Failed to parse AI response: {e}\nResponse: {response_text[:300]}")
# # #             raise ValueError("Invalid JSON response from AI.")


# # # # --- FastAPI Setup and Global Instances ---
# # # api_router = APIRouter()
# # # task_statuses: Dict[UUID, Dict[str, Any]] = {}
# # # db_manager = DatabaseManager()
# # # # --- FastAPI Imports ---
# # # from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
# # # from pathlib import Path # <--- ADD THIS IMPORT AT THE TOP OF THE FILE

# # # # ... other imports ...

# # # # --- FastAPI Setup and Global Instances ---
# # # api_router = APIRouter()
# # # task_statuses: Dict[UUID, Dict[str, Any]] = {}
# # # db_manager = DatabaseManager()

# # # # --- ADDED: Instantiate the Investor Matching Service (SIMPLE PATH) ---
# # # try:
# # #     # This path now looks for the CSV in the same directory where you run the server.
# # #     # Because we moved the file there in Step 1, this will now work.
# # #     DATASET_PATH = 'cleaned_sectors_dataset.csv'
    
# # #     if not os.path.exists(DATASET_PATH):
# # #         raise FileNotFoundError(f"CRITICAL: Investor dataset not found at {DATASET_PATH}")

# # #     matching_service = MatchingService(DATASET_PATH)
# # #     logger.info("Successfully loaded investor matching service.")

# # # except Exception as e:
# # #     logger.error(f"Failed to initialize investor matching service: {e}")
# # #     matching_service = None


# # # # --- Section 1: Pitch Deck Analysis Endpoints (Existing) ---

# # # def run_real_analysis(task_id: UUID, file_content: bytes, filename: str, file_type: str, category: str, persona_name: str):
# # #     # ... (Your existing run_real_analysis function is unchanged)
# # #     try:
# # #         db_manager.init_db()
# # #         clean_filename = unquote(filename)
# # #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 1/4: Extracting slides..."}
# # #         slides = FileProcessor.extract_slides(file_content, file_type)
# # #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 2/4: Performing local ML analysis..."}
# # #         engine = AnalysisEngine(models=ml_models)
# # #         local_analysis = engine.perform_local_analysis(slides)
# # #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 3/4: Getting insights from AI..."}
# # #         persona_desc = Config.VC_PERSONAS.get(persona_name, "a standard venture capitalist")
# # #         ai_analyzer = AIAnalyzer()
# # #         ai_analysis = ai_analyzer.perform_ai_analysis(slides, category, persona_desc, local_analysis)
# # #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 4/4: Saving analysis..."}
# # #         full_analysis = {**asdict(ai_analysis), **asdict(local_analysis), "slide_texts": slides, "timestamp": datetime.now().isoformat(), "analyzed_as_persona": persona_name, "category": category, "filename": clean_filename}
# # #         analysis_id = db_manager.save_analysis(clean_filename, category, persona_name, full_analysis)
# # #         task_statuses[task_id] = {"status": TaskStatus.COMPLETE, "analysis_id": analysis_id}
# # #         logger.info(f"Analysis complete for task {task_id}. Saved as ID {analysis_id}")
# # #     except Exception as e:
# # #         logger.error(f"Analysis failed for task {task_id}: {e}", exc_info=True)
# # #         task_statuses[task_id] = {"status": TaskStatus.FAILED, "message": "An internal error occurred. Please check server logs."}

# # # @api_router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
# # # async def analyze_deck(background_tasks: BackgroundTasks, file: UploadFile = File(...), category: str = Form(...), persona_name: str = Form(...)):
# # #     # ... (Your existing endpoint is unchanged)
# # #     task_id = uuid.uuid4()
# # #     task_statuses[task_id] = {"status": TaskStatus.PENDING}
# # #     file_content = await file.read()
# # #     background_tasks.add_task(run_real_analysis, task_id, file_content, file.filename, file.content_type, category, persona_name)
# # #     return AnalyzeResponse(task_id=task_id)

# # # @api_router.get("/status/{task_id}", response_model=StatusResponse)
# # # async def get_analysis_status(task_id: UUID):
# # #     # ... (Your existing endpoint is unchanged)
# # #     status_info = task_statuses.get(task_id)
# # #     if not status_info: raise HTTPException(status_code=404, detail="Task not found")
# # #     return StatusResponse(task_id=task_id, **status_info)

# # # @api_router.get("/results/{analysis_id}")
# # # async def get_analysis_results(analysis_id: int):
# # #     # ... (Your existing endpoint is unchanged)
# # #     results = db_manager.load_analysis(analysis_id)
# # #     if not results: raise HTTPException(status_code=404, detail="Analysis results not found")
# # #     return results

# # # @api_router.get("/history/{filename}")
# # # async def get_analysis_history(filename: str):
# # #     # ... (Your existing endpoint is unchanged)
# # #     logger.info(f"Fetching history for decoded filename: {filename}")
# # #     history = db_manager.get_history(filename)
# # #     if not history:
# # #         logger.warning(f"No history found in DB for filename: {filename}")
# # #         raise HTTPException(status_code=404, detail="No history found for this file") # Changed to 404 for clarity
# # #     processed_history = []
# # #     for record_row in history:
# # #         try:
# # #             record = dict(record_row)
# # #             analysis = json.loads(record['full_analysis'])
# # #             processed_history.append({"id": record['id'], "timestamp": record['timestamp'], "persona": record.get('persona', 'Unknown'), "investment_readiness_score": analysis.get('investment_readiness_score', 0), "competitive_moat_score": analysis.get('competitive_moat', {}).get('score', 0), "gtm_strategy_score": analysis.get('gtm_strategy', {}).get('score', 0)})
# # #         except (json.JSONDecodeError, KeyError) as e:
# # #             logger.error(f"Could not process history record {record.get('id', 'N/A')}: {e}")
# # #             continue
# # #     logger.info(f"Found {len(processed_history)} history records for {filename}")
# # #     return processed_history


# # # # --- Section 2: Real-time Investor Matching Endpoint (New) ---

# # # @api_router.post("/find-matches", response_model=MatchResponse)
# # # async def find_matches(request: MatchRequest):
# # #     """
# # #     Finds top investor matches based on a startup description.
# # #     This is a synchronous, real-time endpoint.
# # #     """
# # #     if not matching_service:
# # #         raise HTTPException(status_code=503, detail="Investor Matching Service is currently unavailable.")
    
# # #     if not request.description or len(request.description.strip()) < 10:
# # #         raise HTTPException(status_code=400, detail="Please provide a more detailed startup description.")

# # #     try:
# # #         # 1. Extract keywords and industries
# # #         keywords = matching_service.extract_keywords(request.description)
# # #         industries = matching_service.find_similar_industries(keywords)
        
# # #         # 2. Find and score top investors
# # #         top_firms = matching_service.find_top_investors(request.stage, keywords, request.investor_type)
        
# # #         # 3. Generate actionable insights
# # #         insights = matching_service.generate_insights(top_firms)
        
# # #         # 4. Format the investor data for the response schema
# # #         investor_list = [
# # #             Investor(
# # #                 investor_name=row['Investor Name'],
# # #                 website=row.get('Website', 'N/A'),
# # #                 fund_type=row['Fund Type'],
# # #                 fund_stage=row['Fund Stage'],
# # #                 focus_areas=[f.strip() for f in str(row['Fund Focus (Sectors)']).split(',') if f.strip()],
# # #                 match_score=int(row['score'])
# # #             ) for _, row in top_firms.iterrows()
# # #         ]
        
# # #         return MatchResponse(
# # #             extracted_keywords=keywords,
# # #             matching_industries=industries,
# # #             insights=insights,
# # #             investors=investor_list
# # #         )
# # #     except Exception as e:
# # #         logger.error(f"Error during investor matching: {e}", exc_info=True)
# # #         raise HTTPException(status_code=500, detail="An internal error occurred while finding matches.")








# # # vc-copilot-backend/app/api/endpoints.py
# # # CORRECTED AND UNIFIED VERSION (with new evaluation endpoint added)

# # import uuid
# # import json
# # import re
# # import os
# # import logging
# # from datetime import datetime
# # from uuid import UUID
# # from typing import Dict, List, Any
# # from urllib.parse import unquote

# # # Third-party imports
# # import textstat
# # import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity
# # import google.generativeai as genai
# # from dotenv import load_dotenv
# # from dataclasses import dataclass, asdict

# # # FastAPI imports
# # from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException

# # # Local app imports
# # from app.core.ml import ml_models
# # from app.db.database import DatabaseManager, NumpyEncoder
# # from app.services.file_processor import FileProcessor
# # from app.services.matching_service import MatchingService
# # # --- ADDED: Import for the new evaluation service ---
# # from app.services.evaluation_service import evaluation_service

# # # --- MODIFIED: Import all necessary schemas, including new ones ---
# # from app.models.schemas import (
# #     AnalyzeResponse, 
# #     StatusResponse, 
# #     TaskStatus,
# #     MatchRequest,
# #     MatchResponse,
# #     Investor,
# #     EvaluationRequest,  # Added
# #     EvaluationResponse  # Added
# # )


# # # --- Basic Setup ---
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)
# # load_dotenv()
# # try:
# #     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# # except Exception as e:
# #     logger.error(f"AI model configuration failed: {e}")

# # # --- DATA CLASSES AND CONFIG (Existing - Unchanged) ---
# # @dataclass
# # class SlideAnalysis:
# #     readability_interpretation: str; quantitative_count: int; sentiment: Dict[str, Any]; named_entities: List[Dict[str, str]]

# # @dataclass
# # class LocalAnalysisResult:
# #     redundant_slides: List[Dict[str, Any]]; local_metrics: List[Any]; prob_sol_alignment: float; checklist: Dict[str, bool]; buzzword_analysis: Dict[str, Any]; ner_insights: Dict[str, Any]

# # @dataclass
# # class AIAnalysisResult:
# #     dimensional_scores: Dict[str, float]; vc_verdict: str; investment_readiness_score: float; competitive_moat: Dict[str, Any]; gtm_strategy: Dict[str, Any]; team_market_fit: Dict[str, Any]; slide_by_slide_feedback: List[Dict[str, str]]; key_strengths: List[str]; key_improvement_suggestions: List[str]

# # class Config:
# #     VC_PERSONAS = {"The Friendly Mentor": "a friendly, encouraging mentor who focuses on nurturing potential", "The Data-Driven Analyst": "a sharp, data-driven analyst who demands concrete metrics and evidence", "The 'Big Picture' Strategist": "a 'big picture' strategist who evaluates market timing and scalability", "The Devil's Advocate": "a devil's advocate who challenges assumptions and identifies risks"}
# #     BUZZWORDS = ['synergy', 'disrupt', 'paradigm shift', 'game-changer', 'revolutionize', 'innovative', 'cutting-edge', 'breakthrough', 'revolutionary']
# #     CHECKLIST_PATTERNS = {"TAM/SAM/SOM": r'tam|sam|som|total addressable market|market size', "Go-to-Market Strategy": r'go-to-market|gtm|target audience|distribution channels|marketing plan|customer acquisition', "Funding Ask": r'(\$[0-9,.]+m? seed|\$[0-9,.]+m? series [a-z]|raising|seeking|funding)', "Traction Metrics (KPIs)": r'mrr|arr|cac|ltv|mau|daily active users|kpi|revenue|growth|metrics', "Milestones / Timeline": r'q[1-4] 202[4-9]|roadmap|timeline|launch|scale|milestones', "Team Credentials": r'ex-google|ex-meta|phd|serial entrepreneur|[0-9]+\+ years of experience|founder|cto|ceo', "Tech Stack / IP": r'ai|blockchain|patent|proprietary algorithm|ml model|technology|infrastructure', "Regulatory / Compliance": r'hipaa|gdpr|fda approval|iso certified|compliance|regulation', "Risk Disclosure": r'risk|challenge|headwind|burn rate|scalability issues|threats', "Financial Projections": r'revenue projection|financial forecast|p&l|unit economics|burn rate'}

# # # --- ANALYSIS ENGINE & AI ANALYZER CLASSES (Existing - Unchanged) ---
# # class AnalysisEngine:
# #     def __init__(self, models: Dict[str, Any]):
# #         self.models = models
# #     def perform_local_analysis(self, slides: List[str]) -> LocalAnalysisResult:
# #         if not slides: raise ValueError("No slides available for analysis.")
# #         embeddings = self.models['bert'].encode(slides)
# #         sim_matrix = cosine_similarity(embeddings)
# #         redundant_pairs = []
# #         for i in range(len(slides)):
# #             for j in range(i + 1, len(slides)):
# #                 if sim_matrix[i][j] > 0.85: redundant_pairs.append({'slide1': i + 1, 'slide2': j + 1, 'similarity': float(sim_matrix[i][j])})
# #         prob_sol_alignment = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]) if len(slides) > 1 else 0.0
# #         local_metrics = [self._analyze_single_slide(text) for text in slides if text.strip()]
# #         combined_text = " ".join(slides).lower()
# #         checklist = self._perform_checklist_analysis(combined_text)
# #         buzzword_count = self._count_buzzwords(combined_text)
# #         buzzword_analysis = {"count": buzzword_count}
# #         ner_insights = self._perform_ner_analysis(slides)
# #         return LocalAnalysisResult(redundant_slides=redundant_pairs, local_metrics=[asdict(m) for m in local_metrics], prob_sol_alignment=prob_sol_alignment, checklist=checklist, buzzword_analysis=buzzword_analysis, ner_insights=ner_insights)
# #     def _analyze_single_slide(self, slide_text: str) -> SlideAnalysis:
# #         flesch_score = textstat.flesch_reading_ease(slide_text)
# #         if flesch_score > 80: readability = "Very Easy"
# #         elif flesch_score > 60: readability = "Easy"
# #         elif flesch_score > 40: readability = "Standard"
# #         else: readability = "Difficult"
# #         quant_patterns = [r'\$\d[\d,.]*[kmb]?', r'\d[\d,.]*%', r'\d[\d,.]+\s?(?:million|billion|m|b)', r'\d+x', r'[0-9]+\+\s*(?:users|customers|clients)']
# #         quantitative_count = sum(len(re.findall(p, slide_text, re.I)) for p in quant_patterns)
# #         try:
# #             sentiment_scores = self.models['sentiment'](slide_text[:512])[0]
# #             sentiment = max(sentiment_scores, key=lambda x: x['score'])
# #             sentiment['label'] = sentiment['label'].replace('LABEL_', '').upper()
# #         except Exception as e:
# #             logger.warning(f"Sentiment analysis failed: {e}")
# #             sentiment = {'label': 'NEUTRAL', 'score': 0.0}
# #         try:
# #             entities = self.models['ner'](slide_text[:512])
# #             named_entities = [{'entity': e['word'], 'type': e.get('entity_group', 'UNKNOWN')} for e in entities if e.get('score', 0) > 0.8]
# #         except Exception: named_entities = []
# #         return SlideAnalysis(readability, quantitative_count, sentiment, named_entities)
# #     def _perform_checklist_analysis(self, text: str) -> Dict[str, bool]: return {item: bool(re.search(pattern, text, re.I)) for item, pattern in Config.CHECKLIST_PATTERNS.items()}
# #     def _count_buzzwords(self, text: str) -> int: return sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text, re.I)) for word in Config.BUZZWORDS)
# #     def _perform_ner_analysis(self, slides: List[str]) -> Dict[str, str]:
# #         insights = {}
# #         all_entities = []
# #         for slide in slides:
# #             if slide:
# #                 try:
# #                     entities = self.models['ner'](slide[:512])
# #                     for entity in entities:
# #                         if entity.get('score', 0) > 0.8: all_entities.append({'text': entity['word'].replace('##', ''), 'type': entity.get('entity_group', 'UNKNOWN'), 'confidence': entity.get('score', 0)})
# #                 except Exception: continue
# #         team_slide = self._find_slide_by_keywords(slides, ['team', 'founder', 'ceo', 'cto', 'management', 'leadership'])
# #         if team_slide:
# #             try:
# #                 team_entities = self.models['ner'](team_slide[:512])
# #                 people = [e for e in team_entities if e.get('entity_group') == 'PER' and e.get('score', 0) > 0.8]
# #                 orgs = [e for e in team_entities if e.get('entity_group') == 'ORG' and e.get('score', 0) > 0.8]
# #                 previous_companies = []
# #                 for org in orgs:
# #                     org_name = org['word'].replace('##', '')
# #                     if any(keyword in team_slide.lower() for keyword in ['ex-', 'former', 'previously', 'worked at']): previous_companies.append(org_name)
# #                 insights['team_summary'] = f"Found {len(people)} team members"
# #                 if previous_companies: insights['team_summary'] += f" with experience at: {', '.join(previous_companies[:3])}"
# #                 if orgs and not previous_companies: insights['team_summary'] += f" associated with {len(orgs)} organizations"
# #             except Exception: insights['team_summary'] = "Team slide found but analysis failed"
# #         else: insights['team_summary'] = "❌ No team slide detected - major red flag for investors"
# #         comp_slide = self._find_slide_by_keywords(slides, ['competition', 'competitor', 'competitive', 'market', 'landscape'])
# #         if comp_slide:
# #             try:
# #                 comp_entities = self.models['ner'](comp_slide[:512])
# #                 competitors = [e for e in comp_entities if e.get('entity_group') == 'ORG' and e.get('score', 0) > 0.8]
# #                 competitor_names = [e['word'].replace('##', '') for e in competitors]
# #                 insights['competition_summary'] = f"Identified {len(competitors)} competitors: {', '.join(competitor_names[:5])}"
# #                 if len(competitor_names) > 5: insights['competition_summary'] += f" and {len(competitor_names) - 5} more"
# #             except Exception: insights['competition_summary'] = "Competition slide found but analysis failed"
# #         else: insights['competition_summary'] = "⚠️ No competitive analysis slide - investors need to see market awareness"
# #         financial_entities = []
# #         money_pattern = r'\$[\d,.]+(M|B|K|million|billion|thousand)?'
# #         for slide in slides:
# #             if slide:
# #                 money_mentions = re.findall(money_pattern, slide, re.IGNORECASE)
# #                 financial_entities.extend(money_mentions)
# #         insights['financial_summary'] = f"Found {len(financial_entities)} financial figures across all slides"
# #         locations = [e for e in all_entities if e['type'] == 'LOC']
# #         if locations:
# #             location_names = list(set([e['text'] for e in locations]))
# #             insights['market_geography'] = f"Geographic focus: {', '.join(location_names[:3])}"
# #         else: insights['market_geography'] = "No clear geographic market focus identified"
# #         return insights
# #     def _find_slide_by_keywords(self, slides: List[str], keywords: List[str]) -> str:
# #         for slide in slides:
# #             if slide and any(keyword in slide.lower() for keyword in keywords): return slide
# #         return ""

# # class AIAnalyzer:
# #     def __init__(self):
# #         self.ai_model = None
# #         self.setup_ai_model()
# #     def setup_ai_model(self):
# #         try:
# #             if os.getenv("GEMINI_API_KEY"):
# #                 self.ai_model = genai.GenerativeModel("models/gemini-2.5-flash")
# #                 logger.info("Successfully configured AI model")
# #             else: logger.error("GEMINI_API_KEY not found in environment variables")
# #         except Exception as e: logger.error(f"Failed to configure AI model: {e}")
# #     def perform_ai_analysis(self, slides: List[str], category: str, persona_desc: str, local_context: LocalAnalysisResult) -> AIAnalysisResult:
# #         if not self.ai_model: raise ValueError("AI model is not available for analysis.")
# #         context_json = json.dumps(asdict(local_context), cls=NumpyEncoder)
# #         prompt = self._build_analysis_prompt(slides, category, persona_desc, context_json)
# #         response = self.ai_model.generate_content(prompt)
# #         return self._parse_ai_response(response.text)
# #     def _build_analysis_prompt(self, slides: List[str], category: str, persona_desc: str, context_json: str) -> str:
# #         return f"""As a venture capitalist acting as {persona_desc}, analyze this '{category}' pitch deck. CONTEXT FROM PRE-ANALYSIS: {context_json} FULL DECK TEXT: {json.dumps(slides[:10])} Provide your analysis in a single, valid JSON object with this exact structure: {{"dimensional_scores": {{"clarity": 0, "completeness": 0, "persuasiveness": 0, "data_driven": 0}}, "vc_verdict": "Your final, conclusive statement as this VC persona", "investment_readiness_score": 0, "competitive_moat": {{"score": 0, "feedback": "Detailed assessment of competitive advantages"}}, "gtm_strategy": {{"score": 0, "feedback": "Analysis of go-to-market approach"}}, "team_market_fit": {{"score": 0, "feedback": "Evaluation of team capabilities vs market needs"}}, "slide_by_slide_feedback": [{{"slide_number": 1, "clarity_feedback": "...", "tone_feedback": "...", "intent_fulfillment_feedback": "..."}}], "key_strengths": ["strength1", "strength2"], "key_improvement_suggestions": ["improvement1", "improvement2"]}} CRITICAL REQUIREMENTS: - The "vc_verdict" MUST be a single, conversational paragraph of approximately 80-85 words. - The "key_strengths" and "key_improvement_suggestions" MUST be arrays of simple strings. - DO NOT include any Markdown (like ** **, #,  ' ' , ` ` or *) inside any string values in the JSON. - All scores MUST be integers from 0-10. - Respond ONLY with the valid JSON object and nothing else."""
# #     def _parse_ai_response(self, response_text: str) -> AIAnalysisResult:
# #         try:
# #             json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
# #             json_str = json_match.group(1) if json_match else response_text[response_text.find('{'):response_text.rfind('}')+1]
# #             parsed = json.loads(json_str)
# #             return AIAnalysisResult(**parsed)
# #         except (json.JSONDecodeError, TypeError) as e:
# #             logger.error(f"Failed to parse AI response: {e}\nResponse: {response_text[:300]}")
# #             raise ValueError("Invalid JSON response from AI.")


# # # --- FastAPI Setup and Global Instances ---
# # api_router = APIRouter()
# # task_statuses: Dict[UUID, Dict[str, Any]] = {}
# # db_manager = DatabaseManager()

# # try:
# #     DATASET_PATH = 'cleaned_sectors_dataset.csv'
# #     if not os.path.exists(DATASET_PATH):
# #         raise FileNotFoundError(f"CRITICAL: Investor dataset not found at {DATASET_PATH}")
# #     matching_service = MatchingService(DATASET_PATH)
# #     logger.info("Successfully loaded investor matching service.")
# # except Exception as e:
# #     logger.error(f"Failed to initialize investor matching service: {e}")
# #     matching_service = None


# # # --- Section 1: Pitch Deck Analysis Endpoints (Existing) ---

# # def run_real_analysis(task_id: UUID, file_content: bytes, filename: str, file_type: str, category: str, persona_name: str):
# #     try:
# #         db_manager.init_db()
# #         clean_filename = unquote(filename)
# #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 1/4: Extracting slides..."}
# #         slides = FileProcessor.extract_slides(file_content, file_type)
# #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 2/4: Performing local ML analysis..."}
# #         engine = AnalysisEngine(models=ml_models)
# #         local_analysis = engine.perform_local_analysis(slides)
# #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 3/4: Getting insights from AI..."}
# #         persona_desc = Config.VC_PERSONAS.get(persona_name, "a standard venture capitalist")
# #         ai_analyzer = AIAnalyzer()
# #         ai_analysis = ai_analyzer.perform_ai_analysis(slides, category, persona_desc, local_analysis)
# #         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 4/4: Saving analysis..."}
# #         full_analysis = {**asdict(ai_analysis), **asdict(local_analysis), "slide_texts": slides, "timestamp": datetime.now().isoformat(), "analyzed_as_persona": persona_name, "category": category, "filename": clean_filename}
# #         analysis_id = db_manager.save_analysis(clean_filename, category, persona_name, full_analysis)
# #         task_statuses[task_id] = {"status": TaskStatus.COMPLETE, "analysis_id": analysis_id}
# #         logger.info(f"Analysis complete for task {task_id}. Saved as ID {analysis_id}")
# #     except Exception as e:
# #         logger.error(f"Analysis failed for task {task_id}: {e}", exc_info=True)
# #         task_statuses[task_id] = {"status": TaskStatus.FAILED, "message": "An internal error occurred. Please check server logs."}

# # @api_router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
# # async def analyze_deck(background_tasks: BackgroundTasks, file: UploadFile = File(...), category: str = Form(...), persona_name: str = Form(...)):
# #     task_id = uuid.uuid4()
# #     task_statuses[task_id] = {"status": TaskStatus.PENDING}
# #     file_content = await file.read()
# #     background_tasks.add_task(run_real_analysis, task_id, file_content, file.filename, file.content_type, category, persona_name)
# #     return AnalyzeResponse(task_id=task_id)

# # @api_router.get("/status/{task_id}", response_model=StatusResponse)
# # async def get_analysis_status(task_id: UUID):
# #     status_info = task_statuses.get(task_id)
# #     if not status_info: raise HTTPException(status_code=404, detail="Task not found")
# #     return StatusResponse(task_id=task_id, **status_info)

# # @api_router.get("/results/{analysis_id}")
# # async def get_analysis_results(analysis_id: int):
# #     results = db_manager.load_analysis(analysis_id)
# #     if not results: raise HTTPException(status_code=404, detail="Analysis results not found")
# #     return results

# # @api_router.get("/history/{filename}")
# # async def get_analysis_history(filename: str):
# #     logger.info(f"Fetching history for decoded filename: {filename}")
# #     history = db_manager.get_history(filename)
# #     if not history:
# #         logger.warning(f"No history found in DB for filename: {filename}")
# #         raise HTTPException(status_code=404, detail="No history found for this file")
# #     processed_history = []
# #     for record_row in history:
# #         try:
# #             record = dict(record_row)
# #             analysis = json.loads(record['full_analysis'])
# #             processed_history.append({"id": record['id'], "timestamp": record['timestamp'], "persona": record.get('persona', 'Unknown'), "investment_readiness_score": analysis.get('investment_readiness_score', 0), "competitive_moat_score": analysis.get('competitive_moat', {}).get('score', 0), "gtm_strategy_score": analysis.get('gtm_strategy', {}).get('score', 0)})
# #         except (json.JSONDecodeError, KeyError) as e:
# #             logger.error(f"Could not process history record {record.get('id', 'N/A')}: {e}")
# #             continue
# #     logger.info(f"Found {len(processed_history)} history records for {filename}")
# #     return processed_history


# # # --- Section 2: Real-time Investor Matching Endpoint (Existing) ---

# # @api_router.post("/find-matches", response_model=MatchResponse)
# # async def find_matches(request: MatchRequest):
# #     if not matching_service:
# #         raise HTTPException(status_code=503, detail="Investor Matching Service is currently unavailable.")
# #     if not request.description or len(request.description.strip()) < 10:
# #         raise HTTPException(status_code=400, detail="Please provide a more detailed startup description.")
# #     try:
# #         keywords = matching_service.extract_keywords(request.description)
# #         industries = matching_service.find_similar_industries(keywords)
# #         top_firms = matching_service.find_top_investors(request.stage, keywords, request.investor_type)
# #         insights = matching_service.generate_insights(top_firms)
# #         investor_list = [
# #             Investor(
# #                 investor_name=row['Investor Name'],
# #                 website=row.get('Website', 'N/A'),
# #                 fund_type=row['Fund Type'],
# #                 fund_stage=row['Fund Stage'],
# #                 focus_areas=[f.strip() for f in str(row['Fund Focus (Sectors)']).split(',') if f.strip()],
# #                 match_score=int(row['score'])
# #             ) for _, row in top_firms.iterrows()
# #         ]
# #         return MatchResponse(
# #             extracted_keywords=keywords,
# #             matching_industries=industries,
# #             insights=insights,
# #             investors=investor_list
# #         )
# #     except Exception as e:
# #         logger.error(f"Error during investor matching: {e}", exc_info=True)
# #         raise HTTPException(status_code=500, detail="An internal error occurred while finding matches.")


# # # --- Section 3: 7-Domain Startup Evaluation (New) ---

# # @api_router.post("/evaluate", response_model=EvaluationResponse)
# # async def evaluate_startup(request: EvaluationRequest):
# #     """
# #     Performs a comprehensive 7-domain evaluation of a startup idea using
# #     RAG and live web search.
# #     """
# #     if not request.startup_idea or len(request.startup_idea.strip()) < 20:
# #         raise HTTPException(status_code=400, detail="Please provide a detailed startup idea for a meaningful evaluation.")

# #     try:
# #         logger.info(f"Received evaluation request for startup: '{request.startup_name or 'Unnamed'}'")
# #         # Call the evaluation service with the data from the request body
# #         evaluation_result = evaluation_service.comprehensive_startup_evaluation(
# #             startup_idea=request.startup_idea,
# #             startup_name=request.startup_name,
# #             uploaded_content=request.uploaded_content
# #         )
        
# #         # The service might return an error dictionary
# #         if "error" in evaluation_result:
# #             logger.error(f"Evaluation service returned an error: {evaluation_result['error']}")
# #             raise HTTPException(status_code=500, detail=evaluation_result["error"])
        
# #         return EvaluationResponse(evaluation=evaluation_result)

# #     except Exception as e:
# #         logger.error(f"An unexpected error occurred in the /evaluate endpoint: {e}", exc_info=True)
# #         # Re-raise as HTTPException to be handled by FastAPI
# #         if not isinstance(e, HTTPException):
# #             raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
# #         raise e


# # vc-copilot-backend/app/api/endpoints.py
# # CORRECTED AND UNIFIED VERSION (with Success Predictor endpoints added)

# import uuid
# import json
# import re
# import os
# import logging
# from datetime import datetime
# from uuid import UUID
# from typing import Dict, List, Any
# from urllib.parse import unquote

# # Third-party imports
# import textstat
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import google.generativeai as genai
# from dotenv import load_dotenv
# from dataclasses import dataclass, asdict
# import logging

# # Set up logging
# logger = logging.getLogger(__name__)

# # FastAPI imports
# from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
# from pydantic import BaseModel # <-- ADDED: For Success Predictor Input

# # Local app imports
# from app.core.ml import ml_models
# from app.db.database import DatabaseManager, NumpyEncoder
# from app.services.file_processor import FileProcessor
# from app.services.matching_service import MatchingService
# from app.services.evaluation_service import evaluation_service

# # --- START: NEW IMPORTS & UTILS FOR SUCCESS PREDICTOR ---
# import joblib 
# import pandas as pd 
# import string 

# # Pydantic model for the new endpoints (Copied from open source main.py)
# class SuccessInput(BaseModel): 
#     text: str
#     goal: float = 0
#     usd_pledged: float = 0
#     usd_pledged_real: float = 0
#     usd_goal_real: float = 0
#     pledged: float = 0
#     backers: int = 0

# # Text cleaning function (Copied from open source trainmodel.py)
# def clean_text_basic(text: str) -> str: 
#     text = str(text).lower()
#     text = re.sub(r"http\S+", " ", text)
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     text = re.sub(r"\d+", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# # Feature list (Copied from open source trainmodel.py)
# numeric_features = ["goal", "usd_pledged", "usd_pledged_real", "usd_goal_real", "pledged", "backers"] 

# # Load model and utility functions globally
# try: 
#     # Assumes pitchmodel.pkl and trainmodel.py were moved to the backend root
#     model = joblib.load("pitchmodel.pkl") 
#     from trainmodel import get_shap_explanation, generate_narrative_summary
#     logger.info("Successfully loaded pitchmodel.pkl and trainmodel utilities.")
# except FileNotFoundError: 
#     model = None
#     logger.error("pitchmodel.pkl or trainmodel.py not found in the backend root.")
#     # Dummy functions to prevent errors if files are missing
#     def get_shap_explanation(*args, **kwargs): return None, [], 0 
#     def generate_narrative_summary(*args, **kwargs): return "Prediction model or utility files are missing on the server."
# except ImportError: 
#     model = None
#     logger.error("Could not import utility functions from trainmodel.py.")
#     def get_shap_explanation(*args, **kwargs): return None, [], 0 
#     def generate_narrative_summary(*args, **kwargs): return "Prediction model or utility files are missing on the server."

# # --- END: NEW IMPORTS & UTILS FOR SUCCESS PREDICTOR ---

# # --- MODIFIED: Import all necessary schemas, including new ones ---
# from app.models.schemas import (
#     AnalyzeResponse, 
#     StatusResponse, 
#     TaskStatus,
#     MatchRequest,
#     MatchResponse,
#     Investor,
#     EvaluationRequest,
#     EvaluationResponse
# )


# # --- Basic Setup (Existing) ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# load_dotenv()
# try:
#     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# except Exception as e:
#     logger.error(f"AI model configuration failed: {e}")

# # --- DATA CLASSES AND CONFIG (Existing - Unchanged) ---
# @dataclass
# class SlideAnalysis:
#     readability_interpretation: str; quantitative_count: int; sentiment: Dict[str, Any]; named_entities: List[Dict[str, str]]

# @dataclass
# class LocalAnalysisResult:
#     redundant_slides: List[Dict[str, Any]]; local_metrics: List[Any]; prob_sol_alignment: float; checklist: Dict[str, bool]; buzzword_analysis: Dict[str, Any]; ner_insights: Dict[str, Any]

# @dataclass
# class AIAnalysisResult:
#     dimensional_scores: Dict[str, float]; vc_verdict: str; investment_readiness_score: float; competitive_moat: Dict[str, Any]; gtm_strategy: Dict[str, Any]; team_market_fit: Dict[str, Any]; slide_by_slide_feedback: List[Dict[str, str]]; key_strengths: List[str]; key_improvement_suggestions: List[str]

# class Config:
#     VC_PERSONAS = {"The Friendly Mentor": "a friendly, encouraging mentor who focuses on nurturing potential", "The Data-Driven Analyst": "a sharp, data-driven analyst who demands concrete metrics and evidence", "The 'Big Picture' Strategist": "a 'big picture' strategist who evaluates market timing and scalability", "The Devil's Advocate": "a devil's advocate who challenges assumptions and identifies risks"}
#     BUZZWORDS = ['synergy', 'disrupt', 'paradigm shift', 'game-changer', 'revolutionize', 'innovative', 'cutting-edge', 'breakthrough', 'revolutionary']
#     CHECKLIST_PATTERNS = {"TAM/SAM/SOM": r'tam|sam|som|total addressable market|market size', "Go-to-Market Strategy": r'go-to-market|gtm|target audience|distribution channels|marketing plan|customer acquisition', "Funding Ask": r'(\$[0-9,.]+m? seed|\$[0-9,.]+m? series [a-z]|raising|seeking|funding)', "Traction Metrics (KPIs)": r'mrr|arr|cac|ltv|mau|daily active users|kpi|revenue|growth|metrics', "Milestones / Timeline": r'q[1-4] 202[4-9]|roadmap|timeline|launch|scale|milestones', "Team Credentials": r'ex-google|ex-meta|phd|serial entrepreneur|[0-9]+\+ years of experience|founder|cto|ceo', "Tech Stack / IP": r'ai|blockchain|patent|proprietary algorithm|ml model|technology|infrastructure', "Regulatory / Compliance": r'hipaa|gdpr|fda approval|iso certified|compliance|regulation', "Risk Disclosure": r'risk|challenge|headwind|burn rate|scalability issues|threats', "Financial Projections": r'revenue projection|financial forecast|p&l|unit economics|burn rate'}

# # --- ANALYSIS ENGINE & AI ANALYZER CLASSES (Existing - Unchanged) ---
# class AnalysisEngine:
#     def __init__(self, models: Dict[str, Any]):
#         self.models = models
#     def perform_local_analysis(self, slides: List[str]) -> LocalAnalysisResult:
#         if not slides: raise ValueError("No slides available for analysis.")
#         embeddings = self.models['bert'].encode(slides)
#         sim_matrix = cosine_similarity(embeddings)
#         redundant_pairs = []
#         for i in range(len(slides)):
#             for j in range(i + 1, len(slides)):
#                 if sim_matrix[i][j] > 0.85: redundant_pairs.append({'slide1': i + 1, 'slide2': j + 1, 'similarity': float(sim_matrix[i][j])})
#         prob_sol_alignment = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]) if len(slides) > 1 else 0.0
#         local_metrics = [self._analyze_single_slide(text) for text in slides if text.strip()]
#         combined_text = " ".join(slides).lower()
#         checklist = self._perform_checklist_analysis(combined_text)
#         buzzword_count = self._count_buzzwords(combined_text)
#         buzzword_analysis = {"count": buzzword_count}
#         ner_insights = self._perform_ner_analysis(slides)
#         return LocalAnalysisResult(redundant_slides=redundant_pairs, local_metrics=[asdict(m) for m in local_metrics], prob_sol_alignment=prob_sol_alignment, checklist=checklist, buzzword_analysis=buzzword_analysis, ner_insights=ner_insights)
#     def _analyze_single_slide(self, slide_text: str) -> SlideAnalysis:
#         flesch_score = textstat.flesch_reading_ease(slide_text)
#         if flesch_score > 80: readability = "Very Easy"
#         elif flesch_score > 60: readability = "Easy"
#         elif flesch_score > 40: readability = "Standard"
#         else: readability = "Difficult"
#         quant_patterns = [r'\$\d[\d,.]*[kmb]?', r'\d[\d,.]*%', r'\d[\d,.]+\s?(?:million|billion|m|b)', r'\d+x', r'[0-9]+\+\s*(?:users|customers|clients)']
#         quantitative_count = sum(len(re.findall(p, slide_text, re.I)) for p in quant_patterns)
#         try:
#             sentiment_scores = self.models['sentiment'](slide_text[:512])[0]
#             sentiment = max(sentiment_scores, key=lambda x: x['score'])
#             sentiment['label'] = sentiment['label'].replace('LABEL_', '').upper()
#         except Exception as e:
#             logger.warning(f"Sentiment analysis failed: {e}")
#             sentiment = {'label': 'NEUTRAL', 'score': 0.0}
#         try:
#             entities = self.models['ner'](slide_text[:512])
#             named_entities = [{'entity': e['word'], 'type': e.get('entity_group', 'UNKNOWN')} for e in entities if e.get('score', 0) > 0.8]
#         except Exception: named_entities = []
#         return SlideAnalysis(readability, quantitative_count, sentiment, named_entities)
#     def _perform_checklist_analysis(self, text: str) -> Dict[str, bool]: return {item: bool(re.search(pattern, text, re.I)) for item, pattern in Config.CHECKLIST_PATTERNS.items()}
#     def _count_buzzwords(self, text: str) -> int: return sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text, re.I)) for word in Config.BUZZWORDS)
#     def _perform_ner_analysis(self, slides: List[str]) -> Dict[str, str]:
#         insights = {}
#         all_entities = []
#         for slide in slides:
#             if slide:
#                 try:
#                     entities = self.models['ner'](slide[:512])
#                     for entity in entities:
#                         if entity.get('score', 0) > 0.8: all_entities.append({'text': entity['word'].replace('##', ''), 'type': entity.get('entity_group', 'UNKNOWN'), 'confidence': entity.get('score', 0)})
#                 except Exception: continue
#         team_slide = self._find_slide_by_keywords(slides, ['team', 'founder', 'ceo', 'cto', 'management', 'leadership'])
#         if team_slide:
#             try:
#                 team_entities = self.models['ner'](team_slide[:512])
#                 people = [e for e in team_entities if e.get('entity_group') == 'PER' and e.get('score', 0) > 0.8]
#                 orgs = [e for e in team_entities if e.get('entity_group') == 'ORG' and e.get('score', 0) > 0.8]
#                 previous_companies = []
#                 for org in orgs:
#                     org_name = org['word'].replace('##', '')
#                     if any(keyword in team_slide.lower() for keyword in ['ex-', 'former', 'previously', 'worked at']): previous_companies.append(org_name)
#                 insights['team_summary'] = f"Found {len(people)} team members"
#                 if previous_companies: insights['team_summary'] += f" with experience at: {', '.join(previous_companies[:3])}"
#                 if orgs and not previous_companies: insights['team_summary'] += f" associated with {len(orgs)} organizations"
#             except Exception: insights['team_summary'] = "Team slide found but analysis failed"
#         else: insights['team_summary'] = "❌ No team slide detected - major red flag for investors"
#         comp_slide = self._find_slide_by_keywords(slides, ['competition', 'competitor', 'competitive', 'market', 'landscape'])
#         if comp_slide:
#             try:
#                 comp_entities = self.models['ner'](comp_slide[:512])
#                 competitors = [e for e in comp_entities if e.get('entity_group') == 'ORG' and e.get('score', 0) > 0.8]
#                 competitor_names = [e['word'].replace('##', '') for e in competitors]
#                 insights['competition_summary'] = f"Identified {len(competitors)} competitors: {', '.join(competitor_names[:5])}"
#                 if len(competitor_names) > 5: insights['competition_summary'] += f" and {len(competitor_names) - 5} more"
#             except Exception: insights['competition_summary'] = "Competition slide found but analysis failed"
#         else: insights['competition_summary'] = "⚠️ No competitive analysis slide - investors need to see market awareness"
#         financial_entities = []
#         money_pattern = r'\$[\d,.]+(M|B|K|million|billion|thousand)?'
#         for slide in slides:
#             if slide:
#                 money_mentions = re.findall(money_pattern, slide, re.IGNORECASE)
#                 financial_entities.extend(money_mentions)
#         insights['financial_summary'] = f"Found {len(financial_entities)} financial figures across all slides"
#         locations = [e for e in all_entities if e['type'] == 'LOC']
#         if locations:
#             location_names = list(set([e['text'] for e in locations]))
#             insights['market_geography'] = f"Geographic focus: {', '.join(location_names[:3])}"
#         else: insights['market_geography'] = "No clear geographic market focus identified"
#         return insights
#     def _find_slide_by_keywords(self, slides: List[str], keywords: List[str]) -> str:
#         for slide in slides:
#             if slide and any(keyword in slide.lower() for keyword in keywords): return slide
#         return ""

# # In FoundersFuelFrontend/vc-copilot-backend/app/api/endpoints.py

# class AIAnalyzer:
#     def __init__(self):
#         self.ai_model = None
#         self.setup_ai_model()
#     def setup_ai_model(self):
#         try:
#             if os.getenv("GEMINI_API_KEY"):
#                 # Ensure you are using a model capable of following instructions well
#                 self.ai_model = genai.GenerativeModel("models/gemini-2.5-flash") 
#                 logger.info("Successfully configured AI model")
#             else: logger.error("GEMINI_API_KEY not found in environment variables")
#         except Exception as e: logger.error(f"Failed to configure AI model: {e}")
    
#     def perform_ai_analysis(self, slides: List[str], category: str, persona_desc: str, local_context: LocalAnalysisResult) -> AIAnalysisResult:
#         if not self.ai_model: raise ValueError("AI model is not available for analysis.")
#         context_json = json.dumps(asdict(local_context), cls=NumpyEncoder)
#         prompt = self._build_analysis_prompt(slides, category, persona_desc, context_json)
        
#         # Increased temperature may lead to better JSON adherence sometimes
#         response = self.ai_model.generate_content(prompt, generation_config={"temperature": 0.2})
#         return self._parse_ai_response(response.text)

#     def _build_analysis_prompt(self, slides: List[str], category: str, persona_desc: str, context_json: str) -> str:
#         # (This remains unchanged from the original provided code)
#         return f"""As a venture capitalist acting as {persona_desc}, analyze this '{category}' pitch deck. CONTEXT FROM PRE-ANALYSIS: {context_json} FULL DECK TEXT: {json.dumps(slides[:10])} Provide your analysis in a single, valid JSON object with this exact structure: {{"dimensional_scores": {{"clarity": 0, "completeness": 0, "persuasiveness": 0, "data_driven": 0}}, "vc_verdict": "Your final, conclusive statement as this VC persona", "investment_readiness_score": 0, "competitive_moat": {{"score": 0, "feedback": "Detailed assessment of competitive advantages"}}, "gtm_strategy": {{"score": 0, "feedback": "Analysis of go-to-market approach"}}, "team_market_fit": {{"score": 0, "feedback": "Evaluation of team capabilities vs market needs"}}, "slide_by_slide_feedback": [{{"slide_number": 1, "clarity_feedback": "...", "tone_feedback": "...", "intent_fulfillment_feedback": "..."}}], "key_strengths": ["strength1", "strength2"], "key_improvement_suggestions": ["improvement1", "improvement2"]}} CRITICAL REQUIREMENTS: - The "vc_verdict" MUST be a single, conversational paragraph of approximately 80-85 words. - The "key_strengths" and "key_improvement_suggestions" MUST be arrays of simple strings. - DO NOT include any Markdown (like ** **, #,  ' ' , ` ` or *) inside any string values in the JSON. - All scores MUST be integers from 0-10. - Respond ONLY with the valid JSON object and nothing else."""
    
#     # --- FIX APPLIED HERE ---
#     def _parse_ai_response(self, response_text: str) -> AIAnalysisResult:
#         try:
#             json_str = response_text.strip()
            
#             # 1. Try to find JSON block enclosed in triple backticks (```json or ```)
#             # This is a more general regex that captures any content between the first { and last } inside a code block
#             json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', json_str, re.DOTALL)
            
#             if json_match:
#                 json_str = json_match.group(1).strip()
#             else:
#                 # 2. Fallback: Find the index of the first '{' and the last '}' 
#                 # This is still dangerous but necessary if the model returns raw JSON without markdown
#                 start_index = json_str.find('{')
#                 end_index = json_str.rfind('}')
                
#                 if start_index != -1 and end_index != -1 and end_index > start_index:
#                     json_str = json_str[start_index:end_index + 1].strip()
#                 else:
#                     # If we can't isolate the JSON, the full text is passed, which will fail the parser
#                     logger.warning("Could not isolate JSON block from response text.")
            
#             parsed = json.loads(json_str)
            
#             # Optional: Add extra validation here if keys are missing
            
#             return AIAnalysisResult(**parsed)
#         except (json.JSONDecodeError, TypeError) as e:
#             logger.error(f"Failed to parse AI response: {e}\nResponse text attempt (first 300 chars): {json_str[:300]}")
#             raise ValueError("Invalid JSON response from AI.")

# # --- FastAPI Setup and Global Instances ---
# api_router = APIRouter()
# task_statuses: Dict[UUID, Dict[str, Any]] = {}
# db_manager = DatabaseManager()

# try:
#     DATASET_PATH = 'cleaned_sectors_dataset.csv'
#     if not os.path.exists(DATASET_PATH):
#         raise FileNotFoundError(f"CRITICAL: Investor dataset not found at {DATASET_PATH}")
#     matching_service = MatchingService(DATASET_PATH)
#     logger.info("Successfully loaded investor matching service.")
# except Exception as e:
#     logger.error(f"Failed to initialize investor matching service: {e}")
#     matching_service = None


# # --- Section 1: Pitch Deck Analysis Endpoints (Existing) ---

# def run_real_analysis(task_id: UUID, file_content: bytes, filename: str, file_type: str, category: str, persona_name: str):
#     try:
#         db_manager.init_db()
#         clean_filename = unquote(filename)
#         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 1/4: Extracting slides..."}
#         slides = FileProcessor.extract_slides(file_content, file_type)
#         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 2/4: Performing local ML analysis..."}
#         engine = AnalysisEngine(models=ml_models)
#         local_analysis = engine.perform_local_analysis(slides)
#         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 3/4: Getting insights from AI..."}
#         persona_desc = Config.VC_PERSONAS.get(persona_name, "a standard venture capitalist")
#         ai_analyzer = AIAnalyzer()
#         ai_analysis = ai_analyzer.perform_ai_analysis(slides, category, persona_desc, local_analysis)
#         task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 4/4: Saving analysis..."}
#         full_analysis = {**asdict(ai_analysis), **asdict(local_analysis), "slide_texts": slides, "timestamp": datetime.now().isoformat(), "analyzed_as_persona": persona_name, "category": category, "filename": clean_filename}
#         analysis_id = db_manager.save_analysis(clean_filename, category, persona_name, full_analysis)
#         task_statuses[task_id] = {"status": TaskStatus.COMPLETE, "analysis_id": analysis_id}
#         logger.info(f"Analysis complete for task {task_id}. Saved as ID {analysis_id}")
#     except Exception as e:
#         logger.error(f"Analysis failed for task {task_id}: {e}", exc_info=True)
#         task_statuses[task_id] = {"status": TaskStatus.FAILED, "message": "An internal error occurred. Please check server logs."}

# @api_router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
# async def analyze_deck(background_tasks: BackgroundTasks, file: UploadFile = File(...), category: str = Form(...), persona_name: str = Form(...)):
#     task_id = uuid.uuid4()
#     task_statuses[task_id] = {"status": TaskStatus.PENDING}
#     file_content = await file.read()
#     background_tasks.add_task(run_real_analysis, task_id, file_content, file.filename, file.content_type, category, persona_name)
#     return AnalyzeResponse(task_id=task_id)

# @api_router.get("/status/{task_id}", response_model=StatusResponse)
# async def get_analysis_status(task_id: UUID):
#     status_info = task_statuses.get(task_id)
#     if not status_info: raise HTTPException(status_code=404, detail="Task not found")
#     return StatusResponse(task_id=task_id, **status_info)

# @api_router.get("/results/{analysis_id}")
# async def get_analysis_results(analysis_id: int):
#     results = db_manager.load_analysis(analysis_id)
#     if not results: raise HTTPException(status_code=404, detail="Analysis results not found")
#     return results

# @api_router.get("/history/{filename}")
# async def get_analysis_history(filename: str):
#     logger.info(f"Fetching history for decoded filename: {filename}")
#     history = db_manager.get_history(filename)
#     if not history:
#         logger.warning(f"No history found in DB for filename: {filename}")
#         raise HTTPException(status_code=404, detail="No history found for this file")
#     processed_history = []
#     for record_row in history:
#         try:
#             record = dict(record_row)
#             analysis = json.loads(record['full_analysis'])
#             processed_history.append({"id": record['id'], "timestamp": record['timestamp'], "persona": record.get('persona', 'Unknown'), "investment_readiness_score": analysis.get('investment_readiness_score', 0), "competitive_moat_score": analysis.get('competitive_moat', {}).get('score', 0), "gtm_strategy_score": analysis.get('gtm_strategy', {}).get('score', 0)})
#         except (json.JSONDecodeError, KeyError) as e:
#             logger.error(f"Could not process history record {record.get('id', 'N/A')}: {e}")
#             continue
#     logger.info(f"Found {len(processed_history)} history records for {filename}")
#     return processed_history


# # --- Section 2: Real-time Investor Matching Endpoint (Existing) ---

# @api_router.post("/find-matches", response_model=MatchResponse)
# async def find_matches(request: MatchRequest):
#     if not matching_service:
#         raise HTTPException(status_code=503, detail="Investor Matching Service is currently unavailable.")
#     if not request.description or len(request.description.strip()) < 10:
#         raise HTTPException(status_code=400, detail="Please provide a more detailed startup description.")
#     try:
#         keywords = matching_service.extract_keywords(request.description)
#         industries = matching_service.find_similar_industries(keywords)
#         top_firms = matching_service.find_top_investors(request.stage, keywords, request.investor_type)
#         insights = matching_service.generate_insights(top_firms)
#         investor_list = [
#             Investor(
#                 investor_name=row['Investor Name'],
#                 website=row.get('Website', 'N/A'),
#                 fund_type=row['Fund Type'],
#                 fund_stage=row['Fund Stage'],
#                 focus_areas=[f.strip() for f in str(row['Fund Focus (Sectors)']).split(',') if f.strip()],
#                 match_score=int(row['score'])
#             ) for _, row in top_firms.iterrows()
#         ]
#         return MatchResponse(
#             extracted_keywords=keywords,
#             matching_industries=industries,
#             insights=insights,
#             investors=investor_list
#         )
#     except Exception as e:
#         logger.error(f"Error during investor matching: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An internal error occurred while finding matches.")


# # --- Section 3: 7-Domain Startup Evaluation (Existing) ---

# @api_router.post("/evaluate", response_model=EvaluationResponse)
# async def evaluate_startup(request: EvaluationRequest):
#     """
#     Performs a comprehensive 7-domain evaluation of a startup idea using
#     RAG and live web search.
#     """
#     if not request.startup_idea or len(request.startup_idea.strip()) < 20:
#         raise HTTPException(status_code=400, detail="Please provide a detailed startup idea for a meaningful evaluation.")

#     try:
#         logger.info(f"Received evaluation request for startup: '{request.startup_name or 'Unnamed'}'")
#         # Call the evaluation service with the data from the request body
#         evaluation_result = evaluation_service.comprehensive_startup_evaluation(
#             startup_idea=request.startup_idea,
#             startup_name=request.startup_name,
#             uploaded_content=request.uploaded_content
#         )
        
#         # The service might return an error dictionary
#         if "error" in evaluation_result:
#             logger.error(f"Evaluation service returned an error: {evaluation_result['error']}")
#             raise HTTPException(status_code=500, detail=evaluation_result["error"])
        
#         return EvaluationResponse(evaluation=evaluation_result)

#     except Exception as e:
#         logger.error(f"An unexpected error occurred in the /evaluate endpoint: {e}", exc_info=True)
#         # Re-raise as HTTPException to be handled by FastAPI
#         if not isinstance(e, HTTPException):
#             raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
#         raise e
        
# # --- Section 4: Success Predictor Endpoints (NEW) ---

# @api_router.post("/predict")
# def predict(data: SuccessInput):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Prediction model is not loaded. Check server logs.")
        
#     user_input = data.dict()
#     # Clean text using the utility function defined above
#     user_input["text"] = clean_text_basic(user_input["text"])

#     # Ensure all required numeric features are present for the model
#     for col in numeric_features:
#         if col not in user_input:
#             user_input[col] = 0

#     input_df = pd.DataFrame([user_input])

#     try:
#         # Prediction using the globally loaded model
#         prob = model.predict_proba(input_df)[0, 1] * 100
#         return {"success_probability": round(prob, 2)}
#     except Exception as e:
#         logger.error(f"Prediction error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Error during prediction.")


# @api_router.post("/explain")
# def explain(data: SuccessInput):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Prediction model is not loaded. Check server logs.")
        
#     user_input = data.dict()
#     user_input["text"] = clean_text_basic(user_input["text"])
#     for col in numeric_features:
#         if col not in user_input:
#             user_input[col] = 0

#     try:
#         # Generate SHAP explanation and narrative summary
#         shap_values, feature_names, probability = get_shap_explanation(user_input)
#         summary = generate_narrative_summary(user_input, shap_values, feature_names, probability)
#         return {"narrative": summary}
#     except Exception as e:
#         logger.error(f"Explanation error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Error during explanation.")
import uuid
import json
import re
import os
import logging
from datetime import datetime
from uuid import UUID
from typing import Dict, List, Any
from urllib.parse import unquote

# Third-party imports
import textstat
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
import joblib
import pandas as pd
import string

# FastAPI imports
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Local app imports
from app.core.ml import ml_models
from app.db.database import DatabaseManager, NumpyEncoder
from app.services.file_processor import FileProcessor
from app.services.matching_service import MatchingService
from app.services.evaluation_service import evaluation_service

# Import schemas
from app.models.schemas import (
    AnalyzeResponse, 
    StatusResponse, 
    TaskStatus,
    MatchRequest,
    MatchResponse,
    Investor,
    EvaluationRequest,
    EvaluationResponse
)

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    logger.error(f"AI model configuration failed: {e}")

# --- SUCCESS PREDICTOR SETUP ---
class SuccessInput(BaseModel): 
    text: str
    goal: float = 0
    usd_pledged: float = 0
    usd_pledged_real: float = 0
    usd_goal_real: float = 0
    pledged: float = 0
    backers: int = 0

def clean_text_basic(text: str) -> str: 
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

numeric_features = ["goal", "usd_pledged", "usd_pledged_real", "usd_goal_real", "pledged", "backers"] 

# Try loading the prediction model
try: 
    model = joblib.load("pitchmodel.pkl") 
    # Try importing util functions if they exist in trainmodel.py
    from trainmodel import get_shap_explanation, generate_narrative_summary
    logger.info("Successfully loaded pitchmodel.pkl and trainmodel utilities.")
except Exception as e: 
    model = None
    logger.warning(f"Prediction features disabled. Reason: {e}")
    def get_shap_explanation(*args, **kwargs): return None, [], 0 
    def generate_narrative_summary(*args, **kwargs): return "Prediction model not available."


# --- DATA CLASSES AND CONFIG ---
@dataclass
class SlideAnalysis:
    readability_interpretation: str; quantitative_count: int; sentiment: Dict[str, Any]; named_entities: List[Dict[str, str]]

@dataclass
class LocalAnalysisResult:
    redundant_slides: List[Dict[str, Any]]; local_metrics: List[Any]; prob_sol_alignment: float; checklist: Dict[str, bool]; buzzword_analysis: Dict[str, Any]; ner_insights: Dict[str, Any]

@dataclass
class AIAnalysisResult:
    dimensional_scores: Dict[str, float]; vc_verdict: str; investment_readiness_score: float; competitive_moat: Dict[str, Any]; gtm_strategy: Dict[str, Any]; team_market_fit: Dict[str, Any]; slide_by_slide_feedback: List[Dict[str, str]]; key_strengths: List[str]; key_improvement_suggestions: List[str]

class Config:
    VC_PERSONAS = {
        "The Friendly Mentor": "a friendly, encouraging mentor who focuses on nurturing potential",
        "The Data-Driven Analyst": "a sharp, data-driven analyst who demands concrete metrics and evidence",
        "The 'Big Picture' Strategist": "a 'big picture' strategist who evaluates market timing and scalability",
        "The Devil's Advocate": "a devil's advocate who challenges assumptions and identifies risks"
    }
    BUZZWORDS = ['synergy', 'disrupt', 'paradigm shift', 'game-changer', 'revolutionize', 'innovative', 'cutting-edge', 'breakthrough', 'revolutionary']
    CHECKLIST_PATTERNS = {
        "TAM/SAM/SOM": r'tam|sam|som|total addressable market|market size',
        "Go-to-Market Strategy": r'go-to-market|gtm|target audience|distribution channels|marketing plan|customer acquisition',
        "Funding Ask": r'(\$[0-9,.]+m? seed|\$[0-9,.]+m? series [a-z]|raising|seeking|funding)',
        "Traction Metrics (KPIs)": r'mrr|arr|cac|ltv|mau|daily active users|kpi|revenue|growth|metrics',
        "Milestones / Timeline": r'q[1-4] 202[4-9]|roadmap|timeline|launch|scale|milestones',
        "Team Credentials": r'ex-google|ex-meta|phd|serial entrepreneur|[0-9]+\+ years of experience|founder|cto|ceo',
        "Tech Stack / IP": r'ai|blockchain|patent|proprietary algorithm|ml model|technology|infrastructure',
        "Regulatory / Compliance": r'hipaa|gdpr|fda approval|iso certified|compliance|regulation',
        "Risk Disclosure": r'risk|challenge|headwind|burn rate|scalability issues|threats',
        "Financial Projections": r'revenue projection|financial forecast|p&l|unit economics|burn rate'
    }

# --- ANALYSIS ENGINE ---
class AnalysisEngine:
    def __init__(self, models: Dict[str, Any]):
        self.models = models
    def perform_local_analysis(self, slides: List[str]) -> LocalAnalysisResult:
        if not slides: raise ValueError("No slides available for analysis.")
        embeddings = self.models['bert'].encode(slides)
        sim_matrix = cosine_similarity(embeddings)
        redundant_pairs = []
        for i in range(len(slides)):
            for j in range(i + 1, len(slides)):
                if sim_matrix[i][j] > 0.85: redundant_pairs.append({'slide1': i + 1, 'slide2': j + 1, 'similarity': float(sim_matrix[i][j])})
        prob_sol_alignment = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]) if len(slides) > 1 else 0.0
        local_metrics = [self._analyze_single_slide(text) for text in slides if text.strip()]
        combined_text = " ".join(slides).lower()
        checklist = self._perform_checklist_analysis(combined_text)
        buzzword_count = self._count_buzzwords(combined_text)
        buzzword_analysis = {"count": buzzword_count}
        ner_insights = self._perform_ner_analysis(slides)
        return LocalAnalysisResult(redundant_slides=redundant_pairs, local_metrics=[asdict(m) for m in local_metrics], prob_sol_alignment=prob_sol_alignment, checklist=checklist, buzzword_analysis=buzzword_analysis, ner_insights=ner_insights)
    def _analyze_single_slide(self, slide_text: str) -> SlideAnalysis:
        flesch_score = textstat.flesch_reading_ease(slide_text)
        if flesch_score > 80: readability = "Very Easy"
        elif flesch_score > 60: readability = "Easy"
        elif flesch_score > 40: readability = "Standard"
        else: readability = "Difficult"
        quant_patterns = [r'\$\d[\d,.]*[kmb]?', r'\d[\d,.]*%', r'\d[\d,.]+\s?(?:million|billion|m|b)', r'\d+x', r'[0-9]+\+\s*(?:users|customers|clients)']
        quantitative_count = sum(len(re.findall(p, slide_text, re.I)) for p in quant_patterns)
        try:
            sentiment_scores = self.models['sentiment'](slide_text[:512])[0]
            sentiment = max(sentiment_scores, key=lambda x: x['score'])
            sentiment['label'] = sentiment['label'].replace('LABEL_', '').upper()
        except Exception:
            sentiment = {'label': 'NEUTRAL', 'score': 0.0}
        try:
            entities = self.models['ner'](slide_text[:512])
            named_entities = [{'entity': e['word'], 'type': e.get('entity_group', 'UNKNOWN')} for e in entities if e.get('score', 0) > 0.8]
        except Exception: named_entities = []
        return SlideAnalysis(readability, quantitative_count, sentiment, named_entities)
    def _perform_checklist_analysis(self, text: str) -> Dict[str, bool]: return {item: bool(re.search(pattern, text, re.I)) for item, pattern in Config.CHECKLIST_PATTERNS.items()}
    def _count_buzzwords(self, text: str) -> int: return sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text, re.I)) for word in Config.BUZZWORDS)
    def _perform_ner_analysis(self, slides: List[str]) -> Dict[str, str]:
        insights = {}
        all_entities = []
        for slide in slides:
            if slide:
                try:
                    entities = self.models['ner'](slide[:512])
                    for entity in entities:
                        if entity.get('score', 0) > 0.8: all_entities.append({'text': entity['word'].replace('##', ''), 'type': entity.get('entity_group', 'UNKNOWN'), 'confidence': entity.get('score', 0)})
                except Exception: continue
        # Simple heuristics for slide identification
        team_slide = next((s for s in slides if s and any(k in s.lower() for k in ['team', 'founder', 'ceo'])), "")
        comp_slide = next((s for s in slides if s and any(k in s.lower() for k in ['competitor', 'market landscape'])), "")
        
        insights['team_summary'] = "Team slide found" if team_slide else "No team slide detected"
        insights['competition_summary'] = "Competition slide found" if comp_slide else "No competition slide detected"
        
        financial_entities = []
        money_pattern = r'\$[\d,.]+(M|B|K|million|billion|thousand)?'
        for slide in slides:
            if slide:
                money_mentions = re.findall(money_pattern, slide, re.IGNORECASE)
                financial_entities.extend(money_mentions)
        insights['financial_summary'] = f"Found {len(financial_entities)} financial figures"
        
        return insights

class AIAnalyzer:
    def __init__(self):
        self.ai_model = None
        self.setup_ai_model()
    
    def setup_ai_model(self):
        try:
            if os.getenv("GEMINI_API_KEY"):
                self.ai_model = genai.GenerativeModel("models/gemini-2.5-flash")
                logger.info("Successfully configured AI model")
            else: 
                logger.error("GEMINI_API_KEY not found")
        except Exception as e:
            logger.error(f"Failed to configure AI model: {e}")
    
    def perform_ai_analysis(self, slides: List[str], category: str, persona_desc: str, local_context: LocalAnalysisResult) -> AIAnalysisResult:
        if not self.ai_model: raise ValueError("AI model is not available.")
        context_json = json.dumps(asdict(local_context), cls=NumpyEncoder)
        prompt = self._build_analysis_prompt(slides, category, persona_desc, context_json)
        response = self.ai_model.generate_content(prompt, generation_config={"temperature": 0.2})
        return self._parse_ai_response(response.text)

    def _build_analysis_prompt(self, slides: List[str], category: str, persona_desc: str, context_json: str) -> str:
        return f"""As a venture capitalist acting as {persona_desc}, analyze this '{category}' pitch deck. CONTEXT FROM PRE-ANALYSIS: {context_json} FULL DECK TEXT: {json.dumps(slides[:10])} Provide your analysis in a single, valid JSON object with this exact structure: {{"dimensional_scores": {{"clarity": 0, "completeness": 0, "persuasiveness": 0, "data_driven": 0}}, "vc_verdict": "Your final, conclusive statement as this VC persona", "investment_readiness_score": 0, "competitive_moat": {{"score": 0, "feedback": "Detailed assessment of competitive advantages"}}, "gtm_strategy": {{"score": 0, "feedback": "Analysis of go-to-market approach"}}, "team_market_fit": {{"score": 0, "feedback": "Evaluation of team capabilities vs market needs"}}, "slide_by_slide_feedback": [{{"slide_number": 1, "clarity_feedback": "...", "tone_feedback": "...", "intent_fulfillment_feedback": "..."}}], "key_strengths": ["strength1", "strength2"], "key_improvement_suggestions": ["improvement1", "improvement2"]}} CRITICAL REQUIREMENTS: - The "vc_verdict" MUST be a single, conversational paragraph of approximately 80-85 words. - The "key_strengths" and "key_improvement_suggestions" MUST be arrays of simple strings. - DO NOT include any Markdown (like ** **, #,  ' ' , ` ` or *) inside any string values in the JSON. - All scores MUST be integers from 0-10. - Respond ONLY with the valid JSON object and nothing else."""
    
    def _parse_ai_response(self, response_text: str) -> AIAnalysisResult:
        try:
            json_str = response_text.strip()
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                start_index = json_str.find('{')
                end_index = json_str.rfind('}')
                if start_index != -1 and end_index != -1:
                    json_str = json_str[start_index:end_index + 1].strip()
            parsed = json.loads(json_str)
            return AIAnalysisResult(**parsed)
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            raise ValueError("Invalid JSON response from AI.")

# --- ROUTER SETUP ---
api_router = APIRouter()
task_statuses: Dict[UUID, Dict[str, Any]] = {}
db_manager = DatabaseManager()

# Init Matching Service
try:
    DATASET_PATH = 'cleaned_sectors_dataset.csv'
    if os.path.exists(DATASET_PATH):
        matching_service = MatchingService(DATASET_PATH)
    else:
        matching_service = None
        logger.warning(f"Dataset {DATASET_PATH} not found. Matching service disabled.")
except Exception as e:
    matching_service = None
    logger.error(f"Matching service init failed: {e}")

# --- ENDPOINTS ---

# 1. ANALYSIS
def run_real_analysis(task_id: UUID, file_content: bytes, filename: str, file_type: str, category: str, persona_name: str):
    try:
        db_manager.init_db()
        clean_filename = unquote(filename)
        task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 1/4: Extracting slides..."}
        slides = FileProcessor.extract_slides(file_content, file_type)
        
        task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 2/4: Local ML analysis..."}
        engine = AnalysisEngine(models=ml_models)
        local_analysis = engine.perform_local_analysis(slides)
        
        task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Step 3/4: AI Analysis..."}
        persona_desc = Config.VC_PERSONAS.get(persona_name, "a standard venture capitalist")
        ai_analyzer = AIAnalyzer()
        ai_analysis = ai_analyzer.perform_ai_analysis(slides, category, persona_desc, local_analysis)
        
        task_statuses[task_id] = {"status": TaskStatus.PROCESSING, "message": "Saving..."}
        full_analysis = {**asdict(ai_analysis), **asdict(local_analysis), "slide_texts": slides, "timestamp": datetime.now().isoformat(), "analyzed_as_persona": persona_name, "category": category, "filename": clean_filename}
        analysis_id = db_manager.save_analysis(clean_filename, category, persona_name, full_analysis)
        
        task_statuses[task_id] = {"status": TaskStatus.COMPLETE, "analysis_id": analysis_id}
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        task_statuses[task_id] = {"status": TaskStatus.FAILED, "message": "Internal error."}

@api_router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
async def analyze_deck(background_tasks: BackgroundTasks, file: UploadFile = File(...), category: str = Form(...), persona_name: str = Form(...)):
    task_id = uuid.uuid4()
    task_statuses[task_id] = {"status": TaskStatus.PENDING}
    file_content = await file.read()
    background_tasks.add_task(run_real_analysis, task_id, file_content, file.filename, file.content_type, category, persona_name)
    return AnalyzeResponse(task_id=task_id)

@api_router.get("/status/{task_id}", response_model=StatusResponse)
async def get_analysis_status(task_id: UUID):
    status_info = task_statuses.get(task_id)
    if not status_info: raise HTTPException(status_code=404, detail="Task not found")
    return StatusResponse(task_id=task_id, **status_info)

@api_router.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: int):
    results = db_manager.load_analysis(analysis_id)
    if not results: raise HTTPException(status_code=404, detail="Analysis results not found")
    return results

@api_router.get("/history/{filename}")
async def get_analysis_history(filename: str):
    history = db_manager.get_history(filename)
    if not history: raise HTTPException(status_code=404, detail="No history found")
    processed_history = []
    for record in history:
        try:
            r_dict = dict(record)
            analysis = json.loads(r_dict['full_analysis'])
            processed_history.append({
                "id": r_dict['id'],
                "timestamp": r_dict['timestamp'],
                "persona": r_dict.get('persona', 'Unknown'),
                "investment_readiness_score": analysis.get('investment_readiness_score', 0),
                "competitive_moat_score": analysis.get('competitive_moat', {}).get('score', 0),
                "gtm_strategy_score": analysis.get('gtm_strategy', {}).get('score', 0)
            })
        except Exception: continue
    return processed_history

# 2. MATCHING
@api_router.post("/find-matches", response_model=MatchResponse)
async def find_matches(request: MatchRequest):
    if not matching_service: raise HTTPException(status_code=503, detail="Matching Service unavailable.")
    try:
        keywords = matching_service.extract_keywords(request.description)
        industries = matching_service.find_similar_industries(keywords)
        top_firms = matching_service.find_top_investors(request.stage, keywords, request.investor_type)
        insights = matching_service.generate_insights(top_firms)
        investor_list = [
            Investor(
                investor_name=row['Investor Name'],
                website=row.get('Website', 'N/A'),
                fund_type=row['Fund Type'],
                fund_stage=row['Fund Stage'],
                focus_areas=[f.strip() for f in str(row['Fund Focus (Sectors)']).split(',') if f.strip()],
                match_score=int(row['score'])
            ) for _, row in top_firms.iterrows()
        ]
        return MatchResponse(extracted_keywords=keywords, matching_industries=industries, insights=insights, investors=investor_list)
    except Exception as e:
        logger.error(f"Matching error: {e}")
        raise HTTPException(status_code=500, detail="Error during matching.")

# 3. EVALUATION
@api_router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_startup(request: EvaluationRequest):
    if not request.startup_idea: raise HTTPException(status_code=400, detail="Idea required.")
    try:
        result = evaluation_service.comprehensive_startup_evaluation(request.startup_idea, request.startup_name, request.uploaded_content)
        if "error" in result: raise HTTPException(status_code=500, detail=result["error"])
        return EvaluationResponse(evaluation=result)
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail="Server error.")

# 4. PREDICTION
@api_router.post("/predict")
def predict(data: SuccessInput):
    if model is None: raise HTTPException(status_code=500, detail="Prediction model not loaded.")
    user_input = data.dict()
    user_input["text"] = clean_text_basic(user_input["text"])
    for col in numeric_features:
        if col not in user_input: user_input[col] = 0
    input_df = pd.DataFrame([user_input])
    try:
        prob = model.predict_proba(input_df)[0, 1] * 100
        return {"success_probability": round(prob, 2)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction.")

@api_router.post("/explain")
def explain(data: SuccessInput):
    if model is None: raise HTTPException(status_code=500, detail="Prediction model not loaded.")
    user_input = data.dict()
    user_input["text"] = clean_text_basic(user_input["text"])
    for col in numeric_features:
        if col not in user_input: user_input[col] = 0
    try:
        shap_values, feature_names, probability = get_shap_explanation(user_input)
        summary = generate_narrative_summary(user_input, shap_values, feature_names, probability)
        return {"narrative": summary}
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail="Error during explanation.")