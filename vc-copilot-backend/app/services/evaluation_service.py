import os
import re
import requests
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer
import pinecone
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StartupEvaluator:
    """
    A service class to handle the comprehensive 7-domain evaluation of a startup idea.
    It integrates RAG from a Pinecone index and live web search for market intelligence.
    """
    def __init__(self):
        """
        Initializes the models and API clients required for the evaluation service.
        """
        try:
            # Initialize Sentence Transformer for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Configure Google Gemini API
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

            # Initialize Pinecone
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables.")
            pc = pinecone.Pinecone(api_key=pinecone_api_key)
            self.pinecone_index = pc.Index("startwise-rag-knowledge") # Make sure this index name is correct

            # Check SerpApi key for web search functionality
            if not os.getenv("SERPAPI_API_KEY"):
                logger.warning("SERPAPI_API_KEY not found. Web search capabilities will be limited.")

        except Exception as e:
            logger.error(f"Failed to initialize StartupEvaluator service: {e}")
            raise

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieves relevant context from the Pinecone knowledge base.
        """
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            logger.info(f"Querying Pinecone with: '{query}'")
            
            results = self.pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            all_matches = results.get('matches', [])
            logger.info(f"Pinecone returned {len(all_matches)} total matches")
            
            if all_matches:
                logger.info(f"All similarity scores: {[match['score'] for match in all_matches]}")
            
            # Lower threshold to 0.15 since general knowledge base has lower similarity scores
            contexts = [
                {
                    'text': match['metadata']['text'],
                    'source': match['metadata'].get('source_file', 'Unknown'),
                    'domain': match['metadata'].get('domain', 'General'),
                    'similarity': match['score']
                }
                for match in all_matches if match['score'] > 0.15
            ]
            
            if len(all_matches) > 0 and len(contexts) == 0:
                logger.warning(f"All {len(all_matches)} matches were below 0.15 similarity threshold")
            
            return contexts
        except Exception as e:
            logger.error(f"Context retrieval from Pinecone failed: {e}")
            return []

    def scrape_url_content(self, url: str) -> Optional[str]:
        """
        Scrapes the main textual content from a given URL.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml')
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            main_content = soup.find('main') or soup.find('article') or soup
            text = re.sub(r'\s+', ' ', main_content.get_text(separator=' ', strip=True))
            return text[:2500] # Limit content size
        except Exception as e:
            logger.warning(f"URL scraping failed for {url}: {e}")
            return None

    def web_search_with_scraping(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Performs a web search using SerpApi and scrapes content from trusted sources.
        """
        serpapi_key = os.getenv('SERPAPI_API_KEY')
        if not serpapi_key:
            return []

        try:
            search = GoogleSearch({
                "q": query,
                "api_key": serpapi_key,
                "engine": "google",
                "num": max_results
            })
            results = search.get_dict().get("organic_results", [])
            
            enhanced_results = []
            for result in results:
                link = result.get('link')
                enhanced_result = {
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    'link': link
                }
                # Scrape content from the URL
                if link:
                    enhanced_result['full_content'] = self.scrape_url_content(link)
                enhanced_results.append(enhanced_result)

            return enhanced_results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def parse_evaluation_response(self, response_text: str) -> Dict[str, str]:
        """
        Parses the raw text response from Gemini into a structured dictionary.
        """
        sections = {
            "idea_evaluation": "", "uniqueness": "", "tech_stack": "",
            "similar_startups": "", "pitch": "", "improvements": "",
            "success_metrics": "", "additional": ""
        }
        
        # Use regex to split the text by the numbered domain headers
        pattern = r"##\s*\d+\.\s*(STARTUP IDEA EVALUATION|UNIQUENESS CHECK|TECH STACK RECOMMENDATION|SIMILAR STARTUPS|PITCH GENERATION|IMPROVEMENT SUGGESTIONS|SUCCESS PROBABILITY & INVESTMENT METRICS|ADDITIONAL ANALYSIS)[\s\S]*?(?=(?:##\s*\d+\.|$))"
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        
        key_mapping = {
            "startup idea evaluation": "idea_evaluation",
            "uniqueness check": "uniqueness",
            "tech stack recommendation": "tech_stack",
            "similar startups": "similar_startups",
            "pitch generation": "pitch",
            "improvement suggestions": "improvements",
            "success probability & investment metrics": "success_metrics",
            "additional analysis": "additional"
        }

        for match in matches:
            lines = match.strip().split('\n')
            header = lines[0]
            content = "\n".join(lines[1:]).strip()
            
            for title, key in key_mapping.items():
                if title in header.lower():
                    sections[key] = content
                    break
        
        if not any(sections.values()):
            sections["additional"] = response_text

        return sections

    def comprehensive_startup_evaluation(self, startup_idea: str, startup_name: str = "", uploaded_content: str = "") -> Dict[str, str]:
        """
        Orchestrates the entire evaluation process by gathering context, building a prompt,
        and calling the Gemini API for a single, comprehensive analysis.
        """
        logger.info(f"Starting evaluation for startup: '{startup_name or 'Unnamed'}'")
        
        rag_contexts = self.retrieve_context(startup_idea, top_k=8)
        logger.info(f"Retrieved {len(rag_contexts)} contexts from Pinecone knowledge base")
        if rag_contexts:
            logger.info(f"Top context similarity scores: {[ctx['similarity'] for ctx in rag_contexts[:3]]}")

        web_searches = [
            f"{startup_idea} market size and trends 2025",
            f"{startup_idea} top competitors",
            f"'{startup_idea}' business model analysis",
        ]
        all_web_contexts = []
        for search_query in web_searches:
            all_web_contexts.extend(self.web_search_with_scraping(search_query, max_results=2))
        
        logger.info(f"Retrieved {len(all_web_contexts)} web search results")

        context_text = ""
        if uploaded_content:
            context_text += f"\n--- UPLOADED STARTUP DOCUMENTS ---\n{uploaded_content[:2000]}\n"
            logger.info(f"Added {len(uploaded_content)} characters from uploaded content")
        if rag_contexts:
            context_text += "\n--- EXPERT KNOWLEDGE BASE ---\n"
            context_text += "\n".join([f"Source: {ctx['source']} (Relevance: {ctx['similarity']:.2f})\nContent: {ctx['text']}" for ctx in rag_contexts]) + "\n"
            logger.info(f"Added {len(rag_contexts)} contexts from knowledge base to prompt")
        if all_web_contexts:
            context_text += "\n--- CURRENT MARKET INTELLIGENCE ---\n"
            # Fixed: Handle None values properly by using 'or' operator
            context_text += "\n".join([
                f"Title: {ctx.get('title') or 'N/A'}\n"
                f"Snippet: {ctx.get('snippet') or 'N/A'}\n"
                f"Full Content: {(ctx.get('full_content') or 'N/A')[:500]}..." 
                for ctx in all_web_contexts if ctx
            ]) + "\n"
            logger.info(f"Added {len(all_web_contexts)} web search results to prompt")

        comprehensive_prompt = f"""
        You are an expert VC analyst. Provide a deep, structured analysis of the following startup.

        STARTUP DETAILS:
        - Name: {startup_name or 'Not specified'}
        - Idea: {startup_idea}

        CONTEXTUAL DATA:
        {context_text}

        ---

        TASK:
        Provide a comprehensive evaluation covering these EXACT 7 domains. Be critical, insightful, and provide actionable advice. Use markdown for structure.

        ## 1. STARTUP IDEA EVALUATION
        - Analyze problem-solution fit, target market, business model viability, value proposition, and market timing.
        - Provide a Core Concept Validation Score (1-10) with detailed reasoning.

        ## 2. UNIQUENESS CHECK
        - Assess market differentiation, identify competitors (direct/indirect), and analyze competitive advantages.
        - Provide a Uniqueness Score (1-10) with reasoning.

        ## 3. TECH STACK RECOMMENDATION
        - Suggest a full tech stack (frontend, backend, database), architecture patterns, key integrations, and assess complexity and risks.

        ## 4. SIMILAR STARTUPS
        - List 5-8 relevant competitors. Detail their successes, failures, and market position to draw insights.

        ## 5. PITCH GENERATION
        - Write a compelling 60-second elevator pitch.
        - Outline a detailed investment pitch covering the 10 key investor points (Problem, Solution, Market, etc.).

        ## 6. IMPROVEMENT SUGGESTIONS
        - Provide specific, actionable recommendations for the product, market strategy, and business model.

        ## 7. SUCCESS PROBABILITY & INVESTMENT METRICS
        - Estimate the overall success probability (%).
        - Provide an Investment Attractiveness Rating (1-10).
        - Analyze scalability potential, risk-reward, and funding readiness.
        """

        try:
            logger.info("Generating comprehensive evaluation with Gemini API.")
            response = self.gemini_model.generate_content(comprehensive_prompt)
            
            parsed_sections = self.parse_evaluation_response(response.text)
            logger.info("Evaluation generated successfully.")
            return parsed_sections
            
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            return {"error": f"Failed to generate evaluation due to an API error: {e}"}

# Create a single, reusable instance of the service
evaluation_service = StartupEvaluator()