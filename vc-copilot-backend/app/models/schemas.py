# # # # vc-copilot-backend/app/models/schemas.py

# # # from pydantic import BaseModel
# # # from typing import Optional
# # # from uuid import UUID
# # # from enum import Enum

# # # class TaskStatus(str, Enum):
# # #     PENDING = "pending"
# # #     PROCESSING = "processing"
# # #     COMPLETE = "complete"
# # #     FAILED = "failed"

# # # class AnalyzeResponse(BaseModel):
# # #     task_id: UUID
# # #     status: TaskStatus = TaskStatus.PENDING

# # # class StatusResponse(BaseModel):
# # #     task_id: UUID
# # #     status: TaskStatus
# # #     message: Optional[str] = None
# # #     analysis_id: Optional[int] = None # Will be set when status is 'complete'


    
# # from pydantic import BaseModel
# # from typing import List, Optional, Dict, Any
# # from uuid import UUID
# # from enum import Enum

# # # --- Schemas for Asynchronous Task Processing ---
# # # These models are used for endpoints that might run long processes,
# # # like analyzing a pitch deck file.

# # class TaskStatus(str, Enum):
# #     """Enumeration for the status of a background task."""
# #     PENDING = "pending"
# #     PROCESSING = "processing"
# #     COMPLETE = "complete"
# #     FAILED = "failed"

# # class AnalyzeResponse(BaseModel):
# #     """Response model when a new analysis task is created."""
# #     task_id: UUID
# #     status: TaskStatus = TaskStatus.PENDING

# # class StatusResponse(BaseModel):
# #     """Response model for checking the status of a task."""
# #     task_id: UUID
# #     status: TaskStatus
# #     message: Optional[str] = None
# #     analysis_id: Optional[int] = None  # Will be set when status is 'complete'


# # # --- Schemas for Direct Investor Matching ---
# # # These models are used for the real-time investor matching endpoint.

# # class MatchRequest(BaseModel):
# #     """Request model for finding investor matches."""
# #     description: str
# #     stage: str
# #     investor_type: str

# # class Investor(BaseModel):
# #     """Data model for a single investor."""
# #     investor_name: str
# #     website: str
# #     fund_type: str
# #     fund_stage: str
# #     focus_areas: List[str]
# #     match_score: int

# # class MatchResponse(BaseModel):
# #     """Response model containing the results of an investor match search."""
# #     extracted_keywords: List[str]
# #     matching_industries: List[str]
# #     insights: List[str]
# #     investors: List[Investor]



# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict, Any
# from uuid import UUID
# from enum import Enum

# # --- Schemas for Asynchronous Task Processing (EXISTING) ---
# class TaskStatus(str, Enum):
#     """Enumeration for the status of a background task."""
#     PENDING = "pending"
#     PROCESSING = "processing"
#     COMPLETE = "complete"
#     FAILED = "failed"

# class AnalyzeResponse(BaseModel):
#     """Response model when a new analysis task is created."""
#     task_id: UUID
#     status: TaskStatus = TaskStatus.PENDING

# class StatusResponse(BaseModel):
#     """Response model for checking the status of a task."""
#     task_id: UUID
#     status: TaskStatus
#     message: Optional[str] = None
#     analysis_id: Optional[int] = None  # Will be set when status is 'complete'


# # --- Schemas for Direct Investor Matching (EXISTING) ---
# class MatchRequest(BaseModel):
#     """Request model for finding investor matches."""
#     description: str
#     stage: str
#     investor_type: str

# class Investor(BaseModel):
#     """Data model for a single investor."""
#     investor_name: str
#     website: str
#     fund_type: str
#     fund_stage: str
#     focus_areas: List[str]
#     match_score: int

# class MatchResponse(BaseModel):
#     """Response model containing the results of an investor match search."""
#     extracted_keywords: List[str]
#     matching_industries: List[str]
#     insights: List[str]
#     investors: List[Investor]

# # --- ADDITIONS for 7-Domain Startup Evaluation ---

# class AnalysisSection(BaseModel):
#     """Structure for each of the seven evaluation domains/subsections."""
#     title: str
#     content: str

# # Alias mapping for consistent parsing of the LLM output in evaluation_service.py
# EVALUATION_SECTION_ALIASES = {
#     "1. STARTUP IDEA EVALUATION": "idea_evaluation",
#     "2. UNIQUENESS CHECK": "uniqueness_check",
#     "3. TECH STACK RECOMMENDATION": "tech_stack",
#     "4. SIMILAR STARTUPS": "similar_startups",
#     "5. PITCH GENERATION": "pitch_generation",
#     "6. IMPROVEMENT SUGGESTIONS": "improvement_suggestions",
#     "7. SUCCESS PROBABILITY & INVESTMENT METRICS": "success_metrics",
#     "ADDITIONAL ANALYSIS": "additional_analysis"
# }

# class ComprehensiveEvaluationResponse(BaseModel):
#     """The structured response for the full 7-domain evaluation."""
#     # These map directly to the keys used in evaluation_service.py and the LLM prompt
#     idea_evaluation: AnalysisSection
#     uniqueness_check: AnalysisSection
#     tech_stack: AnalysisSection
#     similar_startups: AnalysisSection
#     pitch_generation: AnalysisSection
#     improvement_suggestions: AnalysisSection
#     success_metrics: AnalysisSection
#     additional_analysis: Optional[AnalysisSection] = None
    
#     status: str = "success"


# vc-copilot-backend/app/models/schemas.py
# FINAL AND UNIFIED VERSION

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import UUID
from enum import Enum

# --- Schemas for Asynchronous Task Processing ---
# These models are used for endpoints that might run long processes,
# like analyzing a pitch deck file.

class TaskStatus(str, Enum):
    """Enumeration for the status of a background task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"

class AnalyzeResponse(BaseModel):
    """Response model when a new analysis task is created."""
    task_id: UUID
    status: TaskStatus = TaskStatus.PENDING

class StatusResponse(BaseModel):
    """Response model for checking the status of a task."""
    task_id: UUID
    status: TaskStatus
    message: Optional[str] = None
    analysis_id: Optional[int] = None  # Will be set when status is 'complete'


# --- Schemas for Direct Investor Matching ---
# These models are used for the real-time investor matching endpoint.

class MatchRequest(BaseModel):
    """Request model for finding investor matches."""
    description: str
    stage: str
    investor_type: str

class Investor(BaseModel):
    """Data model for a single investor."""
    investor_name: str
    website: str
    fund_type: str
    fund_stage: str
    focus_areas: List[str]
    match_score: int

class MatchResponse(BaseModel):
    """Response model containing the results of an investor match search."""
    extracted_keywords: List[str]
    matching_industries: List[str]
    insights: List[str]
    investors: List[Investor]


# --- Schemas for 7-Domain Startup Evaluation (NEW) ---
# These models are required for the new /evaluate endpoint.

class EvaluationRequest(BaseModel):
    """Request model for the 7-domain startup evaluation."""
    startup_idea: str
    startup_name: Optional[str] = None
    uploaded_content: Optional[str] = None

class EvaluationResponse(BaseModel):
    """Response model containing the completed evaluation."""
    evaluation: Dict[str, str]