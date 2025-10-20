# vc-copilot-backend/app/db/database.py
# FINAL CORRECTED VERSION

import sqlite3
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    # ... (This class is correct and remains the same)
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

DB_SCHEMA = '''
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    category TEXT NOT NULL,
    filename TEXT NOT NULL,
    full_analysis TEXT NOT NULL,
    persona TEXT,
    version INTEGER DEFAULT 1
)
'''

class DatabaseManager:
    def __init__(self, db_path: str = 'pitch_analysis.db'):
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def get_connection(self):
        # ... (This method is correct and remains the same)
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def init_db(self):
        # ... (This method is correct and remains the same)
        with self.get_connection() as conn:
            conn.execute(DB_SCHEMA)
            # Add migration logic for older databases if needed
            cursor = conn.execute("PRAGMA table_info(analyses)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'persona' not in columns:
                conn.execute("ALTER TABLE analyses ADD COLUMN persona TEXT")
            if 'version' not in columns:
                conn.execute("ALTER TABLE analyses ADD COLUMN version INTEGER DEFAULT 1")
            conn.commit()

    def save_analysis(self, filename: str, category: str, persona: str, analysis: Dict[str, Any]) -> int:
        # ... (This method is correct and remains the same)
        with self.get_connection() as conn:
            analysis_json = json.dumps(analysis, cls=NumpyEncoder)
            cursor = conn.execute(
                'INSERT INTO analyses (timestamp, category, filename, persona, full_analysis) VALUES (?, ?, ?, ?, ?)',
                (
                    analysis.get('timestamp', datetime.now().isoformat()),
                    category,
                    filename,
                    persona,
                    analysis_json
                )
            )
            conn.commit()
            return cursor.lastrowid

    def load_analysis(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        # ... (This method is correct and remains the same)
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT full_analysis FROM analyses WHERE id = ?', (analysis_id,))
            result = cursor.fetchone()
            if not result: return None
            return json.loads(result['full_analysis'])

    # --- THE CRUCIAL FIX IS HERE ---
    # This is the full, correct get_history method from your Streamlit script,
    # adapted for our FastAPI backend.
    def get_history(self, filename: str) -> List[sqlite3.Row]:
        """Get analysis history for a specific filename."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT id, timestamp, category, filename, persona, full_analysis 
                    FROM analyses 
                    WHERE filename = ? 
                    ORDER BY timestamp DESC
                ''', (filename,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get history for {filename}: {e}")
            return []