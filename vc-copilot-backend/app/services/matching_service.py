import pandas as pd
import re
from collections import Counter
from typing import List, Tuple, Dict, Any

class MatchingService:
    def __init__(self, data_path: str):
        self.df = self._load_data(data_path)
        self.master_industry_list = self._get_unique_industries()

    def _load_data(self, filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            for col in ['Fund Stage', 'Fund Focus (Sectors)', 'Fund Type']:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('')
            return df
        except FileNotFoundError:
            raise ValueError(f"Investor data file not found at: {filepath}")

    def _get_unique_industries(self) -> List[str]:
        all_industries = set()
        for industries_str in self.df['Fund Focus (Sectors)'].dropna():
            for industry in str(industries_str).split(','):
                clean_industry = industry.strip()
                if len(clean_industry) > 2 and not any(char.isdigit() for char in clean_industry):
                    all_industries.add(clean_industry.title())
        all_industries.add("General Consumer Goods")
        return sorted(list(all_industries))

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_counts = Counter(word for word in words if word not in stopwords)
        return [word for word, count in word_counts.most_common(max_keywords)]

    def find_similar_industries(self, keywords: List[str]) -> List[str]:
        similar_industries = []
        for keyword in keywords:
            matches = [ind for ind in self.master_industry_list if keyword in ind.lower()]
            for match in matches:
                if match not in similar_industries:
                    similar_industries.append(match)
        return similar_industries[:5]

    def _score_investor(self, row: pd.Series, stage: str, keywords: List[str], fund_type: str) -> int:
        score = 0
        clean_stage = stage.strip()
        clean_fund_type = fund_type.strip().lower()
        if stage not in str(row['Fund Stage']):
            return 0
        score += 25

        investor_focuses = [f.strip().lower() for f in str(row['Fund Focus (Sectors)']).split(',')]
        
        keyword_score = 0
        for keyword in keywords:
            if any(keyword == focus for focus in investor_focuses):
                keyword_score += 20
            elif any(keyword in focus for focus in investor_focuses):
                keyword_score += 10
        
        score += min(keyword_score, 40)

        if fund_type.lower() in str(row['Fund Type']).lower():
            score += 10
        
        if 0 < len(investor_focuses) <= 5:
            score += 5
            
        return score

    def find_top_investors(self, stage: str, keywords: List[str], fund_type: str, top_n: int = 4) -> pd.DataFrame:
        df_copy = self.df.copy()
        df_copy['score'] = df_copy.apply(self._score_investor, axis=1, args=(stage, keywords, fund_type))
        return df_copy[df_copy['score'] > 25].sort_values(by='score', ascending=False).head(top_n)

    def generate_insights(self, top_firms: pd.DataFrame) -> List[str]:
        if top_firms.empty:
            return ["No highly-matched investors found. Try broadening your criteria."]
        
        insights = []
        avg_score = top_firms['score'].mean()
        
        if avg_score > 50:
            insights.append("Excellent matches found! These investors align strongly with your startup.")
        else:
            insights.append("Good matches identified. Consider reaching out to these investors.")
        
        if top_firms['Fund Type'].nunique() > 1:
            insights.append(f"Diverse investor types found ({top_firms['Fund Type'].nunique()} different types)")
            
        return insights