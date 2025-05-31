import pandas as pd
import numpy as np
from typing import List, Dict
import re
from difflib import SequenceMatcher

class LesionGrouper:
    """Groups similar lesions based on naming patterns and similarity"""
    
    def __init__(self):
        self.similarity_threshold = 0.8
        
    def group_similar_lesions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group lesions with similar names into the same identifier"""
        if df.empty:
            return df
        
        df_grouped = df.copy()
        lesion_mapping = self._create_lesion_mapping(list(df['lesao_id'].unique()))
        
        # Apply mapping to standardize lesion names
        df_grouped['lesao_id'] = df_grouped['lesao_id'].map(
            lambda x: lesion_mapping.get(x, x)
        )
        
        return df_grouped
    
    def _create_lesion_mapping(self, lesion_names: List[str]) -> Dict[str, str]:
        """Create mapping from original names to standardized names"""
        mapping = {}
        groups = []
        
        for lesion in lesion_names:
            # Check if this lesion belongs to any existing group
            assigned = False
            for group in groups:
                if self._should_group_together(lesion, group[0]):
                    group.append(lesion)
                    assigned = True
                    break
            
            if not assigned:
                groups.append([lesion])
        
        # Create mapping with the "best" name for each group
        for group in groups:
            canonical_name = self._get_canonical_name(group)
            for lesion in group:
                mapping[lesion] = canonical_name
        
        return mapping
    
    def _should_group_together(self, lesion1: str, lesion2: str) -> bool:
        """Determine if two lesions should be grouped together"""
        # Normalize names
        norm1 = self._normalize_lesion_name(lesion1)
        norm2 = self._normalize_lesion_name(lesion2)
        
        # Check exact match after normalization
        if norm1 == norm2:
            return True
        
        # Check similarity score
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        if similarity >= self.similarity_threshold:
            return True
        
        # Check for common patterns
        return self._check_pattern_similarity(lesion1, lesion2)
    
    def _normalize_lesion_name(self, name: str) -> str:
        """Normalize lesion name for comparison"""
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Standardize common terms
        replacements = {
            'lesao': 'lesão',
            'nodulo': 'nódulo', 
            'metastase': 'metástase',
            'tumor': 'tumor',
            'massa': 'massa'
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def _check_pattern_similarity(self, lesion1: str, lesion2: str) -> bool:
        """Check if lesions follow similar naming patterns"""
        # Extract base name and identifier
        pattern1 = self._extract_pattern(lesion1)
        pattern2 = self._extract_pattern(lesion2)
        
        # If same base type with different identifiers, consider grouping
        if (pattern1['base'] == pattern2['base'] and 
            pattern1['base'] and 
            pattern1['identifier'] and 
            pattern2['identifier']):
            
            # Group if identifiers are similar (e.g., A vs a, 1 vs I)
            id1 = pattern1['identifier'].lower()
            id2 = pattern2['identifier'].lower()
            
            return id1 == id2 or self._similar_identifiers(id1, id2)
        
        return False
    
    def _extract_pattern(self, lesion: str) -> Dict[str, str]:
        """Extract base type and identifier from lesion name"""
        # Common patterns: "Lesão A", "Nódulo 1", "Metástase B"
        pattern = r'(lesão|nódulo|metástase|tumor|massa)\s*([a-z0-9]+)?'
        match = re.search(pattern, lesion.lower())
        
        if match:
            return {
                'base': match.group(1),
                'identifier': match.group(2) or ''
            }
        
        return {'base': '', 'identifier': ''}
    
    def _similar_identifiers(self, id1: str, id2: str) -> bool:
        """Check if identifiers are similar enough to group"""
        # Roman numerals vs letters/numbers
        roman_map = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5'}
        
        # Normalize identifiers
        norm_id1 = roman_map.get(id1, id1)
        norm_id2 = roman_map.get(id2, id2)
        
        return norm_id1 == norm_id2
    
    def _get_canonical_name(self, group: List[str]) -> str:
        """Get the best representative name for a group"""
        if len(group) == 1:
            return group[0]
        
        # Prefer names with clear patterns
        for name in group:
            if re.search(r'(lesão|nódulo|metástase|tumor|massa)\s+[a-z0-9]+', name.lower()):
                return name
        
        # Return the shortest clear name
        return min(group, key=len)