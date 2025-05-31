import PyPDF2
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from openai_text_processor import OpenAITextProcessor

class PDFProcessor:
    """Handles PDF text extraction and medical data parsing"""
    
    def __init__(self):
        # Initialize OpenAI processor for enhanced extraction
        self.openai_processor = OpenAITextProcessor()
        
        # Fallback patterns for date recognition (DD/MM/YYYY and YYYY-MM-DD)
        self.date_patterns = [
            r'(?:Data\s+do\s+[Ee]xame|Data|Exame)[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
            r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
            r'(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})'
        ]
        
        # Fallback patterns for lesion identification and measurement
        self.lesion_patterns = [
            # Lesão A: 1,2 cm
            r'([Ll]es[ãa]o\s+[A-Za-z0-9]+)[:\s]*([0-9]+[,\.][0-9]+)\s*(cm|mm)',
            # Nódulo B: 0,8 cm
            r'([Nn][óo]dulo\s+[A-Za-z0-9]+)[:\s]*([0-9]+[,\.][0-9]+)\s*(cm|mm)',
            # Metástase C: 1.5 cm
            r'([Mm]et[áa]stase\s+[A-Za-z0-9]+)[:\s]*([0-9]+[,\.][0-9]+)\s*(cm|mm)',
            # Massa D: 2,3 cm
            r'([Mm]assa\s+[A-Za-z0-9]+)[:\s]*([0-9]+[,\.][0-9]+)\s*(cm|mm)',
            # Tumor E: 1,8 cm
            r'([Tt]umor\s+[A-Za-z0-9]+)[:\s]*([0-9]+[,\.][0-9]+)\s*(cm|mm)',
            # Generic pattern: <identifier>: <number> cm/mm
            r'([A-Za-z]+\s*[A-Za-z0-9]*)[:\s]*([0-9]+[,\.][0-9]+)\s*(cm|mm)'
        ]
        
        # Treatment keywords for context
        self.treatment_keywords = [
            'cirurgia', 'ressecção', 'quimioterapia', 'radioterapia', 
            'tratamento', 'remoção', 'excisão', 'ablação'
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
        
        except Exception as e:
            raise Exception(f"Erro ao extrair texto do PDF: {str(e)}")
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract exam date from text"""
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                date_str = matches[0]
                # Try to parse and standardize date
                return self._standardize_date(date_str)
        
        return None
    
    def _standardize_date(self, date_str: str) -> Optional[str]:
        """Convert date string to ISO format (YYYY-MM-DD)"""
        # Remove any extra whitespace
        date_str = date_str.strip()
        
        # Try different date formats
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y',
            '%Y/%m/%d', '%Y-%m-%d',
            '%m/%d/%Y', '%m-%d-%Y'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format matches, return None
        return None
    
    def extract_lesions(self, text: str) -> List[Tuple[str, float]]:
        """Extract lesion identifiers and measurements from text"""
        lesions = []
        
        for pattern in self.lesion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if len(match) == 3:
                    lesion_id, size_str, unit = match
                    
                    # Clean and standardize lesion identifier
                    lesion_id = self._clean_lesion_id(lesion_id)
                    
                    # Convert size to float and standardize to cm
                    size_cm = self._convert_to_cm(size_str, unit)
                    
                    if lesion_id and size_cm is not None:
                        lesions.append((lesion_id, size_cm))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_lesions = []
        for lesion_id, size in lesions:
            if lesion_id not in seen:
                unique_lesions.append((lesion_id, size))
                seen.add(lesion_id)
        
        return unique_lesions
    
    def _clean_lesion_id(self, lesion_id: str) -> str:
        """Clean and standardize lesion identifier"""
        # Remove extra whitespace and normalize
        lesion_id = re.sub(r'\s+', ' ', lesion_id.strip())
        
        # Capitalize first letter of each word
        words = lesion_id.split()
        cleaned_words = []
        
        for word in words:
            if word:
                cleaned_words.append(word.capitalize())
        
        return ' '.join(cleaned_words)
    
    def _convert_to_cm(self, size_str: str, unit: str) -> Optional[float]:
        """Convert size string to centimeters"""
        try:
            # Replace comma with dot for decimal parsing
            size_str = size_str.replace(',', '.')
            size_value = float(size_str)
            
            # Convert to cm if needed
            if unit.lower() == 'mm':
                size_value = size_value / 10
            
            return round(size_value, 2)
        
        except (ValueError, TypeError):
            return None
    
    def detect_treatment_context(self, text: str) -> List[str]:
        """Detect treatment mentions in the text"""
        treatments_found = []
        
        for keyword in self.treatment_keywords:
            if re.search(keyword, text, re.IGNORECASE):
                treatments_found.append(keyword.capitalize())
        
        return treatments_found
    
    def extract_lesion_data(self, pdf_path: str) -> List[Dict]:
        """Main method to extract all lesion data from a PDF"""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Extract exam date
            exam_date = self.extract_date(text)
            if not exam_date:
                raise Exception("Data do exame não encontrada no PDF")
            
            # Extract lesions
            lesions = self.extract_lesions(text)
            if not lesions:
                # Return empty list instead of raising exception
                # This allows processing to continue with other files
                return []
            
            # Detect treatment context
            treatments = self.detect_treatment_context(text)
            
            # Build data structure
            lesion_data = []
            for lesion_id, size_cm in lesions:
                data_point = {
                    'lesao_id': lesion_id,
                    'data_exame': exam_date,
                    'tamanho_cm': size_cm,
                    'tratamentos': treatments if treatments else []
                }
                lesion_data.append(data_point)
            
            return lesion_data
        
        except Exception as e:
            raise Exception(f"Erro ao processar PDF {pdf_path}: {str(e)}")
    
    def validate_pdf_content(self, pdf_path: str) -> Dict[str, bool]:
        """Validate if PDF contains required medical report elements"""
        try:
            text = self.extract_text_from_pdf(pdf_path)
            
            validation_results = {
                'has_date': self.extract_date(text) is not None,
                'has_lesions': len(self.extract_lesions(text)) > 0,
                'has_measurements': bool(re.search(r'\d+[,\.]\d+\s*(cm|mm)', text, re.IGNORECASE)),
                'is_medical_report': any(keyword in text.lower() for keyword in 
                                       ['laudo', 'exame', 'ressonância', 'tomografia', 'ultrassom', 'raio-x'])
            }
            
            return validation_results
        
        except Exception:
            return {
                'has_date': False,
                'has_lesions': False,
                'has_measurements': False,
                'is_medical_report': False
            }
