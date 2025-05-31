import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict

class SyntheticDataGenerator:
    """Generate realistic synthetic medical data for demonstration"""
    
    def __init__(self):
        self.lesion_types = [
            "Lesão A", "Lesão B", "Lesão C", 
            "Nódulo Pulmonar", "Metástase Hepática", 
            "Massa Abdominal", "Tumor Cerebral",
            "Nódulo Axilar", "Lesão Óssea", "Massa Mediastinal"
        ]
        
        self.treatments = [
            ["Quimioterapia"], ["Radioterapia"], ["Cirurgia"],
            ["Quimioterapia", "Radioterapia"], ["Imunoterapia"],
            [], ["Terapia direcionada"]
        ]
    
    def generate_patient_data(self, num_exams: int = 6, num_lesions: int = 4) -> List[Dict]:
        """Generate synthetic patient data with realistic lesion evolution"""
        
        # Generate exam dates (every 2-3 months)
        start_date = datetime.now() - timedelta(days=365)
        exam_dates = []
        current_date = start_date
        
        for i in range(num_exams):
            exam_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=random.randint(60, 90))
        
        # Generate lesions with evolution patterns
        all_data = []
        selected_lesions = random.sample(self.lesion_types, num_lesions)
        
        for lesion_id in selected_lesions:
            # Define evolution pattern for this lesion
            evolution_type = random.choice(['growing', 'shrinking', 'stable', 'fluctuating'])
            initial_size = round(random.uniform(0.5, 3.0), 2)
            
            for i, exam_date in enumerate(exam_dates):
                size = self._calculate_lesion_size(initial_size, i, evolution_type, num_exams)
                
                # Some lesions may disappear after treatment
                if evolution_type == 'shrinking' and i > num_exams // 2 and random.random() < 0.3:
                    continue
                
                # Add treatment context
                treatments = []
                if i == num_exams // 3:  # Treatment starts after some exams
                    treatments = random.choice(self.treatments)
                
                data_point = {
                    'lesao_id': lesion_id,
                    'data_exame': exam_date,
                    'tamanho_cm': size,
                    'tratamentos': treatments,
                    'source_file': f'laudo_sintetico_{i+1}.pdf'
                }
                all_data.append(data_point)
        
        return all_data
    
    def _calculate_lesion_size(self, initial_size: float, exam_index: int, 
                              evolution_type: str, total_exams: int) -> float:
        """Calculate lesion size based on evolution pattern"""
        
        if evolution_type == 'growing':
            # Progressive growth with some variation
            growth_rate = random.uniform(0.1, 0.3)
            size = initial_size * (1 + growth_rate * exam_index)
            
        elif evolution_type == 'shrinking':
            # Progressive shrinkage, especially after treatment
            if exam_index < total_exams // 3:
                size = initial_size * random.uniform(0.95, 1.05)  # Stable initially
            else:
                shrink_rate = random.uniform(0.15, 0.4)
                size = initial_size * (1 - shrink_rate * (exam_index - total_exams // 3))
                
        elif evolution_type == 'stable':
            # Minimal variation around initial size
            variation = random.uniform(0.85, 1.15)
            size = initial_size * variation
            
        else:  # fluctuating
            # Random fluctuations
            if exam_index == 0:
                size = initial_size
            else:
                change = random.uniform(-0.3, 0.3)
                size = max(0.2, initial_size * (1 + change * exam_index / total_exams))
        
        # Add some random noise
        noise = random.uniform(0.95, 1.05)
        size = max(0.1, size * noise)
        
        return round(size, 2)
    
    def generate_demo_button_data(self) -> List[Dict]:
        """Generate quick demo data for button"""
        return [
            {'lesao_id': 'Lesão A', 'data_exame': '2024-01-15', 'tamanho_cm': 1.2, 'tratamentos': [], 'source_file': 'demo1.pdf'},
            {'lesao_id': 'Lesão A', 'data_exame': '2024-03-20', 'tamanho_cm': 1.5, 'tratamentos': [], 'source_file': 'demo2.pdf'},
            {'lesao_id': 'Lesão A', 'data_exame': '2024-05-10', 'tamanho_cm': 1.8, 'tratamentos': ['Quimioterapia'], 'source_file': 'demo3.pdf'},
            {'lesao_id': 'Lesão A', 'data_exame': '2024-07-25', 'tamanho_cm': 1.1, 'tratamentos': [], 'source_file': 'demo4.pdf'},
            
            {'lesao_id': 'Nódulo B', 'data_exame': '2024-01-15', 'tamanho_cm': 0.8, 'tratamentos': [], 'source_file': 'demo1.pdf'},
            {'lesao_id': 'Nódulo B', 'data_exame': '2024-03-20', 'tamanho_cm': 0.9, 'tratamentos': [], 'source_file': 'demo2.pdf'},
            {'lesao_id': 'Nódulo B', 'data_exame': '2024-05-10', 'tamanho_cm': 0.7, 'tratamentos': ['Radioterapia'], 'source_file': 'demo3.pdf'},
            {'lesao_id': 'Nódulo B', 'data_exame': '2024-07-25', 'tamanho_cm': 0.5, 'tratamentos': [], 'source_file': 'demo4.pdf'},
            
            {'lesao_id': 'Metástase C', 'data_exame': '2024-03-20', 'tamanho_cm': 2.1, 'tratamentos': [], 'source_file': 'demo2.pdf'},
            {'lesao_id': 'Metástase C', 'data_exame': '2024-05-10', 'tamanho_cm': 2.8, 'tratamentos': ['Quimioterapia'], 'source_file': 'demo3.pdf'},
            {'lesao_id': 'Metástase C', 'data_exame': '2024-07-25', 'tamanho_cm': 1.9, 'tratamentos': [], 'source_file': 'demo4.pdf'},
        ]