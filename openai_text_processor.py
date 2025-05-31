import os
import json
from typing import List, Dict, Optional
from openai import OpenAI

class OpenAITextProcessor:
    """Enhanced text processing using OpenAI for medical report analysis"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def extract_medical_data_from_text(self, text: str, filename: str = "") -> Dict:
        """Extract structured medical data from PDF text using GPT-4o"""
        try:
            system_prompt = """
            Você é um especialista em análise de laudos médicos oncológicos em português. 
            Analise o texto de um laudo médico e extraia as seguintes informações com máxima precisão:
            
            1. Data do exame (procure por padrões como "Data do Exame:", "Data:", formato DD/MM/AAAA ou AAAA-MM-DD)
            2. Todas as lesões encontradas com seus identificadores e medidas
            3. Tratamentos mencionados no texto
            
            IMPORTANTE:
            - Converta TODAS as medidas para centímetros (mm → cm dividindo por 10)
            - Identifique lesões por nomes como "Lesão A", "Nódulo B", "Metástase C", etc.
            - Procure por medidas em formato "X,X cm" ou "X mm"
            - Se encontrar várias medidas para uma lesão, use a primeira ou mais relevante
            
            Responda APENAS em formato JSON válido:
            {
                "data_exame": "YYYY-MM-DD ou null se não encontrada",
                "lesoes": [
                    {
                        "identificador": "nome exato da lesão encontrada no texto",
                        "tamanho_cm": número_decimal_em_centímetros
                    }
                ],
                "tratamentos": ["lista de tratamentos mencionados"],
                "confianca": número_de_0_a_1_indicando_confiança_na_extração
            }
            """
            
            user_prompt = f"""
            Analise este texto de laudo médico (arquivo: {filename}) e extraia as informações estruturadas:
            
            {text}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            if result_text:
                result = json.loads(result_text)
                return result
            else:
                return self._empty_result()
            
        except Exception as e:
            print(f"Erro ao processar texto com OpenAI: {str(e)}")
            return self._empty_result()
    
    def improve_extraction_with_context(self, text: str, previous_data: List[Dict] = None) -> Dict:
        """Improve extraction using context from previous reports"""
        try:
            context_info = ""
            if previous_data:
                lesion_names = set()
                for data in previous_data:
                    if 'lesao_id' in data:
                        lesion_names.add(data['lesao_id'])
                
                if lesion_names:
                    context_info = f"\nCONTEXTO: Este paciente já teve as seguintes lesões identificadas anteriormente: {', '.join(lesion_names)}. Use nomes consistentes quando possível."
            
            system_prompt = f"""
            Você é um especialista em análise de laudos médicos oncológicos em português. 
            Analise o texto com máxima precisão para extrair informações de lesões.
            
            {context_info}
            
            REGRAS IMPORTANTES:
            1. Para DATAS: procure padrões como "Data do Exame: DD/MM/AAAA", "Data: AAAA-MM-DD"
            2. Para LESÕES: identifique nomes como "Lesão A", "Nódulo B", "Metástase", "Massa", "Tumor"
            3. Para MEDIDAS: converta mm para cm (divida por 10), mantenha precisão decimal
            4. Se houver múltiplas medidas para uma lesão, use a mais específica
            5. Mantenha consistência com nomes de lesões anteriores quando aplicável
            
            Responda em JSON válido:
            {
                "data_exame": "YYYY-MM-DD",
                "lesoes": [
                    {
                        "identificador": "nome_da_lesão",
                        "tamanho_cm": valor_decimal
                    }
                ],
                "tratamentos": ["tratamentos_mencionados"],
                "confianca": valor_0_a_1
            }
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Texto do laudo:\n{text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content
            if result_text:
                return json.loads(result_text)
            else:
                return self._empty_result()
            
        except Exception as e:
            print(f"Erro na extração melhorada: {str(e)}")
            return self._empty_result()
    
    def validate_and_clean_data(self, extracted_data: Dict) -> Dict:
        """Validate and clean extracted data"""
        try:
            cleaned_data = {
                "data_exame": None,
                "lesoes": [],
                "tratamentos": [],
                "confianca": 0.0
            }
            
            # Validate date
            if extracted_data.get('data_exame') and extracted_data['data_exame'] != "null":
                cleaned_data["data_exame"] = extracted_data['data_exame']
            
            # Validate lesions
            if extracted_data.get('lesoes'):
                for lesao in extracted_data['lesoes']:
                    if (lesao.get('identificador') and 
                        lesao.get('tamanho_cm') and 
                        isinstance(lesao['tamanho_cm'], (int, float)) and 
                        lesao['tamanho_cm'] > 0):
                        
                        cleaned_data["lesoes"].append({
                            "identificador": str(lesao['identificador']).strip(),
                            "tamanho_cm": round(float(lesao['tamanho_cm']), 2)
                        })
            
            # Validate treatments
            if extracted_data.get('tratamentos'):
                cleaned_data["tratamentos"] = [
                    str(t).strip() for t in extracted_data['tratamentos'] 
                    if t and str(t).strip()
                ]
            
            # Validate confidence
            if extracted_data.get('confianca'):
                try:
                    confidence = float(extracted_data['confianca'])
                    cleaned_data["confianca"] = max(0.0, min(1.0, confidence))
                except:
                    cleaned_data["confianca"] = 0.5
            
            return cleaned_data
            
        except Exception as e:
            print(f"Erro na validação dos dados: {str(e)}")
            return self._empty_result()
    
    def convert_to_standard_format(self, extracted_data: Dict, source_file: str = "") -> List[Dict]:
        """Convert extracted data to standard format used by the system"""
        lesion_data = []
        
        if not extracted_data.get('lesoes') or not extracted_data.get('data_exame'):
            return lesion_data
        
        for lesao in extracted_data['lesoes']:
            data_point = {
                'lesao_id': lesao['identificador'],
                'data_exame': extracted_data['data_exame'],
                'tamanho_cm': lesao['tamanho_cm'],
                'tratamentos': extracted_data.get('tratamentos', []),
                'source_file': source_file,
                'confianca': extracted_data.get('confianca', 0.0)
            }
            lesion_data.append(data_point)
        
        return lesion_data
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            "data_exame": None,
            "lesoes": [],
            "tratamentos": [],
            "confianca": 0.0
        }