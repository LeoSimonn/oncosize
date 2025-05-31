import os
import tempfile
from typing import List, Dict, Optional
import streamlit as st
from openai import OpenAI
import json

class AudioProcessor:
    """Handles audio transcription and medical data extraction using OpenAI"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Supported audio formats
        self.supported_formats = ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']
        
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file to text using Whisper"""
        try:
            with open(audio_file_path, 'rb') as audio_file:
                response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="pt"  # Portuguese language
                )
            return response.text
        except Exception as e:
            raise Exception(f"Erro ao transcrever áudio: {str(e)}")
    
    def extract_medical_data_from_transcript(self, transcript: str) -> Dict:
        """Extract medical data from transcript using GPT-4o"""
        try:
            system_prompt = """
            Você é um especialista em análise de laudos médicos oncológicos. 
            Analise o texto transcrito de um laudo médico e extraia as seguintes informações:
            
            1. Data do exame (formato: YYYY-MM-DD)
            2. Lista de lesões com seus identificadores e medidas em centímetros
            3. Tratamentos mencionados
            
            Responda em formato JSON com a seguinte estrutura:
            {
                "data_exame": "YYYY-MM-DD",
                "lesoes": [
                    {
                        "identificador": "nome da lesão",
                        "tamanho_cm": número em centímetros
                    }
                ],
                "tratamentos": ["lista de tratamentos mencionados"],
                "observacoes": "observações importantes"
            }
            
            Se alguma informação não estiver disponível, use null.
            Converta todas as medidas para centímetros (mm → cm).
            """
            
            user_prompt = f"""
            Analise este texto transcrito de um laudo médico e extraia as informações estruturadas:
            
            {transcript}
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
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            raise Exception(f"Erro ao extrair dados médicos: {str(e)}")
    
    def process_audio_file(self, audio_file_path: str) -> List[Dict]:
        """Complete pipeline: transcribe audio and extract medical data"""
        try:
            # Step 1: Transcribe audio
            transcript = self.transcribe_audio(audio_file_path)
            
            # Step 2: Extract medical data
            medical_data = self.extract_medical_data_from_transcript(transcript)
            
            # Step 3: Convert to standard format
            lesion_data = []
            
            if medical_data.get('lesoes') and medical_data.get('data_exame'):
                for lesao in medical_data['lesoes']:
                    if lesao.get('identificador') and lesao.get('tamanho_cm'):
                        data_point = {
                            'lesao_id': lesao['identificador'],
                            'data_exame': medical_data['data_exame'],
                            'tamanho_cm': float(lesao['tamanho_cm']),
                            'tratamentos': medical_data.get('tratamentos', []),
                            'transcript': transcript[:200] + "..." if len(transcript) > 200 else transcript,
                            'observacoes': medical_data.get('observacoes', '')
                        }
                        lesion_data.append(data_point)
            
            return lesion_data
            
        except Exception as e:
            raise Exception(f"Erro ao processar arquivo de áudio: {str(e)}")
    
    def validate_audio_file(self, file_name: str) -> bool:
        """Validate if file is a supported audio format"""
        file_extension = file_name.lower().split('.')[-1]
        return file_extension in self.supported_formats
    
    def get_supported_formats_string(self) -> str:
        """Get comma-separated string of supported formats"""
        return ", ".join(self.supported_formats)
    
    def improve_transcript_quality(self, transcript: str) -> str:
        """Use GPT-4o to improve transcript quality and fix medical terminology"""
        try:
            system_prompt = """
            Você é um especialista em transcrição médica. Corrija e melhore este texto transcrito 
            de um laudo médico, focando em:
            
            1. Correção de terminologia médica oncológica
            2. Formatação adequada de datas e medidas
            3. Clareza na descrição de lesões
            4. Manutenção do significado original
            
            Retorne apenas o texto corrigido, sem comentários adicionais.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Corrija este texto: {transcript}"}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # If improvement fails, return original transcript
            return transcript