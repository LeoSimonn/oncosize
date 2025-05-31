import os
import psycopg2
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import json

class DatabaseManager:
    """Manages PostgreSQL database operations for lesion data"""
    
    def __init__(self):
        self.connection_params = {
            'host': os.environ.get('PGHOST'),
            'port': os.environ.get('PGPORT'),
            'database': os.environ.get('PGDATABASE'), 
            'user': os.environ.get('PGUSER'),
            'password': os.environ.get('PGPASSWORD')
        }
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.connection_params['host'],
            port=self.connection_params['port'],
            database=self.connection_params['database'],
            user=self.connection_params['user'],
            password=self.connection_params['password']
        )
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Create patients table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS patients (
                            id SERIAL PRIMARY KEY,
                            patient_id VARCHAR(100) UNIQUE NOT NULL,
                            name VARCHAR(200),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create lesions table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS lesions (
                            id SERIAL PRIMARY KEY,
                            patient_id VARCHAR(100) REFERENCES patients(patient_id),
                            lesion_id VARCHAR(200) NOT NULL,
                            exam_date DATE NOT NULL,
                            size_cm DECIMAL(10,2) NOT NULL,
                            treatments JSONB DEFAULT '[]',
                            observations TEXT,
                            source_file VARCHAR(500),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(patient_id, lesion_id, exam_date)
                        )
                    """)
                    
                    # Create analysis_sessions table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS analysis_sessions (
                            id SERIAL PRIMARY KEY,
                            patient_id VARCHAR(100) REFERENCES patients(patient_id),
                            session_data JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    conn.commit()
        except Exception as e:
            print(f"Erro ao inicializar banco de dados: {str(e)}")
    
    def save_patient(self, patient_id: str, name: str = None) -> bool:
        """Save or update patient information"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO patients (patient_id, name) 
                        VALUES (%s, %s) 
                        ON CONFLICT (patient_id) 
                        DO UPDATE SET name = EXCLUDED.name
                    """, (patient_id, name))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Erro ao salvar paciente: {str(e)}")
            return False
    
    def save_lesion_data(self, patient_id: str, lesion_data: List[Dict]) -> bool:
        """Save lesion measurements to database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    for data in lesion_data:
                        cursor.execute("""
                            INSERT INTO lesions (patient_id, lesion_id, exam_date, size_cm, treatments, observations, source_file)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (patient_id, lesion_id, exam_date)
                            DO UPDATE SET 
                                size_cm = EXCLUDED.size_cm,
                                treatments = EXCLUDED.treatments,
                                observations = EXCLUDED.observations,
                                source_file = EXCLUDED.source_file
                        """, (
                            patient_id,
                            data['lesao_id'],
                            data['data_exame'],
                            data['tamanho_cm'],
                            json.dumps(data.get('tratamentos', [])),
                            data.get('observacoes', ''),
                            data.get('source_file', '')
                        ))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Erro ao salvar dados de lesão: {str(e)}")
            return False
    
    def get_patient_data(self, patient_id: str) -> pd.DataFrame:
        """Get all lesion data for a patient"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT lesion_id, exam_date, size_cm, treatments, observations, source_file
                    FROM lesions 
                    WHERE patient_id = %s 
                    ORDER BY lesion_id, exam_date
                """
                df = pd.read_sql_query(query, conn, params=[patient_id])
                
                # Rename columns to match existing code
                df = df.rename(columns={
                    'lesion_id': 'lesao_id',
                    'exam_date': 'data_exame', 
                    'size_cm': 'tamanho_cm',
                    'treatments': 'tratamentos'
                })
                
                # Parse JSON treatments
                if not df.empty:
                    df['tratamentos'] = df['tratamentos'].apply(
                        lambda x: json.loads(x) if x else []
                    )
                
                return df
        except Exception as e:
            print(f"Erro ao recuperar dados do paciente: {str(e)}")
            return pd.DataFrame()
    
    def get_all_patients(self) -> List[Dict]:
        """Get list of all patients"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT patient_id, name, created_at FROM patients ORDER BY created_at DESC")
                    patients = []
                    for row in cursor.fetchall():
                        patients.append({
                            'patient_id': row[0],
                            'name': row[1],
                            'created_at': row[2]
                        })
                    return patients
        except Exception as e:
            print(f"Erro ao recuperar lista de pacientes: {str(e)}")
            return []
    
    def save_analysis_session(self, patient_id: str, analysis_data: Dict) -> bool:
        """Save analysis session data"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO analysis_sessions (patient_id, session_data)
                        VALUES (%s, %s)
                    """, (patient_id, json.dumps(analysis_data, default=str)))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Erro ao salvar sessão de análise: {str(e)}")
            return False
    
    def delete_patient_data(self, patient_id: str) -> bool:
        """Delete all data for a patient"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM analysis_sessions WHERE patient_id = %s", (patient_id,))
                    cursor.execute("DELETE FROM lesions WHERE patient_id = %s", (patient_id,))
                    cursor.execute("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Erro ao deletar dados do paciente: {str(e)}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Count patients
                    cursor.execute("SELECT COUNT(*) FROM patients")
                    patient_count = cursor.fetchone()[0]
                    
                    # Count lesions
                    cursor.execute("SELECT COUNT(*) FROM lesions")
                    lesion_count = cursor.fetchone()[0]
                    
                    # Count unique lesions
                    cursor.execute("SELECT COUNT(DISTINCT lesion_id) FROM lesions")
                    unique_lesions = cursor.fetchone()[0]
                    
                    # Date range
                    cursor.execute("SELECT MIN(exam_date), MAX(exam_date) FROM lesions")
                    date_range = cursor.fetchone()
                    
                    return {
                        'total_patients': patient_count,
                        'total_measurements': lesion_count,
                        'unique_lesions': unique_lesions,
                        'date_range': {
                            'start': date_range[0].strftime('%d/%m/%Y') if date_range[0] else None,
                            'end': date_range[1].strftime('%d/%m/%Y') if date_range[1] else None
                        }
                    }
        except Exception as e:
            print(f"Erro ao recuperar estatísticas: {str(e)}")
            return {}