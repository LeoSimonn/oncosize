import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import streamlit as st

class TreatmentManager:
    """Manages treatment periods and correlates them with lesion evolution"""
    
    def __init__(self):
        self.treatment_types = [
            "Quimioterapia",
            "Radioterapia", 
            "Imunoterapia",
            "Terapia Direcionada",
            "Hormonioterapia",
            "Cirurgia",
            "Transplante de Medula Óssea",
            "Crioterapia",
            "Ablação por Radiofrequência",
            "Outro"
        ]
    
    def display_treatment_input_interface(self):
        """Display interface for inputting treatment periods"""
        st.subheader("💊 Registro de Tratamentos")
        st.write("Registre os períodos de tratamento para correlacionar com a evolução das lesões")
        
        # Initialize treatment data in session state
        if 'treatment_periods' not in st.session_state:
            st.session_state.treatment_periods = []
        
        # Input form for new treatment
        with st.expander("➕ Adicionar Novo Tratamento", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                treatment_type = st.selectbox(
                    "Tipo de Tratamento",
                    self.treatment_types,
                    key="new_treatment_type"
                )
                
                if treatment_type == "Outro":
                    treatment_type = st.text_input(
                        "Especificar tratamento",
                        key="custom_treatment"
                    )
                
                start_date = st.date_input(
                    "Data de Início",
                    key="treatment_start"
                )
            
            with col2:
                medication_name = st.text_input(
                    "Nome do Medicamento/Protocolo",
                    placeholder="Ex: Carboplatina + Paclitaxel",
                    key="medication_name"
                )
                
                end_date = st.date_input(
                    "Data de Término",
                    key="treatment_end",
                    help="Deixe em branco se o tratamento ainda está em curso"
                )
            
            dosage = st.text_input(
                "Dosagem/Frequência",
                placeholder="Ex: 200mg/m² a cada 3 semanas",
                key="treatment_dosage"
            )
            
            notes = st.text_area(
                "Observações",
                placeholder="Efeitos colaterais, mudanças de dose, etc.",
                key="treatment_notes"
            )
            
            if st.button("Adicionar Tratamento", type="primary"):
                self.add_treatment_period(
                    treatment_type, medication_name, start_date, 
                    end_date, dosage, notes
                )
        
        # Display existing treatments
        if st.session_state.treatment_periods:
            st.subheader("📋 Tratamentos Registrados")
            self.display_treatment_list()
    
    def add_treatment_period(self, treatment_type: str, medication: str, 
                           start_date, end_date, dosage: str, notes: str):
        """Add a new treatment period"""
        try:
            treatment = {
                "id": len(st.session_state.treatment_periods) + 1,
                "tipo": treatment_type,
                "medicamento": medication,
                "data_inicio": start_date.strftime('%Y-%m-%d') if start_date else None,
                "data_fim": end_date.strftime('%Y-%m-%d') if end_date else None,
                "dosagem": dosage,
                "observacoes": notes,
                "ativo": end_date is None or end_date >= datetime.now().date()
            }
            
            st.session_state.treatment_periods.append(treatment)
            st.success(f"Tratamento {treatment_type} adicionado com sucesso!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro ao adicionar tratamento: {str(e)}")
    
    def display_treatment_list(self):
        """Display list of registered treatments"""
        for i, treatment in enumerate(st.session_state.treatment_periods):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    status = "🟢 Ativo" if treatment["ativo"] else "🔴 Finalizado"
                    st.write(f"**{treatment['tipo']}** - {treatment['medicamento']} ({status})")
                    
                    periodo = f"📅 {treatment['data_inicio']}"
                    if treatment['data_fim']:
                        periodo += f" até {treatment['data_fim']}"
                    else:
                        periodo += " - Em andamento"
                    st.write(periodo)
                    
                    if treatment['dosagem']:
                        st.write(f"💊 {treatment['dosagem']}")
                    
                    if treatment['observacoes']:
                        st.write(f"📝 {treatment['observacoes']}")
                
                with col2:
                    if st.button("✏️", key=f"edit_{i}", help="Editar tratamento"):
                        self.edit_treatment(i)
                
                with col3:
                    if st.button("🗑️", key=f"delete_{i}", help="Remover tratamento"):
                        self.delete_treatment(i)
                
                st.divider()
    
    def edit_treatment(self, index: int):
        """Edit existing treatment"""
        st.info("Funcionalidade de edição será implementada em breve")
    
    def delete_treatment(self, index: int):
        """Delete treatment"""
        try:
            del st.session_state.treatment_periods[index]
            st.success("Tratamento removido com sucesso!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao remover tratamento: {str(e)}")
    
    def correlate_treatments_with_lesions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correlate treatment periods with lesion measurements"""
        if not st.session_state.get('treatment_periods'):
            return df
        
        df_with_treatments = df.copy()
        df_with_treatments['tratamentos_periodo'] = ""
        
        for index, row in df_with_treatments.iterrows():
            exam_date = pd.to_datetime(row['data_exame']).date()
            active_treatments = []
            
            for treatment in st.session_state.treatment_periods:
                start_date = pd.to_datetime(treatment['data_inicio']).date()
                end_date = pd.to_datetime(treatment['data_fim']).date() if treatment['data_fim'] else datetime.now().date()
                
                if start_date <= exam_date <= end_date:
                    treatment_info = f"{treatment['tipo']}"
                    if treatment['medicamento']:
                        treatment_info += f" ({treatment['medicamento']})"
                    active_treatments.append(treatment_info)
            
            df_with_treatments.at[index, 'tratamentos_periodo'] = "; ".join(active_treatments)
        
        return df_with_treatments
    
    def get_treatment_timeline(self) -> List[Dict]:
        """Get timeline of treatments for visualization"""
        if not st.session_state.get('treatment_periods'):
            return []
        
        timeline = []
        for treatment in st.session_state.treatment_periods:
            timeline.append({
                "nome": f"{treatment['tipo']} - {treatment['medicamento']}",
                "inicio": treatment['data_inicio'],
                "fim": treatment['data_fim'],
                "ativo": treatment['ativo']
            })
        
        return timeline
    
    def export_treatments_to_dict(self) -> Dict:
        """Export treatments for analysis storage"""
        return {
            "treatment_periods": st.session_state.get('treatment_periods', []),
            "export_timestamp": datetime.now().isoformat()
        }