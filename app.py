import streamlit as st
import pandas as pd
import os
import tempfile
from pdf_processor import PDFProcessor
from data_analyzer import DataAnalyzer
from visualization import VisualizationGenerator
from utils import format_summary_table, format_detailed_table
from synthetic_data_generator import SyntheticDataGenerator
from lesion_grouper import LesionGrouper
import io

def main():
    st.set_page_config(
        page_title="Análise de Evolução de Lesões Oncológicas",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Análise de Evolução de Lesões Oncológicas")
    st.markdown("**Sistema de análise de laudos médicos em PDF para acompanhamento da evolução de lesões ao longo do tempo**")
    st.markdown("*Processamento inteligente com IA da OpenAI para extração precisa de dados médicos*")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'charts_generated' not in st.session_state:
        st.session_state.charts_generated = False
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload de Laudos")
        uploaded_files = st.file_uploader(
            "Selecione os arquivos PDF dos laudos médicos",
            type=['pdf'],
            accept_multiple_files=True,
            help="Faça upload de múltiplos laudos médicos em PDF do mesmo paciente"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} arquivo(s) carregado(s)")
            
            if st.button("🔍 Processar Laudos", type="primary"):
                process_files(uploaded_files)
        
        st.divider()
        
        # Demo data for testing
        st.subheader("🧪 Dados para Demonstração")
        st.info("Para testar o sistema sem PDFs, use dados sintéticos")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Gerar Dados Demo", help="Dados sintéticos para demonstração"):
                generate_demo_data()
        
        with col2:
            if st.button("🔄 Dados Completos", help="Dataset completo para análise"):
                generate_full_demo_data()
        
        # Option to clear current session
        if st.session_state.processed_data is not None:
            st.divider()
            if st.button("🗑️ Limpar Dados Atuais", type="secondary"):
                st.session_state.processed_data = None
                st.session_state.analysis_results = None
                st.session_state.charts_generated = False
                st.rerun()
    
    # Main content area
    if st.session_state.processed_data is not None and st.session_state.analysis_results is not None:
        display_results()
    else:
        display_instructions()

def process_files(uploaded_files):
    """Process uploaded PDF files and analyze lesion data"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize processors
        pdf_processor = PDFProcessor()
        data_analyzer = DataAnalyzer()
        
        # Process each PDF file
        all_data = []
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processando arquivo {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            
            try:
                # Extract data from PDF
                file_data = pdf_processor.extract_lesion_data(tmp_file_path)
                if file_data:
                    # Add source file info
                    for data_point in file_data:
                        data_point['source_file'] = uploaded_file.name
                    
                    all_data.extend(file_data)
                    st.success(f"✅ {uploaded_file.name}: {len(file_data)} medições extraídas com IA")
                else:
                    st.warning(f"⚠️ {uploaded_file.name}: Nenhuma medição encontrada")
            except Exception as e:
                st.error(f"❌ Erro ao processar {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Group similar lesions
            status_text.text("Agrupando lesões similares...")
            lesion_grouper = LesionGrouper()
            df_grouped = lesion_grouper.group_similar_lesions(df)
            
            # Analyze data
            status_text.text("Analisando dados e calculando variações...")
            analysis_results = data_analyzer.analyze_lesion_evolution(df_grouped)
            
            # Store results in session state
            st.session_state.processed_data = df_grouped
            st.session_state.analysis_results = analysis_results
            st.session_state.charts_generated = False
            
            status_text.text("✅ Processamento concluído!")
            progress_bar.progress(1.0)
            
            # Auto-rerun to show results
            st.rerun()
        else:
            st.error("❌ Nenhum dado de lesão foi encontrado nos arquivos enviados")
            
    except Exception as e:
        st.error(f"❌ Erro durante o processamento: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def generate_demo_data():
    """Generate demo data for testing"""
    try:
        synthetic_generator = SyntheticDataGenerator()
        demo_data = synthetic_generator.generate_demo_button_data()
        
        # Convert to DataFrame and group lesions
        df = pd.DataFrame(demo_data)
        lesion_grouper = LesionGrouper()
        df_grouped = lesion_grouper.group_similar_lesions(df)
        
        # Analyze data
        data_analyzer = DataAnalyzer()
        analysis_results = data_analyzer.analyze_lesion_evolution(df_grouped)
        
        # Store in session state
        st.session_state.processed_data = df_grouped
        st.session_state.analysis_results = analysis_results
        st.session_state.charts_generated = False
        
        st.success("✅ Dados de demonstração carregados!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Erro ao gerar dados demo: {str(e)}")

def generate_full_demo_data():
    """Generate comprehensive demo data"""
    try:
        synthetic_generator = SyntheticDataGenerator()
        full_data = synthetic_generator.generate_patient_data(num_exams=8, num_lesions=6)
        
        # Convert to DataFrame and group lesions
        df = pd.DataFrame(full_data)
        lesion_grouper = LesionGrouper()
        df_grouped = lesion_grouper.group_similar_lesions(df)
        
        # Analyze data
        data_analyzer = DataAnalyzer()
        analysis_results = data_analyzer.analyze_lesion_evolution(df_grouped)
        
        # Store in session state
        st.session_state.processed_data = df_grouped
        st.session_state.analysis_results = analysis_results
        st.session_state.charts_generated = False
        
        st.success("✅ Dataset completo carregado!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Erro ao gerar dados completos: {str(e)}")

def display_results():
    """Display analysis results and visualizations"""
    df = st.session_state.processed_data
    analysis_results = st.session_state.analysis_results
    
    # Generate summary text
    summary_text = generate_summary_text(analysis_results)
    
    # Display summary
    st.header("📊 Resumo Executivo")
    st.info(summary_text)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualizações", "📋 Tabela Resumida", "📅 Dados Detalhados", "💾 Downloads"])
    
    with tab1:
        display_visualizations(analysis_results)
    
    with tab2:
        display_summary_table(analysis_results)
    
    with tab3:
        display_detailed_data(df)
    
    with tab4:
        display_download_options(df, analysis_results)

def display_visualizations(analysis_results):
    """Display charts and visualizations"""
    viz_generator = VisualizationGenerator()
    
    st.subheader("📈 Evolução das Lesões")
    
    # Filter options
    available_lesions = list(analysis_results['summary_table']['Lesão'].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        show_combined = st.checkbox("Mostrar gráfico combinado", value=True)
    with col2:
        selected_lesions = st.multiselect(
            "Filtrar lesões específicas",
            available_lesions,
            default=available_lesions
        )
    
    if selected_lesions:
        # Generate and display charts
        if show_combined:
            st.subheader("Comparação de Todas as Lesões")
            fig_combined = viz_generator.create_combined_chart(
                analysis_results['detailed_data'], 
                selected_lesions
            )
            st.pyplot(fig_combined)
        
        # Individual charts
        st.subheader("Gráficos Individuais")
        for lesion in selected_lesions:
            lesion_data = analysis_results['detailed_data'][
                analysis_results['detailed_data']['lesao_id'] == lesion
            ]
            if not lesion_data.empty:
                fig_individual = viz_generator.create_individual_chart(lesion_data, lesion)
                st.pyplot(fig_individual)
    else:
        st.warning("Selecione pelo menos uma lesão para visualizar")

def display_summary_table(analysis_results):
    """Display summary table"""
    st.subheader("📋 Tabela Resumida")
    
    summary_df = analysis_results['summary_table']
    
    # Format the dataframe for better display
    formatted_df = summary_df.copy()
    formatted_df['Tamanho Inicial (cm)'] = formatted_df['Tamanho Inicial (cm)'].apply(lambda x: f"{x:.2f}")
    formatted_df['Tamanho Final (cm)'] = formatted_df['Tamanho Final (cm)'].apply(lambda x: f"{x:.2f}")
    formatted_df['Variação Total (%)'] = formatted_df['Variação Total (%)'].apply(lambda x: f"{x:+.1f}%")
    
    st.dataframe(formatted_df, use_container_width=True)
    
    # Export as markdown
    if st.button("📝 Gerar Tabela em Markdown"):
        markdown_table = format_summary_table(summary_df)
        st.text_area("Tabela em formato Markdown", markdown_table, height=300)

def display_detailed_data(df):
    """Display detailed measurement data"""
    st.subheader("📅 Dados Detalhados por Data")
    
    # Sort by date and lesion
    sorted_df = df.sort_values(['data_exame', 'lesao_id'])
    
    # Format for display
    display_df = sorted_df.copy()
    display_df['tamanho_cm'] = display_df['tamanho_cm'].apply(lambda x: f"{x:.2f}")
    display_df['data_exame'] = pd.to_datetime(display_df['data_exame']).dt.strftime('%d/%m/%Y')
    
    # Rename columns for Portuguese
    display_df = display_df.rename(columns={
        'lesao_id': 'Lesão',
        'data_exame': 'Data do Exame',
        'tamanho_cm': 'Tamanho (cm)'
    })
    
    st.dataframe(display_df, use_container_width=True)

def display_download_options(df, analysis_results):
    """Display download options for data and charts"""
    st.subheader("💾 Downloads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Dados em CSV**")
        
        # Summary table CSV
        summary_csv = analysis_results['summary_table'].to_csv(index=False)
        st.download_button(
            label="📋 Baixar Tabela Resumida (CSV)",
            data=summary_csv,
            file_name="resumo_lesoes.csv",
            mime="text/csv"
        )
        
        # Detailed data CSV
        detailed_csv = df.to_csv(index=False)
        st.download_button(
            label="📅 Baixar Dados Detalhados (CSV)",
            data=detailed_csv,
            file_name="dados_detalhados.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("**📈 Gráficos**")
        
        if st.button("🖼️ Gerar Gráficos PNG"):
            generate_chart_downloads(analysis_results)

def generate_chart_downloads(analysis_results):
    """Generate and offer chart downloads"""
    viz_generator = VisualizationGenerator()
    
    try:
        # Generate combined chart
        fig_combined = viz_generator.create_combined_chart(
            analysis_results['detailed_data'], 
            list(analysis_results['summary_table']['Lesão'].unique())
        )
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        fig_combined.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        
        st.download_button(
            label="📈 Baixar Gráfico Combinado (PNG)",
            data=img_buffer.getvalue(),
            file_name="grafico_todas_lesoes.png",
            mime="image/png"
        )
        
        st.success("✅ Gráficos gerados com sucesso!")
        
    except Exception as e:
        st.error(f"❌ Erro ao gerar gráficos: {str(e)}")

def generate_summary_text(analysis_results):
    """Generate executive summary text"""
    summary_df = analysis_results['summary_table']
    
    if summary_df.empty:
        return "Nenhuma lesão foi encontrada nos laudos processados."
    
    total_lesions = len(summary_df)
    increased_lesions = len(summary_df[summary_df['Variação Total (%)'] > 10])
    decreased_lesions = len(summary_df[summary_df['Variação Total (%)'] < -10])
    stable_lesions = len(summary_df[abs(summary_df['Variação Total (%)']) <= 10])
    
    # Find most significant changes
    max_increase = summary_df.loc[summary_df['Variação Total (%)'].idxmax()] if not summary_df.empty else None
    max_decrease = summary_df.loc[summary_df['Variação Total (%)'].idxmin()] if not summary_df.empty else None
    
    summary_parts = [
        f"**Análise de {total_lesions} lesão(ões) encontrada(s):**",
        f"• {increased_lesions} lesão(ões) com crescimento significativo (>10%)",
        f"• {decreased_lesions} lesão(ões) com redução significativa (>10%)",
        f"• {stable_lesions} lesão(ões) estável(eis) (±10%)"
    ]
    
    if max_increase is not None and max_increase['Variação Total (%)'] > 0:
        summary_parts.append(f"• **Maior crescimento:** {max_increase['Lesão']} (+{max_increase['Variação Total (%)']:.1f}%)")
    
    if max_decrease is not None and max_decrease['Variação Total (%)'] < 0:
        summary_parts.append(f"• **Maior redução:** {max_decrease['Lesão']} ({max_decrease['Variação Total (%)']:.1f}%)")
    
    return "\n".join(summary_parts)

def display_instructions():
    """Display instructions when no data is loaded"""
    st.markdown("""
    ## 📋 Como usar este sistema
    
    ### 1. **Upload de Arquivos**
    - Use a barra lateral para fazer upload dos laudos médicos em PDF
    - Carregue múltiplos arquivos de um mesmo paciente
    - Os arquivos devem conter data do exame e descrições de lesões
    
    ### 2. **Formato dos Laudos**
    Os laudos devem conter:
    - **Data do exame** (formatos aceitos: DD/MM/AAAA ou AAAA-MM-DD)
    - **Descrições de lesões** com medidas (ex: "Lesão A: 1,2 cm", "Nódulo B: 8 mm")
    
    ### 3. **Processamento Inteligente**
    O sistema usa IA da OpenAI para:
    - ✅ Extrair automaticamente datas e medições do texto
    - ✅ Identificar lesões com maior precisão
    - ✅ Padronizar unidades (mm → cm)
    - ✅ Detectar tratamentos mencionados
    - ✅ Fallback para padrões regex se necessário
    
    ### 4. **Análise Gerada**
    - ✅ Calcular variações percentuais entre exames
    - ✅ Gerar tabelas resumidas e detalhadas
    - ✅ Criar gráficos de evolução temporal
    - ✅ Identificar tendências de crescimento/regressão
    
    ### 5. **Resultados**
    - **Resumo executivo** com principais achados
    - **Visualizações** interativas da evolução
    - **Tabelas** em formato Markdown/CSV
    - **Downloads** de dados e gráficos
    
    ---
    👆 **Comece fazendo upload dos arquivos PDF na barra lateral**
    """)

if __name__ == "__main__":
    main()
