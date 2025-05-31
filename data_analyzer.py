import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

class DataAnalyzer:
    """Handles lesion data analysis and evolution tracking"""
    
    def __init__(self):
        self.variation_threshold = 10.0  # Threshold for significant variation (%)
    
    def analyze_lesion_evolution(self, df: pd.DataFrame) -> Dict:
        """
        Main analysis method that processes lesion data and calculates evolution
        
        Args:
            df: DataFrame with columns ['lesao_id', 'data_exame', 'tamanho_cm', 'tratamentos']
        
        Returns:
            Dictionary containing summary table, detailed data, and statistics
        """
        if df.empty:
            return self._empty_results()
        
        # Ensure data types are correct
        df = self._prepare_dataframe(df)
        
        # Generate detailed timeline data
        detailed_data = self._create_detailed_timeline(df)
        
        # Calculate summary statistics for each lesion
        summary_data = []
        
        for lesion_id in df['lesao_id'].unique():
            lesion_data = df[df['lesao_id'] == lesion_id].copy()
            lesion_summary = self._analyze_single_lesion(lesion_data)
            summary_data.append(lesion_summary)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics(summary_df)
        
        return {
            'summary_table': summary_df,
            'detailed_data': detailed_data,
            'overall_statistics': overall_stats,
            'analysis_metadata': {
                'total_lesions': len(summary_df),
                'total_measurements': len(df),
                'date_range': self._get_date_range(df),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate DataFrame for analysis"""
        df_clean = df.copy()
        
        # Convert date column to datetime
        df_clean['data_exame'] = pd.to_datetime(df_clean['data_exame'])
        
        # Ensure tamanho_cm is numeric
        df_clean['tamanho_cm'] = pd.to_numeric(df_clean['tamanho_cm'], errors='coerce')
        
        # Remove rows with invalid data
        df_clean = df_clean.dropna(subset=['data_exame', 'tamanho_cm'])
        
        # Sort by lesion and date
        df_clean = df_clean.sort_values(['lesao_id', 'data_exame'])
        
        return df_clean
    
    def _create_detailed_timeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create detailed timeline with consecutive variations"""
        detailed_data = []
        
        for lesion_id in df['lesao_id'].unique():
            lesion_data = df[df['lesao_id'] == lesion_id].copy()
            lesion_data = lesion_data.sort_values('data_exame')
            
            for i, row in lesion_data.iterrows():
                # Calculate variation from previous measurement
                variation_pct = None
                variation_abs = None
                
                if len(lesion_data) > 1:
                    prev_measurements = lesion_data[lesion_data['data_exame'] < row['data_exame']]
                    if not prev_measurements.empty:
                        prev_size = prev_measurements.iloc[-1]['tamanho_cm']
                        variation_abs = row['tamanho_cm'] - prev_size
                        variation_pct = (variation_abs / prev_size) * 100 if prev_size > 0 else None
                
                detailed_entry = {
                    'lesao_id': lesion_id,
                    'data_exame': row['data_exame'],
                    'tamanho_cm': row['tamanho_cm'],
                    'variacao_absoluta': variation_abs,
                    'variacao_percentual': variation_pct,
                    'tratamentos': row.get('tratamentos', [])
                }
                
                detailed_data.append(detailed_entry)
        
        return pd.DataFrame(detailed_data)
    
    def _analyze_single_lesion(self, lesion_data: pd.DataFrame) -> Dict:
        """Analyze evolution of a single lesion"""
        lesion_data = lesion_data.sort_values('data_exame')
        lesion_id = lesion_data.iloc[0]['lesao_id']
        
        # Basic measurements
        first_measurement = lesion_data.iloc[0]
        last_measurement = lesion_data.iloc[-1]
        
        first_date = first_measurement['data_exame']
        last_date = last_measurement['data_exame']
        first_size = first_measurement['tamanho_cm']
        last_size = last_measurement['tamanho_cm']
        
        # Calculate total variation
        total_variation_abs = last_size - first_size
        total_variation_pct = (total_variation_abs / first_size) * 100 if first_size > 0 else 0
        
        # Determine status
        status = self._determine_lesion_status(
            total_variation_pct, 
            first_size, 
            last_size,
            lesion_data
        )
        
        # Calculate trend statistics
        trend_stats = self._calculate_trend_statistics(lesion_data)
        
        return {
            'Lesão': lesion_id,
            'Primeira Data': first_date.strftime('%d/%m/%Y'),
            'Tamanho Inicial (cm)': first_size,
            'Última Data': last_date.strftime('%d/%m/%Y'),
            'Tamanho Final (cm)': last_size,
            'Variação Total (%)': total_variation_pct,
            'Status Atual': status,
            'Número de Medições': len(lesion_data),
            'Tendência': trend_stats['trend'],
            'Maior Tamanho (cm)': lesion_data['tamanho_cm'].max(),
            'Menor Tamanho (cm)': lesion_data['tamanho_cm'].min()
        }
    
    def _determine_lesion_status(self, variation_pct: float, first_size: float, 
                               last_size: float, lesion_data: pd.DataFrame) -> str:
        """Determine the current status of a lesion"""
        
        # Check if lesion disappeared (might be surgical removal)
        if len(lesion_data) > 1:
            # Check for treatments that might explain disappearance
            treatments = []
            for _, row in lesion_data.iterrows():
                if isinstance(row.get('tratamentos'), list):
                    treatments.extend(row['tratamentos'])
            
            # Look for surgical treatments
            surgical_treatments = [t for t in treatments if any(keyword in t.lower() 
                                 for keyword in ['cirurgia', 'ressecção', 'remoção', 'excisão'])]
        
        # Categorize based on variation percentage
        if abs(variation_pct) <= self.variation_threshold:
            return f"Estável ({variation_pct:+.1f}%)"
        elif variation_pct > self.variation_threshold:
            return f"Aumentou {variation_pct:+.1f}%"
        else:
            if surgical_treatments:
                return f"Reduziu {variation_pct:.1f}% (possível intervenção cirúrgica)"
            else:
                return f"Reduziu {variation_pct:.1f}%"
    
    def _calculate_trend_statistics(self, lesion_data: pd.DataFrame) -> Dict:
        """Calculate trend statistics for a lesion"""
        if len(lesion_data) < 2:
            return {'trend': 'Dados insuficientes', 'slope': 0, 'correlation': 0}
        
        # Convert dates to numeric for trend calculation
        lesion_data = lesion_data.sort_values('data_exame')
        date_numeric = pd.to_numeric(lesion_data['data_exame'])
        sizes = lesion_data['tamanho_cm'].values
        
        # Calculate linear trend
        try:
            correlation_matrix = np.corrcoef(date_numeric, sizes)
            correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0
            
            # Simple slope calculation
            x_mean = np.mean(date_numeric)
            y_mean = np.mean(sizes)
            
            numerator = np.sum((date_numeric - x_mean) * (sizes - y_mean))
            denominator = np.sum((date_numeric - x_mean) ** 2)
            
            slope = numerator / denominator if denominator != 0 else 0
            
            # Determine trend direction
            if abs(correlation) < 0.3:
                trend = "Sem tendência clara"
            elif correlation > 0.3:
                trend = "Tendência de crescimento"
            else:
                trend = "Tendência de redução"
            
            return {
                'trend': trend,
                'slope': slope,
                'correlation': correlation
            }
        
        except Exception:
            return {'trend': 'Erro no cálculo', 'slope': 0, 'correlation': 0}
    
    def _calculate_overall_statistics(self, summary_df: pd.DataFrame) -> Dict:
        """Calculate overall statistics across all lesions"""
        if summary_df.empty:
            return {}
        
        total_lesions = len(summary_df)
        
        # Count lesions by status
        increasing = len(summary_df[summary_df['Variação Total (%)'] > self.variation_threshold])
        decreasing = len(summary_df[summary_df['Variação Total (%)'] < -self.variation_threshold])
        stable = total_lesions - increasing - decreasing
        
        # Calculate statistics
        avg_variation = summary_df['Variação Total (%)'].mean()
        max_increase = summary_df['Variação Total (%)'].max()
        max_decrease = summary_df['Variação Total (%)'].min()
        
        # Find lesions with most significant changes
        most_increased = summary_df.loc[summary_df['Variação Total (%)'].idxmax()] if not summary_df.empty else None
        most_decreased = summary_df.loc[summary_df['Variação Total (%)'].idxmin()] if not summary_df.empty else None
        
        return {
            'total_lesions': total_lesions,
            'increasing_lesions': increasing,
            'decreasing_lesions': decreasing,
            'stable_lesions': stable,
            'average_variation_pct': avg_variation,
            'max_increase_pct': max_increase,
            'max_decrease_pct': max_decrease,
            'most_increased_lesion': most_increased['Lesão'] if most_increased is not None else None,
            'most_decreased_lesion': most_decreased['Lesão'] if most_decreased is not None else None
        }
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get the date range of the analysis"""
        if df.empty:
            return {'start_date': None, 'end_date': None}
        
        min_date = df['data_exame'].min()
        max_date = df['data_exame'].max()
        
        return {
            'start_date': min_date.strftime('%d/%m/%Y'),
            'end_date': max_date.strftime('%d/%m/%Y')
        }
    
    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'summary_table': pd.DataFrame(),
            'detailed_data': pd.DataFrame(),
            'overall_statistics': {},
            'analysis_metadata': {
                'total_lesions': 0,
                'total_measurements': 0,
                'date_range': {'start_date': None, 'end_date': None},
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def filter_lesions_by_criteria(self, summary_df: pd.DataFrame, 
                                 criteria: Dict[str, any]) -> pd.DataFrame:
        """Filter lesions based on specific criteria"""
        filtered_df = summary_df.copy()
        
        # Filter by variation threshold
        if 'min_variation' in criteria:
            filtered_df = filtered_df[abs(filtered_df['Variação Total (%)']) >= criteria['min_variation']]
        
        # Filter by status
        if 'status_contains' in criteria:
            filtered_df = filtered_df[filtered_df['Status Atual'].str.contains(criteria['status_contains'], na=False)]
        
        # Filter by size range
        if 'min_size' in criteria:
            filtered_df = filtered_df[filtered_df['Tamanho Final (cm)'] >= criteria['min_size']]
        
        if 'max_size' in criteria:
            filtered_df = filtered_df[filtered_df['Tamanho Final (cm)'] <= criteria['max_size']]
        
        return filtered_df
    
    def export_analysis_summary(self, analysis_results: Dict) -> str:
        """Export analysis summary as formatted text"""
        summary_df = analysis_results['summary_table']
        stats = analysis_results['overall_statistics']
        metadata = analysis_results['analysis_metadata']
        
        if summary_df.empty:
            return "Nenhuma lesão encontrada para análise."
        
        summary_text = f"""
RELATÓRIO DE ANÁLISE DE EVOLUÇÃO DE LESÕES
==========================================

Período de Análise: {metadata['date_range']['start_date']} a {metadata['date_range']['end_date']}
Total de Lesões: {metadata['total_lesions']}
Total de Medições: {metadata['total_measurements']}

RESUMO EXECUTIVO:
- Lesões com crescimento significativo: {stats.get('increasing_lesions', 0)}
- Lesões com redução significativa: {stats.get('decreasing_lesions', 0)}
- Lesões estáveis: {stats.get('stable_lesions', 0)}

Variação média: {stats.get('average_variation_pct', 0):.1f}%
Maior aumento: {stats.get('max_increase_pct', 0):.1f}% ({stats.get('most_increased_lesion', 'N/A')})
Maior redução: {stats.get('max_decrease_pct', 0):.1f}% ({stats.get('most_decreased_lesion', 'N/A')})

Gerado em: {metadata['analysis_timestamp']}
        """
        
        return summary_text.strip()
