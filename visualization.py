import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict
import seaborn as sns

# Set matplotlib style for medical reports
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'

class VisualizationGenerator:
    """Handles chart generation for lesion evolution analysis"""
    
    def __init__(self):
        # Configure matplotlib for better display
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 100
        
        # Color palette for different lesions
        self.color_palette = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#8e44ad'
        ]
    
    def create_individual_chart(self, lesion_data: pd.DataFrame, lesion_name: str):
        """Create individual evolution chart for a single lesion"""
        
        if lesion_data.empty:
            return self._create_empty_chart(f"Sem dados para {lesion_name}")
        
        # Sort data by date
        lesion_data = lesion_data.sort_values('data_exame')
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert dates for plotting
        dates = pd.to_datetime(lesion_data['data_exame'])
        sizes = lesion_data['tamanho_cm']
        
        # Add treatment periods as background
        self._add_treatment_periods(ax)
        
        # Plot main line and points
        ax.plot(dates, sizes, marker='o', linewidth=2.5, markersize=8, 
                color=self.color_palette[0], label=lesion_name, zorder=5)
        
        # Add data point labels
        for i, (date, size) in enumerate(zip(dates, sizes)):
            ax.annotate(f'{size:.2f} cm', 
                       (date, size), 
                       textcoords="offset points", 
                       xytext=(0, 15), 
                       ha='center',
                       fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                       zorder=6)
        
        # Highlight significant changes
        self._highlight_significant_changes(ax, dates, sizes)
        
        # Formatting
        ax.set_title(f'Evolução da {lesion_name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Data do Exame', fontsize=12)
        ax.set_ylabel('Tamanho (cm)', fontsize=12)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # Add trend line if there are enough points
        if len(dates) > 2:
            self._add_trend_line(ax, dates, sizes)
        
        # Add statistics box
        self._add_statistics_box(ax, lesion_data, lesion_name)
        
        plt.tight_layout()
        return fig
    
    def _add_treatment_periods(self, ax):
        """Add treatment periods as colored background regions"""
        import streamlit as st
        
        # Check if treatment periods are available in session state
        if not st.session_state.get('treatment_periods'):
            return
        
        # Get y-axis limits for background shading
        y_min, y_max = ax.get_ylim()
        
        # Treatment type colors
        treatment_colors = {
            'Quimioterapia': '#ffcccc',
            'Radioterapia': '#ccffcc', 
            'Imunoterapia': '#ccccff',
            'Terapia Direcionada': '#ffffcc',
            'Hormonioterapia': '#ffccff',
            'Cirurgia': '#ccffff',
            'Outro': '#f0f0f0'
        }
        
        for i, treatment in enumerate(st.session_state.treatment_periods):
            try:
                start_date = pd.to_datetime(treatment['data_inicio'])
                end_date = pd.to_datetime(treatment['data_fim']) if treatment['data_fim'] else pd.Timestamp.now()
                
                # Get color for treatment type
                color = treatment_colors.get(treatment['tipo'], treatment_colors['Outro'])
                
                # Add background span
                ax.axvspan(start_date, end_date, alpha=0.3, color=color, zorder=1)
                
                # Add treatment label
                mid_date = start_date + (end_date - start_date) / 2
                label_text = f"{treatment['tipo']}"
                if treatment['medicamento']:
                    label_text = f"{treatment['tipo']}\n({treatment['medicamento']})"
                
                # Add text annotation at the top
                ax.text(mid_date, y_max * 0.95, label_text, 
                       ha='center', va='top', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                       rotation=0, zorder=10)
                       
            except Exception as e:
                continue
    
    def create_combined_chart(self, detailed_data: pd.DataFrame, 
                            selected_lesions: List[str]) -> plt.Figure:
        """Create combined chart showing evolution of multiple lesions"""
        
        if detailed_data.empty:
            return self._create_empty_chart("Sem dados para exibir")
        
        # Filter data for selected lesions
        filtered_data = detailed_data[detailed_data['lesao_id'].isin(selected_lesions)]
        
        if filtered_data.empty:
            return self._create_empty_chart("Nenhuma lesão selecionada tem dados")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Add treatment periods as background first
        self._add_treatment_periods(ax)
        
        # Plot each lesion
        for i, lesion_id in enumerate(selected_lesions):
            lesion_data = filtered_data[filtered_data['lesao_id'] == lesion_id]
            
            if not lesion_data.empty:
                lesion_data = lesion_data.sort_values('data_exame')
                dates = pd.to_datetime(lesion_data['data_exame'])
                sizes = lesion_data['tamanho_cm']
                
                color = self.color_palette[i % len(self.color_palette)]
                
                ax.plot(dates, sizes, marker='o', linewidth=2.5, markersize=6,
                       color=color, label=lesion_id, alpha=0.8)
                
                # Add last measurement annotation
                if len(dates) > 0:
                    last_date = dates.iloc[-1]
                    last_size = sizes.iloc[-1]
                    ax.annotate(f'{last_size:.2f}', 
                               (last_date, last_size),
                               textcoords="offset points",
                               xytext=(10, 0),
                               ha='left',
                               fontsize=8,
                               color=color,
                               fontweight='bold')
        
        # Formatting
        ax.set_title('Evolução Comparativa de Todas as Lesões', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Data do Exame', fontsize=12)
        ax.set_ylabel('Tamanho (cm)', fontsize=12)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add summary statistics
        self._add_combined_statistics(ax, filtered_data)
        
        plt.tight_layout()
        return fig
    
    def create_variation_chart(self, summary_data: pd.DataFrame) -> plt.Figure:
        """Create bar chart showing total variations for all lesions"""
        
        if summary_data.empty:
            return self._create_empty_chart("Sem dados de variação")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        lesions = summary_data['Lesão']
        variations = summary_data['Variação Total (%)']
        
        # Color bars based on positive/negative variation
        colors = ['#e74c3c' if v > 0 else '#2ecc71' if v < 0 else '#95a5a6' for v in variations]
        
        # Create horizontal bar chart
        bars = ax.barh(lesions, variations, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, variations)):
            width = bar.get_width()
            label_x = width + (1 if width >= 0 else -1)
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{value:+.1f}%', 
                   ha='left' if width >= 0 else 'right', 
                   va='center',
                   fontweight='bold',
                   fontsize=10)
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.8)
        
        # Formatting
        ax.set_title('Variação Total por Lesão (%)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Variação Percentual (%)', fontsize=12)
        ax.set_ylabel('Lesões', fontsize=12)
        
        # Grid
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        return fig
    
    def _highlight_significant_changes(self, ax, dates, sizes):
        """Highlight points with significant changes"""
        if len(sizes) < 2:
            return
        
        # Calculate percentage changes
        for i in range(1, len(sizes)):
            prev_size = sizes.iloc[i-1]
            curr_size = sizes.iloc[i]
            
            if prev_size > 0:
                change_pct = ((curr_size - prev_size) / prev_size) * 100
                
                # Highlight significant changes (>20%)
                if abs(change_pct) > 20:
                    color = 'red' if change_pct > 0 else 'green'
                    ax.scatter(dates.iloc[i], curr_size, s=150, 
                             facecolors='none', edgecolors=color, linewidth=3, alpha=0.8)
    
    def _add_trend_line(self, ax, dates, sizes):
        """Add trend line to the chart"""
        try:
            # Convert dates to numeric for trend calculation
            date_numeric = mdates.date2num(dates)
            
            # Calculate linear trend
            z = np.polyfit(date_numeric, sizes, 1)
            p = np.poly1d(z)
            
            # Plot trend line
            ax.plot(dates, p(date_numeric), "--", alpha=0.6, color='gray', linewidth=2, label='Tendência')
            
        except Exception:
            # If trend calculation fails, skip it
            pass
    
    def _add_statistics_box(self, ax, lesion_data: pd.DataFrame, lesion_name: str):
        """Add statistics box to individual chart"""
        if lesion_data.empty:
            return
        
        # Calculate statistics
        first_size = lesion_data.iloc[0]['tamanho_cm']
        last_size = lesion_data.iloc[-1]['tamanho_cm']
        max_size = lesion_data['tamanho_cm'].max()
        min_size = lesion_data['tamanho_cm'].min()
        
        variation_pct = ((last_size - first_size) / first_size) * 100 if first_size > 0 else 0
        
        # Create statistics text
        stats_text = f"""Estatísticas:
• Tamanho inicial: {first_size:.2f} cm
• Tamanho atual: {last_size:.2f} cm
• Variação total: {variation_pct:+.1f}%
• Máximo: {max_size:.2f} cm
• Mínimo: {min_size:.2f} cm
• Medições: {len(lesion_data)}"""
        
        # Add text box
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='white', alpha=0.9, edgecolor='gray'))
    
    def _add_combined_statistics(self, ax, data: pd.DataFrame):
        """Add summary statistics to combined chart"""
        total_lesions = data['lesao_id'].nunique()
        date_range = f"{data['data_exame'].min().strftime('%d/%m/%Y')} - {data['data_exame'].max().strftime('%d/%m/%Y')}"
        
        stats_text = f"""Resumo:
• Lesões: {total_lesions}
• Período: {date_range}
• Total medições: {len(data)}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='white', alpha=0.9, edgecolor='gray'))
    
    def _create_empty_chart(self, message: str) -> plt.Figure:
        """Create empty chart with message"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, transform=ax.transAxes, fontsize=14,
               ha='center', va='center', bbox=dict(boxstyle="round,pad=1", 
               facecolor='lightgray', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    def save_chart(self, fig: plt.Figure, filename: str, dpi: int = 300) -> str:
        """Save chart to file"""
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            return filename
        except Exception as e:
            raise Exception(f"Erro ao salvar gráfico: {str(e)}")
    
    def create_timeline_chart(self, detailed_data: pd.DataFrame):
        """Create timeline chart showing all measurements chronologically"""
        
        if detailed_data.empty:
            return self._create_empty_chart("Sem dados para timeline")
        
        # Sort by date
        detailed_data = detailed_data.sort_values('data_exame')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Add treatment periods as background first
        self._add_treatment_periods(ax)
        
        # Get unique lesions
        unique_lesions = detailed_data['lesao_id'].unique()
        
        # Create y-positions for each lesion
        y_positions = {lesion: i for i, lesion in enumerate(unique_lesions)}
        
        # Plot each measurement as a point
        for lesion in unique_lesions:
            lesion_data = detailed_data[detailed_data['lesao_id'] == lesion]
            dates = pd.to_datetime(lesion_data['data_exame'])
            sizes = lesion_data['tamanho_cm']
            
            # Use consistent color
            color = self.color_palette[y_positions[lesion] % len(self.color_palette)]
            
            # Plot points with higher zorder to appear over treatment periods
            ax.scatter(dates, [y_positions[lesion]] * len(dates), 
                      s=sizes * 100, alpha=0.8, color=color, label=lesion, zorder=5)
            
            # Add size labels
            for date, size in zip(dates, sizes):
                ax.annotate(f'{size:.1f}cm', 
                           (date, y_positions[lesion]), 
                           xytext=(5, 0), textcoords='offset points',
                           va='center', fontsize=8, zorder=6)
        
        # Formatting
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()))
        ax.set_xlabel('Data do Exame', fontsize=12)
        ax.set_ylabel('Lesões', fontsize=12)
        ax.set_title('Timeline de Todas as Medições com Períodos de Tratamento', fontsize=16, fontweight='bold')
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Grid
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
