import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import re
from datetime import datetime
import io

def format_summary_table(summary_df: pd.DataFrame) -> str:
    """Convert summary DataFrame to Markdown format"""
    if summary_df.empty:
        return "| Lesão | Primeira Data | Tamanho Inicial (cm) | Última Data | Tamanho Final (cm) | Variação Total (%) | Status Atual |\n|-------|---------------|---------------------|-------------|-------------------|-------------------|-------------|\n| Nenhuma lesão encontrada | - | - | - | - | - | - |"
    
    # Create markdown table
    markdown_lines = []
    
    # Header
    headers = ["Lesão", "Primeira Data", "Tamanho Inicial (cm)", "Última Data", 
               "Tamanho Final (cm)", "Variação Total (%)", "Status Atual"]
    markdown_lines.append("| " + " | ".join(headers) + " |")
    markdown_lines.append("|" + "|".join(["-------"] * len(headers)) + "|")
    
    # Data rows
    for _, row in summary_df.iterrows():
        row_data = [
            str(row['Lesão']),
            str(row['Primeira Data']),
            f"{row['Tamanho Inicial (cm)']:.2f}",
            str(row['Última Data']),
            f"{row['Tamanho Final (cm)']:.2f}",
            f"{row['Variação Total (%)']:+.1f}%",
            str(row['Status Atual'])
        ]
        markdown_lines.append("| " + " | ".join(row_data) + " |")
    
    return "\n".join(markdown_lines)

def format_detailed_table(detailed_df: pd.DataFrame) -> str:
    """Convert detailed DataFrame to Markdown format"""
    if detailed_df.empty:
        return "| Lesão | Data do Exame | Tamanho (cm) | Variação (%) |\n|-------|---------------|--------------|-------------|\n| Nenhum dado encontrado | - | - | - |"
    
    # Create markdown table
    markdown_lines = []
    
    # Header
    headers = ["Lesão", "Data do Exame", "Tamanho (cm)", "Variação (%)"]
    markdown_lines.append("| " + " | ".join(headers) + " |")
    markdown_lines.append("|" + "|".join(["-------"] * len(headers)) + "|")
    
    # Data rows
    for _, row in detailed_df.iterrows():
        variation_str = f"{row['variacao_percentual']:+.1f}%" if pd.notna(row['variacao_percentual']) else "Primeira medição"
        
        row_data = [
            str(row['lesao_id']),
            row['data_exame'].strftime('%d/%m/%Y'),
            f"{row['tamanho_cm']:.2f}",
            variation_str
        ]
        markdown_lines.append("| " + " | ".join(row_data) + " |")
    
    return "\n".join(markdown_lines)

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """Validate DataFrame structure and content"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame está vazio")
        return validation_results
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Colunas obrigatórias faltando: {missing_columns}")
    
    # Check data types and content
    if 'lesao_id' in df.columns:
        null_lesions = df['lesao_id'].isnull().sum()
        if null_lesions > 0:
            validation_results['warnings'].append(f"{null_lesions} registros com lesao_id nulo")
    
    if 'tamanho_cm' in df.columns:
        # Check for negative or zero sizes
        invalid_sizes = df[df['tamanho_cm'] <= 0].shape[0]
        if invalid_sizes > 0:
            validation_results['warnings'].append(f"{invalid_sizes} registros com tamanho inválido (≤0)")
        
        # Check for extremely large sizes (>50cm)
        large_sizes = df[df['tamanho_cm'] > 50].shape[0]
        if large_sizes > 0:
            validation_results['warnings'].append(f"{large_sizes} registros com tamanho muito grande (>50cm)")
    
    if 'data_exame' in df.columns:
        # Check date format
        try:
            pd.to_datetime(df['data_exame'])
        except:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Formato de data inválido na coluna data_exame")
    
    # Summary statistics
    validation_results['summary'] = {
        'total_records': len(df),
        'unique_lesions': df['lesao_id'].nunique() if 'lesao_id' in df.columns else 0,
        'date_range': get_date_range_summary(df) if 'data_exame' in df.columns else None
    }
    
    return validation_results

def get_date_range_summary(df: pd.DataFrame) -> Dict[str, str]:
    """Get summary of date range in DataFrame"""
    if 'data_exame' not in df.columns or df.empty:
        return {'start': None, 'end': None, 'span_days': 0}
    
    try:
        dates = pd.to_datetime(df['data_exame'])
        start_date = dates.min()
        end_date = dates.max()
        span_days = (end_date - start_date).days
        
        return {
            'start': start_date.strftime('%d/%m/%Y'),
            'end': end_date.strftime('%d/%m/%Y'),
            'span_days': span_days
        }
    except:
        return {'start': None, 'end': None, 'span_days': 0}

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with parsing
    text = re.sub(r'[^\w\s\.,:/\-()]', '', text)
    
    return text

def standardize_units(size_str: str, unit_str: str) -> Optional[float]:
    """Standardize measurement units to centimeters"""
    try:
        # Clean size string
        size_str = size_str.replace(',', '.').strip()
        size_value = float(size_str)
        
        # Convert based on unit
        unit_lower = unit_str.lower().strip()
        
        if unit_lower in ['mm', 'milímetro', 'milímetros']:
            return round(size_value / 10, 2)
        elif unit_lower in ['cm', 'centímetro', 'centímetros']:
            return round(size_value, 2)
        elif unit_lower in ['m', 'metro', 'metros']:
            return round(size_value * 100, 2)
        else:
            # Default to cm if unit is unclear
            return round(size_value, 2)
    
    except (ValueError, TypeError):
        return None

def calculate_percentage_change(old_value: float, new_value: float) -> Optional[float]:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return None  # Cannot calculate percentage change from zero
    
    try:
        change = ((new_value - old_value) / old_value) * 100
        return round(change, 2)
    except (TypeError, ZeroDivisionError):
        return None

def detect_outliers(values: List[float], method: str = 'iqr') -> List[bool]:
    """Detect outliers in a list of values"""
    if not values or len(values) < 4:
        return [False] * len(values)
    
    try:
        if method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            return [v < lower_bound or v > upper_bound for v in values]
        
        elif method == 'zscore':
            # Z-score method
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return [False] * len(values)
            
            z_scores = [(v - mean_val) / std_val for v in values]
            return [abs(z) > 3 for z in z_scores]
        
        else:
            return [False] * len(values)
    
    except Exception:
        return [False] * len(values)

def generate_filename(base_name: str, extension: str, timestamp: bool = True) -> str:
    """Generate standardized filename"""
    # Clean base name
    base_name = re.sub(r'[^\w\-_\.]', '_', base_name)
    
    # Add timestamp if requested
    if timestamp:
        now = datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp_str}.{extension}"
    else:
        filename = f"{base_name}.{extension}"
    
    return filename

def create_csv_download(df: pd.DataFrame, filename: str = None) -> str:
    """Create CSV string for download"""
    if filename is None:
        filename = generate_filename("dados_lesoes", "csv")
    
    # Convert DataFrame to CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_string = csv_buffer.getvalue()
    
    return csv_string

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def parse_portuguese_date(date_str: str) -> Optional[datetime]:
    """Parse Portuguese date formats"""
    if not isinstance(date_str, str):
        return None
    
    # Common Portuguese date patterns
    patterns = [
        (r'(\d{1,2})\s*de\s*(\w+)\s*de\s*(\d{4})', '%d %B %Y'),
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', '%d/%m/%Y'),
        (r'(\d{1,2})-(\d{1,2})-(\d{4})', '%d-%m-%Y'),
        (r'(\d{4})/(\d{1,2})/(\d{1,2})', '%Y/%m/%d'),
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d')
    ]
    
    # Portuguese month names
    month_names = {
        'janeiro': '01', 'fevereiro': '02', 'março': '03', 'abril': '04',
        'maio': '05', 'junho': '06', 'julho': '07', 'agosto': '08',
        'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12'
    }
    
    # Try to parse with different patterns
    for pattern, date_format in patterns:
        match = re.search(pattern, date_str.lower())
        if match:
            try:
                if 'de' in pattern:
                    # Handle "day de month de year" format
                    day, month, year = match.groups()
                    month_num = month_names.get(month.lower())
                    if month_num:
                        date_obj = datetime.strptime(f"{day}/{month_num}/{year}", '%d/%m/%Y')
                        return date_obj
                else:
                    # Handle numeric formats
                    date_obj = datetime.strptime(match.group(), date_format)
                    return date_obj
            except ValueError:
                continue
    
    return None

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with default value for division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def extract_numeric_value(text: str) -> Optional[float]:
    """Extract first numeric value from text string"""
    if not isinstance(text, str):
        return None
    
    # Pattern to match decimal numbers (with comma or dot)
    pattern = r'(\d+[,\.]\d+|\d+)'
    match = re.search(pattern, text)
    
    if match:
        try:
            value_str = match.group(1).replace(',', '.')
            return float(value_str)
        except ValueError:
            return None
    
    return None

def create_backup_data(data: Dict[str, Any], backup_name: str = None) -> str:
    """Create backup of analysis data"""
    if backup_name is None:
        backup_name = generate_filename("backup_analise", "json")
    
    try:
        import json
        
        # Convert pandas objects to serializable format
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                serializable_data[key] = value.to_dict('records')
            elif isinstance(value, pd.Timestamp):
                serializable_data[key] = value.isoformat()
            else:
                serializable_data[key] = value
        
        # Add metadata
        serializable_data['backup_metadata'] = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        json_string = json.dumps(serializable_data, indent=2, ensure_ascii=False)
        return json_string
    
    except Exception as e:
        raise Exception(f"Erro ao criar backup: {str(e)}")
