import os
import pandas as pd
import json
import ast
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_engine():
    """Tạo và trả về một SQLAlchemy engine kết nối tới database."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logging.error("DATABASE_URL không được thiết lập.")
        raise ValueError("DATABASE_URL is not set")
    
    try:
        engine = create_engine(database_url)
        logging.info("Kết nối database thành công.")
        return engine
    except Exception as e:
        logging.error(f"Lỗi khi kết nối database: {e}")
        raise

def load_movies_from_database(engine):
    """Đọc toàn bộ dữ liệu phim từ database."""
    try:
        query = "SELECT id, title, overview, movie_cast, keywords FROM movies"
        df = pd.read_sql(query, engine)
        logging.info(f"Đã tải thành công {len(df)} phim từ database.")
        return df
    except Exception as e:
        logging.error(f"Lỗi khi tải dữ liệu từ database: {e}")
        raise

def clean_missing_values(df):
    """Xử lý giá trị thiếu trong DataFrame."""
    logging.info("Đang xử lý giá trị thiếu...")
    
    # Thay thế giá trị null bằng chuỗi rỗng
    df['overview'] = df['overview'].fillna('')
    df['movie_cast'] = df['movie_cast'].fillna('[]')
    df['keywords'] = df['keywords'].fillna('[]')
    
    # Log thông tin về missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logging.warning(f"Còn lại các giá trị null: {null_counts[null_counts > 0].to_dict()}")
    else:
        logging.info("Đã xử lý xong tất cả giá trị thiếu.")
    
    return df

def robust_parse(data):
    """Robust parsing that handles multiple data formats"""
    # Already a list
    if isinstance(data, list):
        return data
    
    # NaN or empty
    if pd.isna(data) or data == '' or data is None:
        return []
    
    # Convert to string first
    data_str = str(data).strip()
    
    # Empty string
    if not data_str:
        return []
    
    # Try JSON first (double quotes)
    try:
        result = json.loads(data_str)
        return result if isinstance(result, list) else []
    except:
        pass
    
    # Try ast.literal_eval (handles Python literals with single quotes)
    try:
        result = ast.literal_eval(data_str)
        return result if isinstance(result, list) else []
    except:
        pass
    
    # Try manual parsing for simple cases like "['item1', 'item2']"
    try:
        # Remove outer brackets and split by comma
        if data_str.startswith('[') and data_str.endswith(']'):
            inner = data_str[1:-1].strip()
            if not inner:
                return []
            
            # Split by comma and clean each item
            items = []
            for item in inner.split(','):
                item = item.strip().strip("'\"")
                if item:
                    items.append(item)
            return items
    except:
        pass
    
    return []

def parse_json_column(json_str):
    """Parse JSON string và trả về list, sử dụng robust parsing."""
    return robust_parse(json_str)

def extract_names_robust(data_list, max_items=None):
    """Extract names from various data formats"""
    if not isinstance(data_list, list):
        return []
    
    names = []
    items_to_process = data_list[:max_items] if max_items else data_list
    
    for item in items_to_process:
        name = None
        
        # Dictionary (common in JSON APIs)
        if isinstance(item, dict):
            # Try various common keys
            name = (item.get('name') or 
                   item.get('Name') or 
                   item.get('actor_name') or 
                   item.get('character') or
                   item.get('keyword'))
        
        # Already a string
        elif isinstance(item, str):
            name = item
        
        # Add clean name
        if name:
            clean_name = re.sub(r'[^a-zA-Z0-9]', '', str(name).strip())
            if clean_name:
                names.append(clean_name.lower())
    
    return names

def clean_and_join_names(names_list):
    """
    Làm sạch list tên (cast hoặc keywords) và nối thành một chuỗi.
    Sử dụng extract_names_robust để xử lý nhiều format khác nhau.
    """
    if not names_list or not isinstance(names_list, list):
        return ""
    
    # Sử dụng extract_names_robust để xử lý
    cleaned_names = extract_names_robust(names_list)
    return " ".join(cleaned_names)

def clean_overview(overview):
    """Làm sạch text overview với NLTK."""
    if pd.isna(overview) or overview == '':
        return ""
    
    # Convert to lowercase
    text = str(overview).lower()
    
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def standardize_data(df):
    """Chuẩn hóa dữ liệu cast và keywords với robust parsing."""
    logging.info("Đang chuẩn hóa dữ liệu cast và keywords...")
    
    # Parse JSON columns với robust parsing
    df['cast_parsed'] = df['movie_cast'].apply(parse_json_column)
    df['keywords_parsed'] = df['keywords'].apply(parse_json_column)
    
    # Clean và join names với enhanced extraction
    df['cast_clean'] = df['cast_parsed'].apply(lambda x: " ".join(extract_names_robust(x, 5)))
    df['keywords_clean'] = df['keywords_parsed'].apply(lambda x: " ".join(extract_names_robust(x, 10)))
    
    # Clean overview với NLTK
    df['overview_clean'] = df['overview'].apply(clean_overview)
    
    logging.info("Hoàn thành chuẩn hóa dữ liệu.")
    return df

def create_tags_feature(df):
    """Tạo cột tags bằng cách kết hợp overview, cast và keywords với overview-heavy strategy."""
    logging.info("Đang tạo feature 'tags' với overview-heavy strategy...")
    
    def combine_features(row):
        """Kết hợp các features thành một chuỗi tags với tỷ lệ 3:1:2."""
        parts = []
        
        # Overview với trọng số cao hơn (3x) - đã được chứng minh hiệu quả
        if row['overview_clean']:
            parts.extend([row['overview_clean']] * 3)
        
        # Cast (1x)
        if row['cast_clean']:
            parts.append(row['cast_clean'])
        
        # Keywords với trọng số trung bình (2x) để tăng specificity
        if row['keywords_clean']:
            parts.extend([row['keywords_clean']] * 2)
        
        # Kết hợp tất cả
        tags = " ".join(parts)
        
        # Loại bỏ khoảng trắng thừa
        tags = re.sub(r'\s+', ' ', tags).strip()
        
        return tags
    
    df['tags'] = df.apply(combine_features, axis=1)
    
    # Log thống kê
    empty_tags = df['tags'].str.len() == 0
    avg_length = df['tags'].str.len().mean()
    logging.info(f"Đã tạo tags cho {len(df)} phim. {empty_tags.sum()} phim có tags rỗng.")
    logging.info(f"Độ dài trung bình tags: {avg_length:.1f} ký tự")
    
    return df

def save_processed_data(df, output_path):
    """Lưu dữ liệu đã xử lý vào file parquet với metadata bổ sung."""
    try:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Tạo DataFrame cuối cùng với các cột cần thiết và metadata bổ sung
        processed_df = df[['id', 'title', 'tags']].copy()
        processed_df = processed_df.rename(columns={'id': 'movie_id'})
        
        # Thêm metadata để theo dõi chất lượng
        processed_df['tags_length'] = processed_df['tags'].str.len()
        processed_df['has_content'] = processed_df['tags_length'] > 0
        
        # Try to save as parquet, fallback to CSV if pyarrow not available
        try:
            processed_df.to_parquet(output_path, index=False)
            logging.info(f"Đã lưu {len(processed_df)} phim đã xử lý vào: {output_path}")
        except Exception as parquet_error:
            # Fallback to CSV
            csv_path = output_path.replace('.parquet', '.csv')
            processed_df.to_csv(csv_path, index=False)
            logging.warning(f"Không thể lưu Parquet (lỗi: {parquet_error})")
            logging.info(f"Đã lưu {len(processed_df)} phim đã xử lý vào CSV: {csv_path}")
        
        # Log thống kê chi tiết
        avg_tags_length = processed_df['tags_length'].mean()
        content_coverage = processed_df['has_content'].mean() * 100
        logging.info(f"Độ dài trung bình của tags: {avg_tags_length:.1f} ký tự")
        logging.info(f"Coverage (phim có nội dung): {content_coverage:.1f}%")
        
        return True
    except Exception as e:
        logging.error(f"Lỗi khi lưu dữ liệu: {e}")
        return False

def generate_preprocessing_report(df, output_path):
    """Tạo báo cáo chi tiết về quá trình preprocessing."""
    try:
        report_path = output_path.replace('.parquet', '_report.txt')
        
        # Tính toán các thống kê
        total_movies = len(df)
        has_overview = (df['overview_clean'].str.len() > 0).sum()
        has_cast = (df['cast_clean'].str.len() > 0).sum()
        has_keywords = (df['keywords_clean'].str.len() > 0).sum()
        has_tags = (df['tags'].str.len() > 0).sum()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== MOVIE DATA PREPROCESSING REPORT ===\n")
            f.write(f"Thời gian xử lý: {pd.Timestamp.now()}\n")
            f.write(f"Tổng số phim: {total_movies}\n\n")
            
            f.write("=== COVERAGE ANALYSIS ===\n")
            f.write(f"Phim có overview: {has_overview}/{total_movies} ({has_overview/total_movies*100:.1f}%)\n")
            f.write(f"Phim có cast info: {has_cast}/{total_movies} ({has_cast/total_movies*100:.1f}%)\n")
            f.write(f"Phim có keywords: {has_keywords}/{total_movies} ({has_keywords/total_movies*100:.1f}%)\n")
            f.write(f"Phim có tags: {has_tags}/{total_movies} ({has_tags/total_movies*100:.1f}%)\n\n")
            
            f.write("=== THỐNG KÊ TAGS ===\n")
            f.write(f"Độ dài trung bình tags: {df['tags'].str.len().mean():.1f} ký tự\n")
            f.write(f"Độ dài tối đa tags: {df['tags'].str.len().max()} ký tự\n")
            f.write(f"Độ dài tối thiểu tags: {df['tags'].str.len().min()} ký tự\n")
            f.write(f"Số phim có tags rỗng: {(df['tags'].str.len() == 0).sum()}\n\n")
            
            f.write("=== FEATURE ENGINEERING STRATEGY ===\n")
            f.write("Sử dụng overview-heavy strategy với tỷ lệ 3:1:2\n")
            f.write("- Overview: 3x trọng số (context chính)\n")
            f.write("- Cast: 1x trọng số (nhân vật chính)\n")
            f.write("- Keywords: 2x trọng số (specificity)\n\n")
            
            f.write("=== MẪU TAGS ===\n")
            for i, row in df.head(3).iterrows():
                f.write(f"Phim: {row['title']}\n")
                f.write(f"Tags: {row['tags'][:300]}...\n\n")
        
        logging.info(f"Đã tạo báo cáo preprocessing: {report_path}")
    except Exception as e:
        logging.warning(f"Không thể tạo báo cáo: {e}")

def main():
    """Hàm chính để thực hiện preprocessing."""
    try:
        logging.info("Bắt đầu quá trình preprocessing dữ liệu phim...")
        
        # Bước 1: Kết nối database và tải dữ liệu
        engine = get_db_engine()
        df = load_movies_from_database(engine)
        
        if df.empty:
            logging.warning("Không có dữ liệu để xử lý.")
            return
        
        # Bước 2: Làm sạch giá trị thiếu
        df = clean_missing_values(df)
        
        # Bước 3: Chuẩn hóa dữ liệu
        df = standardize_data(df)
        
        # Bước 4: Tạo feature tags
        df = create_tags_feature(df)
        
        # Bước 5: Lưu dữ liệu đã xử lý
        output_path = "data/processed/processed_movies.parquet"
        if save_processed_data(df, output_path):
            # Tạo báo cáo
            generate_preprocessing_report(df, output_path)
            logging.info("Hoàn thành preprocessing dữ liệu phim.")
        else:
            raise Exception("Không thể lưu dữ liệu đã xử lý.")
            
    except Exception as e:
        logging.error(f"Lỗi trong quá trình preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
