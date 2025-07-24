import os
import pandas as pd
import json
import random as randum
from tqdm import tqdm
from service import transfer_name
from dtype_mapping import (
    INTEGER_DTYPE_MAPPING,
    FLOAT_DTYPE_MAPPING,
    OTHER_DTYPE_MAPPING,
    SPECIAL_INTEGER_DTYPE_MAPPING
)
import warnings
from common_utils import transfer_name, SCHEMA_DIR, sql_alchemy_helper


def infer_and_convert(series):
    # 尝试转换为整数
    # Try converting to integer
    try:
        return pd.to_numeric(series, downcast='integer')
    except ValueError:
        pass

    # 尝试转换为浮点数
    # Try converting to float
    try:
        return pd.to_numeric(series, downcast='float')
    except ValueError:
        pass

    # 尝试转换为日期时间
    # Try converting to datetime
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # 忽略特定类型的警告 | Ignore specific warnings
            return pd.to_datetime(series)
    except ValueError:
        pass

    # 如果都不行，返回原始数据
    # If all conversions fail, return the original series
    return series


def pandas_to_mysql_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        if str(dtype) in SPECIAL_INTEGER_DTYPE_MAPPING:
            return SPECIAL_INTEGER_DTYPE_MAPPING[str(dtype)]
        return INTEGER_DTYPE_MAPPING.get(dtype, 'INT')
    #Int is default for map retrieval

    elif pd.api.types.is_float_dtype(dtype):
        return FLOAT_DTYPE_MAPPING.get(dtype, 'FLOAT')

    elif pd.api.types.is_bool_dtype(dtype):
        return OTHER_DTYPE_MAPPING['boolean']

    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return OTHER_DTYPE_MAPPING['datetime']

    elif pd.api.types.is_timedelta64_dtype(dtype):
        return OTHER_DTYPE_MAPPING['timedelta']

    elif pd.api.types.is_string_dtype(dtype):
        return OTHER_DTYPE_MAPPING['string']

    elif pd.api.types.is_categorical_dtype(dtype):
        return OTHER_DTYPE_MAPPING['category']

    else:
        return OTHER_DTYPE_MAPPING['default']

def get_sample_values(series):
    valid_values = [str(x) for x in series.dropna().unique() if pd.notnull(x) and len(str(x)) < 64]
    sample_values = randum.sample(valid_values, min(3, len(valid_values)))
    return sample_values if sample_values else ['no sample values available']

def get_schema_and_data(df):
    column_list = []
    for col in df.columns:
        cur_column_list = []
        if isinstance(df[col], pd.DataFrame):
            print(f"Column {col} is a DataFrame, skipping...")
            raise ValueError(f"Column {col} is a DataFrame, which is not supported.")   
        cur_column_list.append(col)
        cur_column_list.append(pandas_to_mysql_dtype(df[col].dtype))
        cur_column_list.append('sample values:' + str(get_sample_values(df[col])))

        # 形成三元组
        # Form a triplet for each column
        column_list.append(cur_column_list)

    return column_list

def generate_schema_info(df: pd.DataFrame, file_name: str):
    try:
        column_list = get_schema_and_data(df)
    except:
        print(f"{file_name} 列存在问题")  # Column has problems
        raise ValueError(f"Error processing file: {file_name}")

    table_name = transfer_name(file_name)

    schema_dict = {
        'table_name': table_name,
        'column_list': column_list           
    }

    return schema_dict, table_name


def transfer_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 DataFrame 的列名：
    Clean the column names of the DataFrame:
    1. 使用 transfer_name_func 转换列名。
       Use transfer_name_func to convert column names.
    2. 如果第一个列名为空或 NaN，设置为 'No'。
       If the first column name is empty or NaN, set it to 'No'.
    3. 处理重复列名，重复列名加后缀 _1, _2 等。
       Handle duplicate column names by adding suffixes _1, _2, etc.

    参数：
        df: 需要处理列名的 DataFrame。
        df: DataFrame whose columns need to be processed.

    返回：
        列名处理后的 DataFrame。
        DataFrame with processed column names.
    """
    df = df.copy()

    # 第一步：统一转换列名
    # Step 1: Standardize column names
    df.columns = [transfer_name(col) for col in df.columns]

    # 第二步：首列为空或 NaN 时命名为 'No'
    # Step 2: Rename first column to 'No' if it is empty or NaN
    df.columns = [
        'No' if i == 0 and (not col or pd.isna(col)) else col
        for i, col in enumerate(df.columns)
    ]

    # 第三步：处理重复列名
    # Step 3: Handle duplicate column names
    seen = {}
    new_columns = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    df.columns = new_columns

    return df


def parse_excel_file_and_insert_to_db(excel_file_outer_dir: str):
    if not os.path.exists(excel_file_outer_dir):
        raise FileNotFoundError(f"File not found: {excel_file_outer_dir}")
    
    for file_name in tqdm(os.listdir(excel_file_outer_dir)):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            full_path = os.path.join(excel_file_outer_dir, file_name)
            df = pd.read_excel(full_path)
            
            df_convert = df.apply(infer_and_convert)
            df_convert = transfer_df_columns(df_convert)

            schema_dict, table_name = generate_schema_info(df_convert, file_name)

            # 确保目录存在
            # Ensure the directory exists
            if not os.path.exists(SCHEMA_DIR):
                os.makedirs(SCHEMA_DIR)

            with open(f"{SCHEMA_DIR}/{table_name}.json", 'w', encoding='utf-8') as f:
                json.dump(schema_dict, f, ensure_ascii=False)
            
            sql_alchemy_helper.insert_dataframe_batch(df_convert, table_name)


if __name__ == "__main__":
    parse_excel_file_and_insert_to_db('../dataset/hybridqa/dev_excel/')
