# data_preprocessing.py
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, lower, regexp_replace
from transformers import DistilBertTokenizer
from Enum.model_type import ModelType
import re
import os

class DataPreprocessor:
    def __init__(self, tokenizer_name=ModelType.DISTILBERT.value):
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.setup_spark_environment()

    def setup_spark_environment(self):
        """Set up the HADOOP_HOME environment variable for Spark on Windows."""
        hadoop_home = r'C:\path\to\winutils'  # Adjust this path to where winutils.exe is located
        os.environ['HADOOP_HOME'] = hadoop_home

    def load_data(self, file_paths):
        """Load CSV files into a dictionary."""
        data = {}
        for data_type, path in file_paths.items():
            try:
                if path.endswith('.csv'):
                    data[data_type] = pd.read_csv(path)  # Load the CSV file using pandas
                else:
                    print(f"Unsupported file format for {path}")
            except FileNotFoundError as e:
                print(f"Error loading data: {e}")
        return data

    def clean_text(self, text):
        """Clean a given text by lowercasing, removing extra spaces, and special characters."""
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text

    def preprocess_text(self, df, question_col, answer_cols, is_spark=False):
        """Preprocess text data, either for Pandas or PySpark DataFrame."""
        if is_spark and isinstance(df, SparkDataFrame):
            return self.preprocess_text_spark(df, question_col, answer_cols)
        elif isinstance(df, pd.DataFrame):
            return self.preprocess_text_pandas(df, question_col, answer_cols)
        else:
            raise TypeError("Input DataFrame must be a Pandas DataFrame or a PySpark DataFrame")

    def preprocess_text_spark(self, df, question_col, answer_cols):
        """Preprocess text using PySpark DataFrame."""
        # Clean the question text
        df = df.withColumn(question_col, lower(col(question_col)))
        df = df.withColumn(question_col, regexp_replace(col(question_col), r"[^a-zA-Z0-9\s]", ""))

        # Clean answer columns
        for col_name in answer_cols:
            df = df.withColumn(col_name, lower(col(col_name)))
            df = df.withColumn(col_name, regexp_replace(col(col_name), r"[^a-zA-Z0-9\s]", ""))

        # Combine question and answers into a single cleaned text column
        df = df.withColumn('cleaned_text', col(question_col))
        for col_name in answer_cols:
            df = df.withColumn('cleaned_text', col('cleaned_text') + ' ' + col(col_name))

        return df

    def preprocess_text_pandas(self, df, question_col, answer_cols):
        """Preprocess text using Pandas DataFrame."""
        required_columns = [question_col] + answer_cols
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise KeyError(f"Missing columns in data: {missing_columns}")

        # Clean the question and answer columns
        cleaned_questions = df[question_col].apply(self.clean_text)
        cleaned_answers = df[answer_cols].apply(lambda row: ' '.join(row.fillna('')), axis=1).apply(self.clean_text)

        # Combine cleaned questions and answers
        df['cleaned_text'] = cleaned_questions + ' ' + cleaned_answers

        # Ensure that the 'CorrectAnswer' or 'labels' column is present, or raise a more descriptive error
        if 'CorrectAnswer' in df.columns:
            return df[['cleaned_text', 'CorrectAnswer']]
        elif 'labels' in df.columns:
            return df[['cleaned_text', 'labels']]
        else:
            raise KeyError("'CorrectAnswer' or 'labels' column not found in the dataset.")
