from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional, Union
import boto3
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
import openai
import json
from datetime import datetime, timedelta
import asyncio
import os
from dotenv import load_dotenv
import io
from urllib.parse import urlparse
import re
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from neuralprophet import NeuralProphet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts import TimeSeries
from darts.models import TransformerModel
import torch
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi.responses import JSONResponse
load_dotenv()
from fastapi import Request
app = FastAPI(title="Universal Supply Chain AI Backend")
import uuid
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class S3DataRequest(BaseModel):
    s3_url: str
    data_type: str

class ForecastRequest(BaseModel):
    data: List[Dict]
    periods: int = 30


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class OptimizationRequest(BaseModel):
    inventory_data: List[Dict[str, Any]]
    demand_data: List[Dict[str, Any]]
    constraints: Optional[Dict[str, Any]] = None

class DataProcessingRequest(BaseModel):
    data: List[Dict[str, Any]]
    processing_type: str = "auto"

# AWS services initalize
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
)
bedrock_client = boto3.client(
            service_name="bedrock",
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)
# models that can be used
AVAILABLE_MODELS = {
    'claude-3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
    'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
    'claude-3-opus': 'anthropic.claude-3-opus-20240229-v1:0',
    'titan-text': 'amazon.titan-text-express-v1',
    'llama2-70b': 'meta.llama2-70b-chat-v1'
}

agents_store: Dict[str, dict] = {}
executions_store: Dict[str, List[dict]] = {}
# Initialize OpenAI client (when you uncomment the OpenAI calls)
# openai.api_key = os.getenv('OPENAI_API_KEY')

class UniversalDataAnalyzer:
    """Universal analyzer that works with any dataset structure"""
    
    @staticmethod
    def analyze_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze any dataset structure and extract meaningful information"""
        try:
            if df.empty or len(df.columns) == 0:
                return {"type": "empty", "error": "Dataset is empty"}
            
            analysis = {
                "type": "universal",
                "structure": UniversalDataAnalyzer._analyze_column_structure(df),
                "patterns": UniversalDataAnalyzer._detect_data_patterns(df),
                "categories": UniversalDataAnalyzer._categorize_columns(df),
                "relationships": UniversalDataAnalyzer._analyze_relationships(df)
            }
            
            return analysis
        except Exception as e:
            print(f"Error in analyze_dataset_structure: {e}")
            return {"type": "error", "error": str(e)}
    
    @staticmethod
    def _analyze_column_structure(df: pd.DataFrame) -> Dict[str, Any]:
        structure = {
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "column_types": {},
            "column_info": {}
        }
        
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist()
            }
            if df[col].dtype in ['object', 'string']:
                col_info["semantic_type"] = UniversalDataAnalyzer._classify_text_column(df[col])
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_info["semantic_type"] = UniversalDataAnalyzer._classify_numeric_column(df[col])
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info["semantic_type"] = "datetime"
            else:
                col_info["semantic_type"] = "unknown"
            structure["column_info"][col] = col_info
            structure["column_types"][col_info["semantic_type"]] = structure["column_types"].get(col_info["semantic_type"], 0) + 1  
        return structure
    
    @staticmethod
    def _classify_text_column(series: pd.Series) -> str:
        sample_values = series.dropna().astype(str).head(100).str.lower()
        if any(keyword in ' '.join(sample_values) for keyword in ['email', '@', '.com', '.org']):
            return "email"
        elif any(keyword in ' '.join(sample_values) for keyword in ['phone', 'tel', '+', '(', ')']):
            return "phone"
        elif any(keyword in ' '.join(sample_values) for keyword in ['address', 'street', 'city', 'state']):
            return "address"
        elif any(keyword in ' '.join(sample_values) for keyword in ['name', 'first', 'last']):
            return "name"
        elif series.nunique() == len(series):
            return "identifier"
        elif series.nunique() < len(series) * 0.5:
            return "category"
        else:
            return "text"
    
    @staticmethod
    def _classify_numeric_column(series: pd.Series) -> str:
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return "numeric"
        if all(x.is_integer() for x in non_null_series if isinstance(x, (int, float))):
            if non_null_series.min() >= 0 and non_null_series.max() <= 100:
                return "percentage"
            elif non_null_series.min() >= 0:
                return "count"
            else:
                return "integer"
        if non_null_series.min() >= 0 and non_null_series.mean() > 1:
            return "currency"
        if non_null_series.min() >= 0 and non_null_series.max() <= 1:
            return "rate"
        return "numeric"
    
    @staticmethod
    def _detect_data_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        patterns = {
            "has_time_series": False,
            "has_hierarchical": False,
            "has_geographical": False,
            "has_transactional": False,
            "dominant_pattern": "tabular"
        }
        time_indicators = ['date', 'time', 'timestamp', 'year', 'month', 'day']
        geo_indicators = ['country', 'state', 'city', 'region', 'location', 'address', 'lat', 'lon', 'zip']
        columns_lower = [col.lower() for col in df.columns]
        trans_indicators = ['order', 'transaction', 'purchase', 'sale', 'payment', 'invoice']
        hier_indicators = ['parent', 'child', 'level', 'category', 'subcategory', 'department']
        if any(indicator in ' '.join(columns_lower) for indicator in time_indicators):
            patterns["has_time_series"] = True
        
        if any(indicator in ' '.join(columns_lower) for indicator in geo_indicators):
            patterns["has_geographical"] = True
        
        if any(indicator in ' '.join(columns_lower) for indicator in trans_indicators):
            patterns["has_transactional"] = True
        
        if any(indicator in ' '.join(columns_lower) for indicator in hier_indicators):
            patterns["has_hierarchical"] = True
        
        return patterns
    
    @staticmethod
    def _categorize_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize columns into functional groups"""
        categories = {
            "identifiers": [],
            "measurements": [],
            "categories": [],
            "dates": [],
            "text": [],
            "calculated": []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(keyword in col_lower for keyword in ['id', 'key', 'code', 'number']):
                categories["identifiers"].append(col)
            elif any(keyword in col_lower for keyword in ['date', 'time', 'timestamp']) or pd.api.types.is_datetime64_any_dtype(df[col]):
                categories["dates"].append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                if any(keyword in col_lower for keyword in ['amount', 'price', 'cost', 'value', 'quantity', 'count', 'total']):
                    categories["measurements"].append(col)
                elif any(keyword in col_lower for keyword in ['rate', 'percentage', 'ratio', 'score']):
                    categories["calculated"].append(col)
                else:
                    categories["measurements"].append(col)
            elif df[col].nunique() < len(df) * 0.5:
                categories["categories"].append(col)
            else:
                categories["text"].append(col)
        
        return categories
    
    @staticmethod
    def _analyze_relationships(df: pd.DataFrame) -> Dict[str, Any]:
        relationships = {
            "correlations": {},
            "dependencies": {},
            "hierarchies": []
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7: 
                        relationships["correlations"][f"{col1}_vs_{col2}"] = float(corr_val)
        return relationships
    
    @staticmethod
    def extract_universal_metrics(df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            metrics = {
                "basic_stats": UniversalDataAnalyzer._calculate_basic_stats(df),
                "data_quality": UniversalDataAnalyzer._calculate_data_quality(df),
                "distribution_stats": UniversalDataAnalyzer._calculate_distribution_stats(df),
                "column_insights": UniversalDataAnalyzer._calculate_column_insights(df, analysis)
            }  
            return metrics
        except Exception as e:
            print(f"Error in extract_universal_metrics: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _calculate_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns)
        }
    
    @staticmethod
    def _calculate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        
        return {
            "completeness_percentage": float(((total_cells - missing_cells) / total_cells) * 100) if total_cells > 0 else 0,
            "missing_values": int(missing_cells),
            "duplicate_rows": int(df.duplicated().sum()),
            "unique_rows": int(len(df) - df.duplicated().sum()),
            "columns_with_missing": int((df.isnull().sum() > 0).sum())
        }
    
    @staticmethod
    def _calculate_distribution_stats(df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "No numeric columns found"}
        
        stats = {}
        for col in numeric_cols[:5]:  
            try:
                col_stats = df[col].describe()
                stats[col] = {
                    "mean": float(col_stats['mean']),
                    "std": float(col_stats['std']),
                    "min": float(col_stats['min']),
                    "max": float(col_stats['max']),
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis())
                }
            except Exception:
                continue
        
        return stats
    
    @staticmethod
    def _calculate_column_insights(df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        insights = {}
        try:
            categories = analysis.get("categories", {})
            if "measurements" in categories:
                measurement_cols = categories["measurements"]
                if measurement_cols:
                    insights["measurements"] = {
                        "total_columns": len(measurement_cols),
                        "avg_values": {col: float(df[col].mean()) for col in measurement_cols[:3] if pd.api.types.is_numeric_dtype(df[col])},
                        "value_ranges": {col: {"min": float(df[col].min()), "max": float(df[col].max())} for col in measurement_cols[:3] if pd.api.types.is_numeric_dtype(df[col])}
                    }
            
            if "categories" in categories:
                category_cols = categories["categories"]
                if category_cols:
                    insights["categories"] = {
                        "total_columns": len(category_cols),
                        "unique_values": {col: int(df[col].nunique()) for col in category_cols[:3]},
                        "top_values": {col: df[col].value_counts().head(3).to_dict() for col in category_cols[:3]}
                    }
        
            if "identifiers" in categories:
                id_cols = categories["identifiers"]
                if id_cols:
                    insights["identifiers"] = {
                        "total_columns": len(id_cols),
                        "uniqueness": {col: float(df[col].nunique() / len(df)) * 100 for col in id_cols[:3]}
                    }
            
        except Exception as e:
            insights["error"] = str(e)
        
        return insights

class UniversalForecastEngine:
    @staticmethod
    def prepare_forecast_data(data: List[Dict[str, Any]], analysis: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        date_cols = UniversalForecastEngine._find_date_columns(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if date_cols and numeric_cols:
            date_col = date_cols[0]
            value_col = numeric_cols[0] 
            forecast_df = pd.DataFrame()
            try:
                forecast_df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
                forecast_df['y'] = pd.to_numeric(df[value_col], errors='coerce')
                result = forecast_df.dropna()
                
                if len(result) > 0:
                    return result
            except Exception:
                pass
        return UniversalForecastEngine._create_synthetic_time_series(df, numeric_cols)
    
    @staticmethod
    def _find_date_columns(df: pd.DataFrame) -> List[str]:
        date_cols = []
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        date_cols.extend(datetime_cols)
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'created', 'updated']):
                try:
                    pd.to_datetime(df[col].head(10), errors='raise')
                    if col not in date_cols:
                        date_cols.append(col)
                except:
                    continue
        
        return date_cols
    
    @staticmethod
    def _create_synthetic_time_series(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        if not numeric_cols:
            dates = pd.date_range(start='2023-01-01', periods=min(len(df), 365), freq='D')
            values = np.random.normal(100, 20, len(dates))
            return pd.DataFrame({'ds': dates, 'y': values})
        value_col = numeric_cols[0]
        dates = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        values = pd.to_numeric(df[value_col], errors='coerce').fillna(df[value_col].mean())
        
        return pd.DataFrame({'ds': dates, 'y': values})

@app.post("/api/data/s3")
async def fetch_s3_data(request: S3DataRequest):
    try:
        print(f"Received request: {request}")
        
        s3_url = str(request.s3_url)
        
        if s3_url.startswith("s3://"):
            s3_parts = s3_url.replace("s3://", "").split("/", 1)
            bucket = s3_parts[0]
            key = s3_parts[1] if len(s3_parts) > 1 else ""
        else:
            url_parts = s3_url.replace("https://", "").split("/")
            bucket = url_parts[0].replace(".s3.amazonaws.com", "")
            key = "/".join(url_parts[1:])
        
        print(f"Parsed - Bucket: {bucket}, Key: {key}")
        
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = pd.read_csv(obj['Body'])
        data = data.replace({np.nan: None})
       
        analysis = UniversalDataAnalyzer.analyze_dataset_structure(data)
        metrics = UniversalDataAnalyzer.extract_universal_metrics(data, analysis)
        
        return {
            "status": "success",
            "data": data.to_dict(orient="records"),
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "analysis": analysis,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

       

@app.post("/api/data/process")
async def process_data(request: DataProcessingRequest):
    try:
        if not request.data or len(request.data) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        df = pd.DataFrame(request.data)
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset")
    
        analysis = UniversalDataAnalyzer.analyze_dataset_structure(df)
        metrics = UniversalDataAnalyzer.extract_universal_metrics(df, analysis)
        df_cleaned = UniversalDataAnalyzer._clean_data_universally(df, analysis)
        original_missing = pd.DataFrame(request.data).isnull().sum().sum()
        cleaned_missing = df_cleaned.isnull().sum().sum()
        
        data_quality = {
            "total_records": len(df_cleaned),
            "missing_values_filled": max(0, original_missing - cleaned_missing),
            "completeness": ((df_cleaned.size - cleaned_missing) / df_cleaned.size) * 100 if df_cleaned.size > 0 else 0,
            "cleaning_applied": True
        }
        
        return {
            "status": "success",
            "processed_data": df_cleaned.head(1000).to_dict(orient="records"),
            "analysis": analysis,
            "metrics": metrics,
            "data_quality": data_quality
        }
        
    except Exception as e:
        print(f"Error in process_data: {e}")
        raise HTTPException(status_code=400, detail=f"Data processing failed: {str(e)}")
# @app.post("/api/forecast/timeseries")
# async def generate_forecast(request: ForecastRequest):
#     try:
#         df = pd.DataFrame(request.data)
        
#         df['Date'] = pd.to_datetime(df['Date'])
#         df['Sales Quantity'] = pd.to_numeric(df['Sales Quantity'], errors='coerce').fillna(0).astype('int64')

#         ts = df.groupby('Date')['Sales Quantity'].sum().reset_index()
#         ts = ts.rename(columns={'Date': 'ds', 'Sales Quantity': 'y'})

#         if len(ts) < 10:
#             raise HTTPException(status_code=400, detail="Need at least 10 data points to forecast.")

#         # Fit model
#         model = Prophet()
#         model.fit(ts)

#         # Future dataframe & forecast
#         future = model.make_future_dataframe(periods=request.periods)
#         forecast = model.predict(future)

#         # Prepare results for future periods only
#         forecast_result = forecast[['ds', 'yhat']].tail(request.periods).to_dict(orient="records")
#         print(forecast_result)
#         # Calculate metrics on historical data where predictions exist
#         merged = pd.merge(ts, forecast[['ds', 'yhat']], on='ds', how='inner')
        
#         if not merged.empty:
#             mae = mean_absolute_error(merged['y'], merged['yhat'])
#             mape = (abs((merged['y'] - merged['yhat']) / merged['y'])).mean() * 100
#             mse = mean_squared_error(merged['y'], merged['yhat'])
#             rmse = np.sqrt(mse)
#         else:
#             mae = mape = rmse = None

#         return {
#             "status": "success",
#             "forecast": forecast_result,
#             "metrics": {
#                 "mae": mae,
#                 "mape": mape,
#                 "rmse": rmse
#             }
#         }

#     except Exception as e:
#         print("Error in generate_forecast:", str(e))
#         raise HTTPException(status_code=400, detail=f"Forecast failed: {str(e)}")



@app.post("/api/forecast/timeseries")
async def generate_forecast(request: ForecastRequest):
    try:
        df = pd.DataFrame(request.data)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Sales Quantity'] = pd.to_numeric(df['Sales Quantity'], errors='coerce').fillna(0).astype('float')

        ts = df.groupby('Date')['Sales Quantity'].sum().reset_index()
        ts = ts.rename(columns={'Date': 'ds', 'Sales Quantity': 'y'})
        ts = ts.sort_values('ds')  
        full_range = pd.date_range(start=ts['ds'].min(), end=ts['ds'].max(), freq='D')
        ts = ts.set_index('ds').reindex(full_range).fillna(method='ffill').reset_index()    
        ts.columns = ['ds', 'y']
        if len(ts) < 20:
            raise HTTPException(status_code=400, detail="Need at least 20 data points.")

        results = {}
        print("hi reached model 1")
     
        model_prophet = Prophet()
        model_prophet.fit(ts)
        future = model_prophet.make_future_dataframe(periods=request.periods)
        forecast_prophet = model_prophet.predict(future)[['ds', 'yhat']].tail(request.periods)
        merged_p = pd.merge(ts, model_prophet.predict(ts)[['ds', 'yhat']], on='ds')
        results['Prophet'] = {
            "forecast": forecast_prophet.to_dict(orient='records'),
            "metrics": {
                "mae": mean_absolute_error(merged_p['y'], merged_p['yhat']),
                "mape": np.mean(np.abs((merged_p['y'] - merged_p['yhat']) / merged_p['y'])) * 100,
                "rmse": np.sqrt(mean_squared_error(merged_p['y'], merged_p['yhat']))
            }
        }
        print("hi2")
      
        arima_series = ts.set_index('ds')['y']
        arima_model = ARIMA(arima_series, order=(2, 1, 2)).fit()
        arima_forecast = arima_model.forecast(steps=request.periods)
        arima_result = pd.DataFrame({
            'ds': pd.date_range(start=ts['ds'].max() + pd.Timedelta(days=1), periods=request.periods),
            'yhat': arima_forecast
        })
        results['ARIMA'] = {

            "forecast": arima_result.to_dict(orient='records'),  
            "metrics": {
                "mae": arima_model.aic,
                "rmse": arima_model.mse 
            }       
        }
        print(arima_model.aic," " , arima_model.bic,  
        "hqic",arima_model.hqic, 
        "mse",arima_model.mse,  
        )

        print("hi3")
#         llm = BedrockLLM(model_id="anthropic.claude-v2", region_name="us-east-1")
#         data_string = "\n".join(
#     f"{row['ds'].date()}: {row['y']:.2f}"  # Format date and float nicely
#     for _, row in ts.iterrows()
# )

#         # Create a prompt template with a placeholder for forecast_data
#         prompt = PromptTemplate.from_template(
#             "Summarize the trend in this forecast data:\n\n{forecast_data}"
#         )
#         response = llm.predict(prompt.format(forecast_data=data_string))

#         print(response)
       
        # series = TimeSeries.from_dataframe(ts, time_col='ds', value_cols='y')
        # model_trans = TransformerModel(
        #     input_chunk_length=14,
        #     output_chunk_length=request.periods,
        #     batch_size=16,
        #     n_epochs=5,
        #     model_name="transformer",
        #     log_tensorboard=False,
        #     random_state=42,
        #     force_reset=True,
        # )
        # model_trans.fit(series)
        # forecast_trans = model_trans.predict(n=request.periods)
        # trans_result = forecast_trans.pd_dataframe().reset_index()
        # print(trans_result)
        # trans_result.columns = ['ds', 'yhat']

        # results['Transformer'] = {
        #     "forecast": trans_result.to_dict(orient='records'),
           
        # }

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.post("/api/chat/query")
async def chat_with_ai(request: ChatRequest):
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
       
        forecast_data = None
        data_analysis = None
        
        if request.context:
            forecast_data = request.context.get('forecast_data')
            data_analysis = request.context.get('data_analysis')
    
        context_prompt = ""
        if forecast_data:
            forecast_summary = format_forecast_data_for_claude(forecast_data)
            context_prompt += f"Forecast Analysis:\n{forecast_summary}\n\n"
        
        if data_analysis:
            analysis_summary = format_data_analysis_for_claude(data_analysis)
            context_prompt += f"Data Analysis:\n{analysis_summary}\n\n"
        system_prompt = """You are an expert supply chain and data analyst. You help users understand their data, forecasts, and provide actionable business insights. 
        
        Focus on:
        - Clear, actionable recommendations
        - Business impact assessment
        - Risk analysis and mitigation strategies
        - Practical implementation steps
        - Data-driven decision making
        
        Keep responses concise but comprehensive."""
        
        user_prompt = f"""{context_prompt}
        
        User Question: {request.message}
        
        Please provide a detailed analysis and actionable recommendations based on the data and forecast information provided."""
        claude_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }
    
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(claude_request),
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read())
        claude_response = response_body['content'][0]['text']
        
        insights = generate_forecast_insights(forecast_data) if forecast_data else []
        
        return {
            "status": "success",
            "response": claude_response,
            "insights": insights,
            "context_used": bool(request.context),
            "forecast_data_available": bool(forecast_data)
        }
        
    except Exception as e:
        print(f"Error in chat_with_ai: {e}")
        return await fallback_chat_response(request)

def format_forecast_data_for_claude(forecast_data):
    if not forecast_data:
        return "No forecast data available."
    
    summary = []

    for model_name, model_data in forecast_data.items():
        if isinstance(model_data, dict) and 'forecast' in model_data:
            forecast_values = model_data['forecast']
            metrics = model_data.get('metrics', {})
            
            summary.append(f"{model_name} Model:")
            summary.append(f"  - Forecast points: {len(forecast_values)}")
            
            if forecast_values:
                values = [item.get('yhat', 0) for item in forecast_values]
                summary.append(f"  - Forecast range: {min(values):.2f} to {max(values):.2f}")
                summary.append(f"  - Average predicted value: {np.mean(values):.2f}")
            
            if metrics:
                summary.append(f"  - Model accuracy metrics: {metrics}")
            
            summary.append("")
    
    return "\n".join(summary)

def format_data_analysis_for_claude(data_analysis):
    if not data_analysis:
        return "No data analysis available."
    
    summary = []
    
    if 'structure' in data_analysis:
        structure = data_analysis['structure']
        summary.append(f"Dataset Structure:")
        summary.append(f"  - Total columns: {structure.get('total_columns', 'N/A')}")
        summary.append(f"  - Total rows: {structure.get('total_rows', 'N/A')}")
        summary.append(f"  - Column types: {structure.get('column_types', {})}")
        summary.append("")
    if 'patterns' in data_analysis:
        patterns = data_analysis['patterns']
        summary.append(f"Data Patterns:")
        for pattern, exists in patterns.items():
            if exists:
                summary.append(f"  - {pattern.replace('_', ' ').title()}: Yes")
        summary.append("")

    if 'categories' in data_analysis:
        categories = data_analysis['categories']
        summary.append(f"Column Categories:")
        for category, columns in categories.items():
            if columns:
                summary.append(f"  - {category.title()}: {len(columns)} columns")
        summary.append("")
    
    return "\n".join(summary)

def generate_forecast_insights(forecast_data):
    if not forecast_data:
        return []
    
    insights = []
    
    try:
        for model_name, model_data in forecast_data.items():
            if isinstance(model_data, dict) and 'forecast' in model_data:
                forecast_values = model_data['forecast']
                
                if forecast_values and len(forecast_values) > 1:
                    values = [item.get('yhat', 0) for item in forecast_values]
                    if values[-1] > values[0]:
                        trend = "increasing"
                    elif values[-1] < values[0]:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                    
                    insights.append({
                        "model": model_name,
                        "trend": trend,
                        "change_percentage": ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0,
                        "volatility": np.std(values) if len(values) > 1 else 0
                    })
        
        if insights:
            avg_change = np.mean([insight['change_percentage'] for insight in insights])
            
            if avg_change > 10:
                insights.append({
                    "type": "recommendation",
                    "message": "Strong growth trend detected. Consider scaling operations and inventory management."
                })
            elif avg_change < -10:
                insights.append({
                    "type": "recommendation", 
                    "message": "Declining trend identified. Review market conditions and consider corrective actions."
                })
            else:
                insights.append({
                    "type": "recommendation",
                    "message": "Stable forecast indicates consistent demand. Focus on efficiency optimization."
                })
        
    except Exception as e:
        print(f"Error generating insights: {e}")
    
    return insights

async def fallback_chat_response(request: ChatRequest):
    context_prompt = ""
    if request.context:
        context_prompt = f"Data context: {json.dumps(request.context, indent=2)}\n\n"
    
    query_lower = request.message.lower()
    responses = {
        "forecast": "Based on your forecast data analysis, I can see trends and patterns that suggest specific actions. The predictive models show valuable insights for planning and decision-making.",
        "trend": "The forecast data reveals important trends in your metrics. These patterns can guide strategic planning and help identify opportunities for optimization.",
        "accuracy": "Model accuracy metrics in your forecast help assess prediction reliability. Consider ensemble methods or model tuning for improved performance.",
        "optimize": "Your forecast data suggests optimization opportunities. Focus on high-impact periods and consider implementing automated responses to predicted changes.",
        "risk": "Risk analysis from forecast data shows potential volatility. Implement monitoring systems and contingency plans for predicted fluctuations.",
        "default": "I can analyze your forecast data and provide insights on trends, accuracy, optimization opportunities, and risk management strategies. What specific aspect interests you most?"
    }
    
    if any(word in query_lower for word in ["forecast", "predict", "future", "trend"]):
        response = responses["forecast"]
    elif any(word in query_lower for word in ["trend", "pattern", "direction"]):
        response = responses["trend"]
    elif any(word in query_lower for word in ["accuracy", "performance", "metrics"]):
        response = responses["accuracy"]
    elif any(word in query_lower for word in ["optimize", "improve", "efficiency"]):
        response = responses["optimize"]
    elif any(word in query_lower for word in ["risk", "volatility", "uncertainty"]):
        response = responses["risk"]
    else:
        response = responses["default"]
    
    return {
        "status": "success",
        "response": response,
        "fallback_used": True,
        "context_used": bool(request.context)
    }
@app.post("/api/optimize/inventory")
async def optimize_inventory(request: OptimizationRequest):
    try:
        inventory_df = pd.DataFrame(request.inventory_data)
        demand_df = pd.DataFrame(request.demand_data)

        inventory_analysis = UniversalDataAnalyzer.analyze_dataset_structure(inventory_df)
        demand_analysis = UniversalDataAnalyzer.analyze_dataset_structure(demand_df)
        inventory_metrics = UniversalDataAnalyzer.extract_universal_metrics(inventory_df, inventory_analysis)
        demand_metrics = UniversalDataAnalyzer.extract_universal_metrics(demand_df, demand_analysis)
        recommendations = UniversalDataAnalyzer._generate_optimization_recommendations(
            inventory_df, demand_df, inventory_analysis, demand_analysis
        )
        

        cost_savings = np.random.uniform(10, 40) # replace with actual cost
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "inventory_analysis": inventory_analysis,
            "demand_analysis": demand_analysis,
            "inventory_metrics": inventory_metrics,
            "demand_metrics": demand_metrics,
            "estimated_cost_savings_percent": cost_savings
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "features": [
            "universal_data_analysis",
            "dynamic_forecasting",
            "flexible_optimization",
            "pattern_recognition",
            "automatic_column_detection"
        ]
    }


UniversalDataAnalyzer._clean_data_universally = staticmethod(
    lambda df, analysis: df.fillna(df.mean(numeric_only=True)).fillna("Unknown")
)

UniversalDataAnalyzer._generate_optimization_recommendations = staticmethod(
    lambda inv_df, dem_df, inv_analysis, dem_analysis: [
        f"Analyzed {len(inv_df)} inventory records and {len(dem_df)} demand records",
        "Implement automated data quality monitoring",
        "Consider implementing real-time analytics dashboards",
        "Optimize data collection processes based on identified patterns",
        "Focus on high-impact variables for maximum ROI",
        "Implement predictive analytics for proactive decision making"
    ]
)
class AgentManager:
    @staticmethod
    def create_agent(name: str, description: str, system_prompt: str = "", code: str = "", model: str = "claude-3-sonnet") -> dict:
        """Create a new AI agent"""
        agent_id = str(uuid.uuid4())
        enhanced_prompt = AgentManager._enhance_system_prompt(system_prompt, name, description)
        
        agent = {
            'id': agent_id,
            'name': name,
            'description': description,
            'systemPrompt': enhanced_prompt,
            'originalPrompt': system_prompt,
            'code': code,
            'model': model,
            'status': 'active',
            'createdAt': datetime.now().isoformat(),
            'lastExecuted': None,
            'executionCount': 0
        }
        
        agents_store[agent_id] = agent
        executions_store[agent_id] = []
        
        return agent
    
    @staticmethod
    def _enhance_system_prompt(original_prompt: str, name: str, description: str) -> str:
      
        if not original_prompt:
            original_prompt = f"You are {name}, an AI agent designed to {description.lower()}."
        
        enhancement_prompt = f"""
        Please enhance this system prompt for an AI agent to make it more effective and comprehensive:

        Agent Name: {name}
        Description: {description}
        Original Prompt: {original_prompt}

        Please provide an enhanced version that:
        1. Clearly defines the agent's role and expertise
        2. Includes relevant instructions for behavior
        3. Specifies output format preferences
        4. Adds helpful context about capabilities
        5. Maintains the original intent while improving clarity

        Return only the enhanced system prompt, nothing else.
        """
        
        try:
            response = bedrock_runtime.invoke_model(
                modelId=AVAILABLE_MODELS['claude-3-sonnet'],
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 1000,
                    'messages': [
                        {
                            'role': 'user',
                            'content': enhancement_prompt
                        }
                    ]
                })
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text'].strip()
            
        except Exception as e:
            print(f"Error enhancing prompt: {e}")
            return original_prompt
    
    @staticmethod
    def get_agents() -> List[dict]:
        return list(agents_store.values())
    
    @staticmethod
    def get_agent(agent_id: str) -> Optional[dict]:
            return agents_store.get(agent_id)
    
    @staticmethod
    def delete_agent(agent_id: str) -> bool:
        if agent_id in agents_store:
            del agents_store[agent_id]
            if agent_id in executions_store:
                del executions_store[agent_id]
            return True
        return False
    
    @staticmethod
    def execute_agent(agent_id: str, input_text: str, context: str = "") -> dict:
        agent = agents_store.get(agent_id)
        if not agent:
            return {'success': False, 'error': 'Agent not found'}
        
        start_time = time.time()
        
        try:
            full_prompt = agent['systemPrompt']
            if context:
                full_prompt += f"\n\nAdditional Context:\n{context}"
            
            if agent['code']:
                full_prompt += f"\n\nRelevant Code Context:\n```\n{agent['code']}\n```"
            
            full_prompt += f"\n\nUser Input: {input_text}"
            if agent['model'].startswith('claude'):
                result = AgentManager._execute_claude(agent['model'], full_prompt)
            elif agent['model'] == 'titan-text':
                result = AgentManager._execute_titan(full_prompt)
            elif agent['model'] == 'llama2-70b':
                result = AgentManager._execute_llama(full_prompt)
            else:
                result = {'success': False, 'error': 'Unsupported model'}
            
            execution_time = int((time.time() - start_time) * 1000)

            execution = {
                'id': str(uuid.uuid4()),
                'agentId': agent_id,
                'input': input_text,
                'context': context,
                'output': result.get('output', ''),
                'success': result['success'],
                'error': result.get('error'),
                'executionTime': execution_time,
                'timestamp': datetime.now().isoformat()
            }
   
            executions_store[agent_id].append(execution)
            agent['executionCount'] += 1
            agent['lastExecuted'] = datetime.now().isoformat()
            
            return execution
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return {
                'id': str(uuid.uuid4()),
                'agentId': agent_id,
                'input': input_text,
                'context': context,
                'output': '',
                'success': False,
                'error': str(e),
                'executionTime': execution_time,
                'timestamp': datetime.now().isoformat()
            }
    
    @staticmethod
    def _execute_claude(model: str, prompt: str) -> dict:
        try:
            response = bedrock_runtime.invoke_model(
                modelId=AVAILABLE_MODELS[model],
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 2000,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                })
            )
            
            result = json.loads(response['body'].read())
            return {
                'success': True,
                'output': result['content'][0]['text']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _execute_titan(prompt: str) -> dict:
        try:
            response = bedrock_runtime.invoke_model(
                modelId=AVAILABLE_MODELS['titan-text'],
                body=json.dumps({
                    'inputText': prompt,
                    'textGenerationConfig': {
                        'maxTokenCount': 2000,
                        'temperature': 0.7,
                        'topP': 0.9
                    }
                })
            )
            
            result = json.loads(response['body'].read())
            return {
                'success': True,
                'output': result['results'][0]['outputText']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _execute_llama(prompt: str) -> dict:
        try:
            response = bedrock_runtime.invoke_model(
                modelId=AVAILABLE_MODELS['llama2-70b'],
                body=json.dumps({
                    'prompt': prompt,
                    'max_gen_len': 2000,
                    'temperature': 0.7,
                    'top_p': 0.9
                })
            )
            
            result = json.loads(response['body'].read())
            return {
                'success': True,
                'output': result['generation']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_executions(agent_id: str) -> List[dict]:
        return executions_store.get(agent_id, [])


@app.get('/api/agents')
async def get_agents():
    try:
        agents = AgentManager.get_agents()
        return {"success": True, "agents": agents}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post('/api/agents')
async def create_agent(request: Request):
    try:
        data = await request.json()
        if not data.get('name') or not data.get('description'):
            return JSONResponse(status_code=400, content={
                'success': False,
                'error': 'Name and description are required'
            })

        agent = AgentManager.create_agent(
            name=data['name'],
            description=data['description'],
            system_prompt=data.get('systemPrompt', ''),
            code=data.get('code', ''),
            model=data.get('model', 'claude-3-sonnet')
        )

        return {"success": True, "agent": agent}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get('/api/agents/{agent_id}')
async def get_agent(agent_id: str):
    try:
        agent = AgentManager.get_agent(agent_id)
        if not agent:
            return JSONResponse(status_code=404, content={
                'success': False,
                'error': 'Agent not found'
            })
        return {"success": True, "agent": agent}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.delete('/api/agents/{agent_id}')
async def delete_agent(agent_id: str):
    try:
        success = AgentManager.delete_agent(agent_id)
        if not success:
            return JSONResponse(status_code=404, content={
                'success': False,
                'error': 'Agent not found'
            })
        return {"success": True, "message": "Agent deleted successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post('/api/agents/{agent_id}/execute')
async def execute_agent(agent_id: str, request: Request):
    try:
        data = await request.json()
        if not data.get('input'):
            return JSONResponse(status_code=400, content={
                'success': False,
                'error': 'Input is required'
            })

        execution = AgentManager.execute_agent(
            agent_id=agent_id,
            input_text=data['input'],
            context=data.get('context', '')
        )

        return {"success": True, "execution": execution}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get('/api/agents/{agent_id}/executions')
async def get_executions(agent_id: str):
    try:
        executions = AgentManager.get_executions(agent_id)
        executions.sort(key=lambda x: x['timestamp'], reverse=True)
        return {"success": True, "executions": executions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/models")
async def get_models():
    """Get available models"""
    return {"success": True, "models": list(AVAILABLE_MODELS.keys())}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)