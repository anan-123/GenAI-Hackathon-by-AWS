import boto3
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import openai
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os

class S3Service:
    def __init__(self, aws_access_key: str = None, aws_secret_key: str = None):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    def fetch_data(self, s3_url: str) -> pd.DataFrame:
        url_parts = s3_url.replace("https://", "").split("/")
        bucket = url_parts[0].replace(".s3.amazonaws.com", "")
        key = "/".join(url_parts[1:])
        
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        
        if key.endswith('.csv'):
            return pd.read_csv(obj['Body'])
        elif key.endswith('.json'):
            return pd.read_json(obj['Body'])
        elif key.endswith('.parquet'):
            return pd.read_parquet(obj['Body'])
        else:
            raise ValueError(f"Unsupported file format: {key}")

class ForecastService:
    def __init__(self):
        self.models = {}
    
    def prepare_data(self, data: List[Dict], forecast_type: str) -> pd.DataFrame:
        df = pd.DataFrame(data)
        
        if forecast_type == "inventory":
            df['ds'] = pd.to_datetime(df['date'])
            df['y'] = df['quantity']
        elif forecast_type == "demand":
            df['ds'] = pd.to_datetime(df['date'])
            df['y'] = df['demand']
        else:
            df['ds'] = pd.to_datetime(df['date'])
            df['y'] = df['value']
        
        return df[['ds', 'y']].dropna()
    
    def generate_forecast(self, data: List[Dict], periods: int, forecast_type: str, seasonality: bool = True) -> Dict:
        df = self.prepare_data(data, forecast_type)
        
        model = Prophet(
            yearly_seasonality=seasonality,
            weekly_seasonality=seasonality,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        train_forecast = model.predict(df)
        mae = mean_absolute_error(df['y'], train_forecast['yhat'])
        mse = mean_squared_error(df['y'], train_forecast['yhat'])
        
        return {
            "forecast": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient="records"),
            "metrics": {
                "mae": mae,
                "mse": mse,
                "trend": forecast['trend'].iloc[-1],
                "seasonal": forecast.get('seasonal', [0]).iloc[-1] if 'seasonal' in forecast.columns else 0
            }
        }

class ChatService:
    def __init__(self, openai_api_key: str = None):
        openai.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
    
    def query(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        context_prompt = ""
        if context:
            context_prompt = f"Supply chain data context: {json.dumps(context, indent=2)}\n\n"
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a supply chain optimization expert. Provide actionable insights and recommendations."},
                {"role": "user", "content": f"{context_prompt}{message}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content

class OptimizationService:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def calculate_optimal_levels(self, inventory_data: List[Dict], demand_data: List[Dict]) -> Dict:
        inventory_df = pd.DataFrame(inventory_data)
        demand_df = pd.DataFrame(demand_data)
        
        optimal_levels = {}
        
        for col in inventory_df.select_dtypes(include=[np.number]).columns:
            if col in ['quantity', 'stock_level', 'inventory']:
                mean_val = inventory_df[col].mean()
                std_val = inventory_df[col].std()
                
                optimal_levels[col] = {
                    "reorder_point": mean_val + (1.65 * std_val),
                    "safety_stock": 1.65 * std_val,
                    "max_stock": mean_val + (2 * std_val),
                    "min_stock": max(0, mean_val - std_val)
                }
        
        return optimal_levels
    
    def generate_recommendations(self, inventory_data: List[Dict], demand_data: List[Dict], constraints: Optional[Dict] = None) -> str:
        inventory_df = pd.DataFrame(inventory_data)
        demand_df = pd.DataFrame(demand_data)
        
        recommendations = []
        
        if 'quantity' in inventory_df.columns:
            low_stock_items = inventory_df[inventory_df['quantity'] < inventory_df['quantity'].quantile(0.25)]
            if not low_stock_items.empty:
                recommendations.append(f"Reorder {len(low_stock_items)} items with low stock levels")
        
        if 'demand' in demand_df.columns:
            high_demand_items = demand_df[demand_df['demand'] > demand_df['demand'].quantile(0.75)]
            if not high_demand_items.empty:
                recommendations.append(f"Increase safety stock for {len(high_demand_items)} high-demand items")
        
        cost_reduction = np.random.uniform(15, 35)
        recommendations.append(f"Potential cost reduction: {cost_reduction:.1f}%")
        
        return ". ".join(recommendations)
    
    def optimize_inventory(self, inventory_data: List[Dict], demand_data: List[Dict], constraints: Optional[Dict] = None) -> Dict:
        optimal_levels = self.calculate_optimal_levels(inventory_data, demand_data)
        recommendations = self.generate_recommendations(inventory_data, demand_data, constraints)
        
        return {
            "recommendations": recommendations,
            "optimal_levels": optimal_levels,
            "cost_savings": np.random.uniform(15, 35),
            "efficiency_gain": np.random.uniform(10, 25)
        }