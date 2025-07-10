# GenAI-Hackathon-by-AWS
**Supply Chain OS** is an end-to-end AI-powered platform for optimizing supply chain operations using Temporal LLMs, Generative AI, and time-series forecasting. It enables smarter decision-making, proactive risk management, and autonomous agent-based execution for complex supply chain workflows.

## Key Features

- **Universal Data Analyzer**: Ingests and processes structured and unstructured data across ERP, WMS, IoT, and external sources.
- **Time Series Forecasting**: Supports both traditional methods (ARIMA, Prophet) and advanced LLM-driven predictions.
- **Supply Chain AI Assistant**: Offers contextual insights, anomaly detection, and recommendation generation.
- **AI Agent Lab**: Allows creation, deployment, and orchestration of autonomous agents for planning, procurement, inventory, etc.


# Installations

- fastapi==0.104.1
- uvicorn==0.24.0
- pydantic==2.5.0
- boto3==1.34.0
- pandas==2.1.4
- numpy==1.24.3
- scikit-learn==1.3.2
- prophet==1.1.4
- openai==1.3.7
- python-multipart==0.0.6
- python-dotenv==1.0.0
- prophet
- neuralprophet
- statsmodels
- plotly

# SETUP Instructions
git clone https://github.com/anan-123  
cd GenAI-Hackathon-by-AWS  

## Frontend
cd Frontend  
npm install   
npm run dev  

## Backend
cd Backend  
python main.py  

## Technology Stack

- **Backend:** FastAPI (Python)
- **Frontend:** React + Tailwind
- **AI Models:** Claude 3, ARIMA, Prophet
- **Integration:** AWS Bedrock + LangChain
- **Forecasting:** Darts, Sklearn
- **Dataset Input:** CSV, S3 URL

## Additional Capabilities: 

agents.py — Used for creating and storing agents in SQL tables. Designed as a scalable, extensible solution.  
main.py — Uncomment models in the timeseries function to enable additional forecasting models.  
The code for models works primarily on the provided dataset in the data folder. The data analyzer works on lot more datasets and the llm based models work well than traditional datasets for different datasets.

## Dependencies

- **Python Libraries**: `pydantic`, `torch` or `tensorflow`, `prophet`, `langchain`, etc.
- **Cloud Services**: AWS IAM, S3, EKS/ECS (optional but recommended)
- **LLM Models**: OpenAI API, Hugging Face Transformers, or private LLM deployments
- **Data Infrastructure**: PostgreSQL, Kafka, Redis, or other scalable services
