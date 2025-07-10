from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import json
import uuid
import time
from datetime import datetime
import asyncio
import traceback
from contextlib import contextmanager
import os

class AgentCreate(BaseModel):
    name: str
    description: str
    systemPrompt: str
    code: Optional[str] = ""

class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    systemPrompt: str
    code: str
    status: str
    executionCount: int
    createdAt: str
    lastExecuted: Optional[str] = None

class ExecutionRequest(BaseModel):
    input: str
    context: Optional[str] = ""

class ExecutionResponse(BaseModel):
    id: str
    agentId: str
    input: str
    context: str
    output: str
    success: bool
    executionTime: int
    timestamp: str
    error: Optional[str] = None

class APIResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class AgentListResponse(APIResponse):
    agents: List[AgentResponse]

class AgentCreateResponse(APIResponse):
    agent: Optional[AgentResponse] = None

class ExecutionListResponse(APIResponse):
    executions: List[ExecutionResponse]

class ExecutionCreateResponse(APIResponse):
    execution: Optional[ExecutionResponse] = None

DATABASE_PATH = "ai_lab.db"

def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            code TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            execution_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_executed TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS executions (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            input TEXT NOT NULL,
            context TEXT DEFAULT '',
            output TEXT NOT NULL,
            success BOOLEAN NOT NULL,
            execution_time INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error TEXT,
            FOREIGN KEY (agent_id) REFERENCES agents (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

app = FastAPI(
    title="AI Agent Lab Backend",
    description="Backend API for AI Agent Lab - Create, manage, and execute AI agents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_database()

def row_to_dict(row):
    """Convert sqlite3.Row to dictionary"""
    return {key: row[key] for key in row.keys()}

def enhance_system_prompt(user_prompt: str, agent_name: str) -> str:
    """Enhance user's system prompt with Claude-style improvements"""
    enhanced_prompt = f"""You are {agent_name}, a specialized AI agent designed to help with specific tasks.

Core Identity:
{user_prompt}

Enhanced Capabilities:
- You provide detailed, accurate, and helpful responses
- You think step-by-step through complex problems
- You ask clarifying questions when needed
- You provide examples and explanations when helpful
- You maintain a professional yet friendly tone
- You adapt your communication style to the user's needs

Response Guidelines:
- Be concise but thorough
- Use clear, structured formatting when appropriate
- Provide actionable insights and recommendations
- Acknowledge limitations and uncertainties
- Focus on being genuinely helpful

Remember: You are an expert in your domain. Use your knowledge effectively to assist users with their specific needs."""
    
    return enhanced_prompt

async def simulate_ai_execution(agent: dict, input_text: str, context: str = "") -> tuple[str, bool, str]:
    try:
        # Simulate processing time
        await asyncio.sleep(0.5 + (len(input_text) / 100))
        
        # Create a simulated response based on the agent's purpose
        agent_name = agent['name']
        system_prompt = agent['system_prompt']
        
        # Simple response generation based on agent type
        if 'optimizer' in agent_name.lower() or 'optimization' in system_prompt.lower():
            response = f"""Based on the analysis of: "{input_text}"

Optimization Recommendations:
1. Primary improvement: Streamline the core process by 15-20%
2. Resource allocation: Redistribute resources to high-impact areas
3. Timeline optimization: Reduce bottlenecks in the critical path
4. Quality enhancement: Implement automated quality checks

Key Metrics to Monitor:
- Efficiency gains: 15-25% improvement expected
- Cost reduction: 10-15% savings potential
- Quality score: Target 95%+ consistency

Next Steps:
1. Implement recommended changes in pilot phase
2. Monitor performance metrics weekly
3. Scale successful optimizations

Context considered: {context if context else 'No additional context provided'}"""
        
        elif 'analyzer' in agent_name.lower() or 'analysis' in system_prompt.lower():
            response = f"""Analysis Report for: "{input_text}"

Executive Summary:
The analysis reveals several key patterns and insights that require attention.

Detailed Findings:
1. Primary factors: Three main drivers identified
2. Risk assessment: Medium risk level with controllable variables
3. Opportunity identification: 2-3 high-value opportunities detected
4. Performance indicators: Mixed results with improvement potential

Recommendations:
• Focus on the top 2 priority areas for immediate impact
• Implement monitoring systems for key metrics
• Consider phased approach for complex changes

Additional Context: {context if context else 'Standard analysis parameters applied'}"""
        
        elif 'generator' in agent_name.lower() or 'creative' in system_prompt.lower():
            response = f"""Creative Output for: "{input_text}"

Generated Content:
Here's a creative approach to your request that balances innovation with practicality.

Key Elements:
- Unique perspective: Fresh angle on the core concept
- Practical application: Real-world implementation considerations
- Creative enhancement: Added value through innovative thinking
- User-focused design: Tailored to specific needs and constraints

Variations:
1. Option A: Bold and innovative approach
2. Option B: Balanced traditional-modern hybrid
3. Option C: Conservative but reliable solution

Implementation Notes:
Consider your target audience and constraints when selecting the best approach.

Context Integration: {context if context else 'General creative parameters applied'}"""
        
        else:
            # Generic helpful response
            response = f"""Response to: "{input_text}"

Analysis:
I've processed your request and here's my comprehensive response.

Key Points:
1. Understanding: Your request focuses on finding effective solutions
2. Approach: I'll provide structured, actionable information
3. Considerations: Multiple factors have been evaluated
4. Recommendations: Based on best practices and current standards

Detailed Response:
Your request has been carefully analyzed. The most effective approach would be to:
- Start with a clear assessment of current state
- Identify specific goals and success metrics
- Develop a step-by-step implementation plan
- Monitor progress and adjust as needed

Additional Insights:
Context provided: {context if context else 'No specific context'}
Confidence level: High
Recommended follow-up: Review and validate approach"""
        
        return response, True, ""
        
    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        return error_msg, False, error_msg

# API Routes

@app.get("/")
async def root():
    return {"message": "AI Agent Lab Backend API", "status": "running"}

@app.get("/api/agents", response_model=AgentListResponse)
async def get_agents():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, system_prompt, code, status, 
                       execution_count, created_at, last_executed
                FROM agents 
                ORDER BY created_at DESC
            """)
            
            agents = []
            for row in cursor.fetchall():
                agent = AgentResponse(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    systemPrompt=row['system_prompt'],
                    code=row['code'] or "",
                    status=row['status'],
                    executionCount=row['execution_count'],
                    createdAt=row['created_at'],
                    lastExecuted=row['last_executed']
                )
                agents.append(agent)
            
            return AgentListResponse(success=True, agents=agents)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/agents", response_model=AgentCreateResponse)
async def create_agent(agent_data: AgentCreate):
    try:
        if not agent_data.name.strip():
            raise HTTPException(status_code=400, detail="Agent name is required")
        if not agent_data.description.strip():
            raise HTTPException(status_code=400, detail="Agent description is required")
        agent_id = str(uuid.uuid4())
        enhanced_prompt = enhance_system_prompt(agent_data.systemPrompt, agent_data.name)
        
        with get_db_connection() as conn: #create agent
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM agents WHERE name = ?", (agent_data.name,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Agent with this name already exists")
            
            cursor.execute("""
                INSERT INTO agents (id, name, description, system_prompt, code, status, execution_count)
                VALUES (?, ?, ?, ?, ?, 'active', 0)
            """, (agent_id, agent_data.name, agent_data.description, enhanced_prompt, agent_data.code or ""))
            
            conn.commit()
            cursor.execute("""
                SELECT id, name, description, system_prompt, code, status, 
                       execution_count, created_at, last_executed
                FROM agents WHERE id = ?
            """, (agent_id,))
            
            row = cursor.fetchone()
            agent = AgentResponse(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                systemPrompt=row['system_prompt'],
                code=row['code'] or "",
                status=row['status'],
                executionCount=row['execution_count'],
                createdAt=row['created_at'],
                lastExecuted=row['last_executed']
            )
            
            return AgentCreateResponse(success=True, agent=agent, message="Agent created successfully")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM agents WHERE id = ?", (agent_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Agent not found")
            cursor.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            conn.commit()
            
            return {"success": True, "message": "Agent deleted successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

@app.post("/api/agents/{agent_id}/execute", response_model=ExecutionCreateResponse)
async def execute_agent(agent_id: str, execution_data: ExecutionRequest):
    try:
        start_time = time.time()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, system_prompt, code, status
                FROM agents WHERE id = ?
            """, (agent_id,))
            
            agent_row = cursor.fetchone()
            if not agent_row:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            agent = row_to_dict(agent_row)
            
            if agent['status'] != 'active':
                raise HTTPException(status_code=400, detail="Agent is not active")
            output, success, error = await simulate_ai_execution(
                agent, 
                execution_data.input, 
                execution_data.context
            )
            
            execution_time = int((time.time() - start_time) * 1000)  
            execution_id = str(uuid.uuid4())

            cursor.execute("""
                INSERT INTO executions (id, agent_id, input, context, output, success, execution_time, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (execution_id, agent_id, execution_data.input, execution_data.context or "", 
                  output, success, execution_time, error))
            
            cursor.execute("""
                UPDATE agents 
                SET execution_count = execution_count + 1, last_executed = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (agent_id,))
            
            conn.commit()
            execution = ExecutionResponse(
                id=execution_id,
                agentId=agent_id,
                input=execution_data.input,
                context=execution_data.context or "",
                output=output,
                success=success,
                executionTime=execution_time,
                timestamp=datetime.now().isoformat(),
                error=error if not success else None
            )
            
            return ExecutionCreateResponse(
                success=True, 
                execution=execution, 
                message="Agent executed successfully"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/agents/{agent_id}/executions", response_model=ExecutionListResponse)
async def get_agent_executions(agent_id: str, limit: int = 10):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM agents WHERE id = ?", (agent_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Agent not found")
            cursor.execute("""
                SELECT id, agent_id, input, context, output, success, execution_time, timestamp, error
                FROM executions 
                WHERE agent_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (agent_id, limit))
            
            executions = []
            for row in cursor.fetchall():
                execution = ExecutionResponse(
                    id=row['id'],
                    agentId=row['agent_id'],
                    input=row['input'],
                    context=row['context'],
                    output=row['output'],
                    success=bool(row['success']),
                    executionTime=row['execution_time'],
                    timestamp=row['timestamp'],
                    error=row['error']
                )
                executions.append(execution)
            
            return ExecutionListResponse(success=True, executions=executions)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get executions: {str(e)}")

@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, system_prompt, code, status, 
                       execution_count, created_at, last_executed
                FROM agents WHERE id = ?
            """, (agent_id,))
            
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            agent = AgentResponse(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                systemPrompt=row['system_prompt'],
                code=row['code'] or "",
                status=row['status'],
                executionCount=row['execution_count'],
                createdAt=row['created_at'],
                lastExecuted=row['last_executed']
            )
            
            return {"success": True, "agent": agent}
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)