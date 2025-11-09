from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import TypedDict, List
import json

# --- 1. SETUP AND CONFIGURATION ---

load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# Global model variable
model = None

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please add it to your .env file.")
    
    # Configure Google Generative AI
    genai.configure(api_key=api_key)
    
    # Try to initialize with different model names (updated for Gemini 2.0/2.5)
    model_names = [
        'models/gemini-2.5-flash',
        'models/gemini-2.0-flash',
        'models/gemini-flash-latest',
        'models/gemini-pro-latest',
        'models/gemini-2.5-pro'
    ]
    
    for model_name in model_names:
        try:
            print(f"Trying model: {model_name}")
            test_model = genai.GenerativeModel(model_name)
            # Test with a simple prompt
            test_response = test_model.generate_content("Hello")
            if test_response and test_response.text:
                model = test_model
                print(f"✅ Model initialized successfully with: {model_name}")
                break
        except Exception as e:
            print(f"Failed with {model_name}: {str(e)[:100]}")
            continue
    
    if model is None:
        raise ValueError("Could not initialize any Gemini model. Please check your API key and internet connection.")

except Exception as e:
    print(f"Error initializing model: {e}")
    raise e

# --- 2. STATE DEFINITION ---

class PlacementAnalysisState(TypedDict):
    raw_data_text: str
    structured_data: str
    insights: str
    trend_comparison: str
    recommendations: str
    report: str
    notifications: str
    workflow_log: List[str]

# --- 3. AGENT FUNCTIONS ---

def call_llm(prompt: str, retry_count: int = 3) -> str:
    """Helper function to call the LLM with retry logic"""
    for attempt in range(retry_count):
        try:
            # Add safety settings to avoid blocks
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Check if response has text
            if response and response.text:
                return response.text
            elif response.prompt_feedback:
                return f"Response blocked: {response.prompt_feedback}"
            else:
                return "No response generated from the model."
                
        except Exception as e:
            error_msg = f"LLM Error (Attempt {attempt + 1}/{retry_count}): {str(e)}"
            print(error_msg)
            if attempt == retry_count - 1:
                return f"Failed after {retry_count} attempts: {str(e)}"
    
    return "Failed to generate response"

def data_collector_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 1: Data Collector - Processing all data files...")
    
    prompt = f"""You are a Data Collector Agent. You have received data from multiple sources (OKRs, Applications, Rejections, Mock Interviews). Your job is to read all this raw text and convert it into a single, clean, structured summary. Combine information for each student logically.

Raw Data:
{state['raw_data_text']}

Consolidated Structured Data Summary:
Provide the response in plain JSON text without any markdown formatting."""
    
    state["structured_data"] = call_llm(prompt)
    return state

def insight_synthesizer_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 2: Insight Synthesizer - Generating key insights...")
    
    prompt = f"""You are an Insight Synthesizer Agent. Analyze the structured placement data and generate 3-5 actionable insights about the root causes of placement rejections.

Structured Placement Data:
{state['structured_data']}

Actionable Insights:"""
    
    state["insights"] = call_llm(prompt)
    return state
    
def trend_comparator_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 3: Trend Comparator - Comparing with historical data...")
    
    try:
        # Check if historical data file exists
        if not os.path.exists('historical_data.txt'):
            state["trend_comparison"] = "No historical data file found. Unable to perform trend comparison."
            return state
            
        with open('historical_data.txt', 'r') as f:
            historical_data = f.read()

        prompt = f"""You are a Trend Comparator Agent. Compare the 'Current Placement Insights' with the 'Historical Data' provided. Highlight new trends or recurring problems.

Historical Data:
{historical_data}

Current Placement Insights:
{state['insights']}

Trend Comparison:"""

        state["trend_comparison"] = call_llm(prompt)

    except Exception as e:
        print(f"Error during trend comparison: {e}")
        state["trend_comparison"] = f"Could not create trend comparison due to an error: {str(e)}"

    return state
    
def action_recommender_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 4: Action Recommender - Suggesting actions...")
    
    prompt = f"""You are an Action Recommender Agent. Based on the insights and trends, suggest 3 concrete actions the college can take.

Placement Insights:
{state['insights']}

Trend Comparison:
{state['trend_comparison']}

Action Recommendations:"""
    
    state["recommendations"] = call_llm(prompt)
    return state

def report_generator_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 5: Report Generator - Compiling final report...")
    
    prompt = f"""You are a Report Generator Agent. Combine all information into a professional Markdown report with these sections: Key Insights Summary, Trend Comparison Analysis, and Actionable Recommendations.

Information to use:
- Key Insights: {state['insights']}
- Trend Comparison: {state['trend_comparison']}
- Recommendations: {state['recommendations']}

Final Placement Report:"""
    
    state["report"] = call_llm(prompt)
    return state

def stakeholder_notification_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 6: Stakeholder Notifier - Drafting communications...")
    
    prompt = f"""Based on the final report, draft a concise email to the Placement Head and a short notification for the college portal.

Report:
{state['report']}

Drafted Communications:"""
    
    state["notifications"] = call_llm(prompt)
    state['workflow_log'].append("✅ Workflow Complete!")
    return state

# --- 4. WORKFLOW EXECUTION ---

def run_workflow(initial_state: PlacementAnalysisState) -> PlacementAnalysisState:
    """Run the complete agent workflow"""
    state = initial_state.copy()
    
    # Execute agents in sequence
    state = data_collector_agent(state)
    state = insight_synthesizer_agent(state)
    state = trend_comparator_agent(state)
    state = action_recommender_agent(state)
    state = report_generator_agent(state)
    state = stakeholder_notification_agent(state)
    
    return state

# --- 5. FLASK API ROUTES ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to run the agentic workflow."""
    try:
        data = request.json
        raw_data_text = data.get('raw_data_text', '')

        if not raw_data_text:
            return jsonify({"error": "No data provided."}), 400

        initial_state = { 
            "raw_data_text": raw_data_text, 
            "structured_data": "",
            "insights": "",
            "trend_comparison": "",
            "recommendations": "",
            "report": "",
            "notifications": "",
            "workflow_log": [] 
        }
        
        final_state = run_workflow(initial_state)

        return jsonify(final_state)
    
    except Exception as e:
        print(f"Error in analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Runs the Flask web server
    app.run(debug=True, port=5001)