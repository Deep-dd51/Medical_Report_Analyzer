# app.py
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import chromadb.utils.embedding_functions as ef
ef.DefaultEmbeddingFunction = lambda: ef.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

from PIL import Image
import pdfplumber
import streamlit as st
import re
import json

from typing import Type
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew,LLM
from crewai.tools import BaseTool

# === TOOLS ===

class OCRInput(BaseModel):
    file_path: str = Field(..., description="Path to uploaded PDF or image")

class OCRTool(BaseTool):
    name: str = "OCR Tool"
    description: str = "Extracts text from a medical PDF or image file"
    args_schema: Type[BaseModel] = OCRInput

    def _run(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: File not found at path: {file_path}"
        
        try:
            if file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            else:
                img = Image.open(file_path)
                return pytesseract.image_to_string(img)
        except Exception as e:
            return f"Error processing file: {str(e)}"

class ParserInput(BaseModel):
    text: str = Field(..., description="Raw text extracted from the report")

class ParserTool(BaseTool):
    name: str = "Parser Tool"
    description: str = "Extracts medical test names and values from raw text"
    args_schema: Type[BaseModel] = ParserInput

    def _run(self, text: str) -> str:
        results = {}
        for line in text.split("\n"):
            match = re.match(r"([A-Za-z \(\)]+)\s+(\d+\.?\d*)", line)
            if match:
                results[match.group(1).strip()] = match.group(2)
        return json.dumps(results)  # Use JSON for safer serialization


class AnalyzerInput(BaseModel):
    results: str = Field(..., description="JSON string of parsed test results")

import openai
import json

class AnalyzerTool(BaseTool):
    name: str = "Analyzer Tool"
    description: str = "Analyzes medical test values using OpenAI for intelligent assessment"
    args_schema: Type[BaseModel] = AnalyzerInput

    def _run(self, results: str) -> str:
        try:
            tests = json.loads(results)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        
        # Create a prompt for OpenAI to analyze the medical results
        test_data = "\n".join([f"{test}: {value}" for test, value in tests.items()])
        
        prompt = f"""You are a medical analysis assistant. Analyze the following medical test results and determine if each value is Normal, High, Low, or requires attention. 

Medical Test Results:
{test_data}

For each test, provide:
1. The test name
2. The value
3. Status (Normal/High/Low/Attention Required)
4. Brief explanation if abnormal

Return the analysis in JSON format like this:
{{
    "test_name": {{
        "value": "original_value",
        "status": "Normal/High/Low/Attention Required",
        "explanation": "Brief explanation if needed"
    }}
}}

Use standard medical reference ranges for common tests like Hemoglobin, WBC Count, Platelets, Cholesterol, etc."""

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical analysis expert with knowledge of standard laboratory reference ranges."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent medical analysis
                max_tokens=1000
            )
            
            # Extract the JSON response
            analysis_text = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                analyzed_results = json.loads(analysis_text)
                return json.dumps(analyzed_results)
            except json.JSONDecodeError:
                # If OpenAI doesn't return valid JSON, create a structured response
                return json.dumps({
                    "analysis": analysis_text,
                    "note": "OpenAI provided text analysis instead of structured JSON"
                })
                
        except Exception as e:
            return json.dumps({"error": f"OpenAI analysis failed: {str(e)}"})

class SummaryInput(BaseModel):
    analysis: str = Field(..., description="JSON string of analyzed medical test results")

import openai 

class SummaryTool(BaseTool):
    name: str = "Summary Tool"
    description: str = "Summarizes analyzed results in user-friendly format using GPT"
    args_schema: Type[BaseModel] = SummaryInput

    def _run(self, analysis: str) -> str:
        try:
            data = json.loads(analysis)
        except json.JSONDecodeError as e:
            return f"Error: Unable to parse analysis data - {str(e)}"

        if "error" in data:
            return f"Analysis Error: {data['error']}"

        # Construct a readable version of the data for GPT
        structured_summary = ""
        for test, info in data.items():
            val = info.get("value", "N/A")
            status = info.get("status", "Unknown")
            structured_summary += f"{test}: {val} ({status})\n"

        prompt = f"""You are a medical assistant helping patients understand their test reports. 
Given the following analyzed medical data, write a friendly and simple layman summary explaining the results clearly and gently:

{structured_summary}

Avoid medical jargon. Use bullet points. Provide actionable tips if needed.
"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            st.markdown("📋 *Medical Report Summary*", unsafe_allow_html=True, color="green")
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ GPT-4 Summary Error: {str(e)}"



# === STREAMLIT APP ===

st.set_page_config(page_title="🧠 Medical Report Analyzer", layout="centered")
st.title("🩺 AI-Powered Medical Report Analyzer")
uploaded_file = st.file_uploader("📤 Upload Medical Report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if st.button("🧪 Analyze Report"):
        with st.spinner("🔍 Running multi-agent analysis..."):

            # Define tools
            ocr_tool = OCRTool()
            parser_tool = ParserTool()
            analyzer_tool = AnalyzerTool()
            summary_tool = SummaryTool()

            # Define agents
            ocr_agent = Agent(
                role="OCR Specialist",
                goal="Extract raw medical text from uploaded reports.",
                backstory="Expert in OCR technology and document parsing.",
                tools=[ocr_tool],
                allow_delegation=False
            )

            parser_agent = Agent(
                role="Data Extractor",
                goal="Identify test names and numeric values from raw text.",
                backstory="Specializes in regex parsing and structure detection in documents.",
                tools=[parser_tool],
                allow_delegation=False
            )

            analyzer_agent = Agent(
                role="Medical Analyst",
                goal="Analyze the parsed test results and flag abnormalities.",
                backstory="Doctor who evaluates lab data against clinical reference ranges.",
                tools=[analyzer_tool],
                allow_delegation=False,
                llm=LLM(model="gpt-4")  
            )

            summary_agent = Agent(
                role="Layman Summary Generator",
                goal="Convert clinical analysis into a friendly summary",
                backstory="You're a patient-friendly AI trained to explain lab reports simply.",
                tools=[],
                allow_delegation=False,
                llm=LLM(model="gpt-4")  
            )


           
            # Define tasks with better context passing
            task1 = Task(
                description="Extract text from the medical report located at {file_path}",
                agent=ocr_agent,
                expected_output="Raw medical text extracted from the document"
            )

            task2 = Task(
                description="Parse test values from the extracted text. Use the raw text from the previous task.", 
                agent=parser_agent, 
                expected_output="JSON string containing parsed test names and values", 
                context=[task1]
            )

            task3 = Task(
                description="Analyze the parsed test results for abnormalities. Use the JSON data from the previous task.", 
                agent=analyzer_agent, 
                expected_output="JSON string showing analysis of normal vs abnormal test results", 
                context=[task2]
            )

            task4 = Task(
                description="Convert the analysis JSON into a simple, friendly summary. Avoid medical jargon. Output bullet points.",
                agent=summary_agent,
                expected_output="Layman's summary of the medical analysis",
                context=[task3]
            )


            
            # Create crew and run with inputs
            crew = Crew(
                agents=[ocr_agent, parser_agent, analyzer_agent, summary_agent], 
                tasks=[task1, task2, task3, task4], 
                verbose=True
            )
            
            # Pass the file path as input to kickoff
            result = crew.kickoff(inputs={"file_path": temp_path})

        st.success("✅ Analysis Complete!")
        st.markdown(result.raw)

        os.remove(temp_path)
