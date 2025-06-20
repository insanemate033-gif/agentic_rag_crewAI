import streamlit as st
import os
import tempfile
import gc
import base64
import time

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from src.agentic_rag.tool.custom_tool import DocumentSearchTool # Ensure this path is correct relative to app.py

# Import dotenv and ChatOpenAI for LLM setup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables (e.g., OPENAI_API_KEY, SERPER_API_KEY)
load_dotenv()

# ===========================
#   Global LLM Configuration
# ===========================
# Define the LLM here to be used by agents and tools
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), # Get OpenAI API key from .env
    model="gpt-4o",                      # Use gpt-4o as the model
    temperature=0.7,                     # Set temperature
    max_tokens=1000,                     # Set max_tokens
)

# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks(pdf_tool):
    """Creates a Crew with the given PDF tool (if any) and a web search tool."""
    # SerperDevTool will automatically pick up SERPER_API_KEY from os.environ
    web_search_tool = SerperDevTool()

    retriever_agent = Agent(
        role="Retrieve relevant information to answer the user query: {query}",
        goal=(
            "Retrieve the most relevant information from the available sources "
            "for the user query: {query}. Always try to use the PDF search tool first. "
            "If you are not able to retrieve the information from the PDF search tool, "
            "then try to use the web search tool."
        ),
        backstory=(
            "You're a meticulous analyst with a keen eye for detail. "
            "You're known for your ability to understand user queries: {query} "
            "and retrieve knowledge from the most suitable knowledge base."
        ),
        verbose=False, # Set to False for stability and less console output
        tools=[t for t in [pdf_tool, web_search_tool] if t],
        llm=llm # Pass the globally defined LLM to the agent
    )

    response_synthesizer_agent = Agent(
        role="Response synthesizer agent for the user query: {query}",
        goal=(
            "Synthesize the retrieved information into a concise and coherent response "
            "based on the user query: {query}. If you are not able to retrieve the "
            'information then respond with "I\'m sorry, I couldn\'t find the information '
            'you\'re looking for."'
        ),
        backstory=(
            "You're a skilled communicator with a knack for turning "
            "complex information into clear and concise responses."
        ),
        verbose=False, # Set to False for stability
        llm=llm # Pass the globally defined LLM to the agent
    )

    retrieval_task = Task(
        description=(
            "Retrieve the most relevant information from the available "
            "sources for the user query: {query}"
        ),
        expected_output=(
            "The most relevant information in the form of text as retrieved "
            "from the sources."
        ),
        agent=retriever_agent
    )

    response_task = Task(
        description="Synthesize the final response for the user query: {query}",
        expected_output=(
            "A concise and coherent response based on the retrieved information "
            "from the right source for the user query: {query}. If you are not "
            "able to retrieve the information, then respond with: "
            '"I\'m sorry, I couldn\'t find the information you\'re looking for."'
        ),
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,  # or Process.hierarchical
        verbose=False # Set to False for stability
    )
    return crew

# ===========================
#   Streamlit Setup
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None  # Store the DocumentSearchTool

if "crew" not in st.session_state:
    st.session_state.crew = None      # Store the Crew object

def reset_chat():
    st.session_state.messages = []
    # Clear PDF tool and crew to allow re-uploading PDF and re-initializing
    st.session_state.pdf_tool = None
    st.session_state.crew = None
    gc.collect()
    st.rerun() # Rerun app to clear display

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF in an iframe."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

# ===========================
#   Sidebar
# ===========================
with st.sidebar:
    st.header("Add Your PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Only process if a new file is uploaded or if pdf_tool is not yet set for this file
        if st.session_state.pdf_tool is None or st.session_state.pdf_tool.file_path != uploaded_file.name:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner("Indexing PDF... Please wait..."):
                    # Configure the LLM and embedder for the DocumentSearchTool itself
                    st.session_state.pdf_tool = DocumentSearchTool(
                        file_path=temp_file_path,
                        config=dict(
                            llm=dict(
                                provider="openai",
                                config=dict(
                                    model="gpt-4o",
                                    # temperature=0.7, # Can add if needed
                                ),
                            ),
                            embedder=dict(
                                provider="huggingface",
                                config=dict(
                                    model="BAAI/bge-small-en-v1.5",
                                ),
                            ),
                        )
                    )
                st.success("PDF indexed! Ready to chat.")
                st.session_state.crew = None # Reset crew so it can be rebuilt with the new PDF tool

        # Optionally display the PDF in the sidebar
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)
    else:
        # If no file is uploaded, ensure pdf_tool is None and show a message
        if st.session_state.pdf_tool is not None:
            st.session_state.pdf_tool = None
            st.session_state.crew = None
            st.info("Please upload a PDF to enable PDF search.")


    st.button("Clear Chat", on_click=reset_chat)

# ===========================
#   Main Chat Interface
# ===========================
# Ensure 'assets' folder and 'crewai.png' exist relative to your app.py
try:
    with open("assets/crewai.png", "rb") as f:
        crewai_logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        # Agentic RAG powered by <img src="data:image/png;base64,{crewai_logo_base64}" width="120" style="vertical-align: -3px;">
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("# Agentic RAG powered by CrewAI")
    st.warning("`assets/crewai.png` not found. Please create 'assets' folder and place 'crewai.png' inside.")


# Render existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about your PDF or the web...")

if prompt:
    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build or reuse the Crew (crew is built when prompt is entered if not already existing)
    if st.session_state.crew is None:
        # If a PDF was uploaded, use it. Otherwise, pdf_tool will be None and agent will rely on web search.
        st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)
        
    if st.session_state.crew is None:
        st.error("Crew could not be initialized. Please upload a PDF or ensure LLM/tool configurations are correct.")
        st.stop()


    # 3. Get the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get the complete response first
        with st.spinner("Thinking..."):
            inputs = {"query": prompt}
            # Ensure .raw is called as Crew.kickoff() returns CrewOutput object in newer versions
            result_output = st.session_state.crew.kickoff(inputs=inputs)
            if hasattr(result_output, 'raw'):
                result = result_output.raw
            else:
                result = str(result_output) # Fallback if .raw is not available or type changed
            
        # Stream the response
        lines = result.split('\n')
        for i, line in enumerate(lines):
            full_response += line
            if i < len(lines) - 1:  # Don't add newline to the last line
                full_response += '\n'
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.05)  # Adjust the speed as needed for a smoother typing effect
        
        # Show the final response without the cursor
        message_placeholder.markdown(full_response)

    # 4. Save assistant's message to session
    st.session_state.messages.append({"role": "assistant", "content": result})