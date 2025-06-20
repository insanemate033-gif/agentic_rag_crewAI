# crew.py
import os # Make sure os is imported at the top
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
# from crewai_tools import PDFSearchTool # This can be removed if you're using your custom DocumentSearchTool
from src.agentic_rag.tool.custom_tool import DocumentSearchTool
from src.agentic_rag.tool.custom_tool import FireCrawlWebSearchTool # Make sure this is imported

# Initialize the tool with a specific PDF path for exclusive search within that document
# The path should be relative to your project root (AGENTIC_RAG)
# Given crew.py is inside src/agentic_rag/, calculate the path robustly:
current_dir = os.path.dirname(os.path.abspath(__file__)) # This is /path/to/AGENTIC_RAG/src/agentic_rag/
pdf_file_path = os.path.join(current_dir, "..", "..", "knowledge", "dspy.pdf")

# Now, initialize your custom DocumentSearchTool correctly
pdf_tool = DocumentSearchTool(file_path=pdf_file_path)

# Initialize SerperDevTool for general web search queries
web_search_tool_serper = SerperDevTool()

# Initialize your custom FireCrawlWebSearchTool for URL scraping
firecrawl_web_search_tool = FireCrawlWebSearchTool()

@CrewBase
class AgenticRag():
    """AgenticRag crew"""

    agents_config = 'config/agents.yaml' # This should point to your agents.yaml
    tasks_config = 'config/tasks.yaml'   # This should point to your tasks.yaml

    @agent
    def retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['retriever_agent'],
            verbose=True,
            tools=[
                pdf_tool,
                web_search_tool_serper, # For general web search (e.g., "Australian Open dates")
                firecrawl_web_search_tool # For scraping specific URLs
            ],
            allow_delegation=False, # Set to False to prevent unintended agent-to-agent loops
            max_iterations=20 # Increase iteration limit for more complex tasks, or to allow recovery
        )

    @agent
    def response_synthesizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['response_synthesizer_agent'],
            verbose=True,
            allow_delegation=False, # Set to False
            max_iterations=10 # Set a reasonable limit
        )

    @task
    def retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieval_task'],
        )

    @task
    def response_task(self) -> Task:
        return Task(
            config=self.tasks_config['response_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AgenticRag crew"""
        # Ensure LLM is configured globally or passed here if needed.
        # CrewAI often picks up LLM config from environment variables (MODEL, OPENAI_API_BASE, etc.)
        # If your LLM configuration is in demo_llama3.2.ipynb and not in crew.py,
        # you might need to ensure it's available as an environment variable or passed.
        # Assuming you've set the env vars like OPENAI_API_BASE and OPENAI_API_KEY (with Groq details).

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=2, # Increased verbosity for detailed logging
        )