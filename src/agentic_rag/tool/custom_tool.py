import os
import requests
from crewai.tools import BaseTool
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from pydantic import PrivateAttr # <-- ADD THIS IMPORT

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "A tool to search for information within a local PDF document."
    file_path: str

    def _run(self, query: str) -> str:
        # Placeholder for actual PDF search logic
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if query.lower() in content.lower():
                return f"Found '{query}' in the PDF. [Actual content extraction needed here]"
            else:
                return "Information not found in the PDF document."
        except Exception as e:
            return f"Error accessing or searching PDF: {e}. Perhaps the PDF content cannot be read as plain text or the file path is incorrect."


class FireCrawlWebSearchTool(BaseTool):
    name: str = "FireCrawlWebSearchTool"
    description: str = "A tool to scrape the content of a specific URL. Input MUST be a valid URL (starting with http:// or https://) provided by a previous web search tool like SerperDevTool."

    _firecrawl: FirecrawlApp = PrivateAttr() # <-- ADD THIS LINE

    def __init__(self, **kwargs):
        super().__init__(**kwargs) # Call super().__init__ first to initialize BaseTool's fields
        load_dotenv()
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        if not firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables.")
        self._firecrawl = FirecrawlApp(api_key=firecrawl_api_key) # <-- ASSIGN TO _firecrawl

    def _run(self, query: str) -> str:
        try:
            if query.startswith(('http://', 'https://')):
                print(f"--- FireCrawl: Attempting to scrape URL: {query} ---")
                # USE _firecrawl here
                result = self._firecrawl.scrape_url(query, {'pageOptions': {'onlyMainContent': True}})
                content = result.get('markdown', 'Could not scrape content from URL.')
                print(f"--- FireCrawl: Scraped content (first 200 chars): {content[:200]} ---")
                return content
            else:
                print(f"--- FireCrawl: Received non-URL input: '{query}' ---")
                return "FireCrawl requires a valid URL. No content scraped."
        except Exception as e:
            print(f"--- FireCrawl: ERROR during web scrape for query '{query}': {e} ---")
            return f"Error during Firecrawl web scrape: {e}"