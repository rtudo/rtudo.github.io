from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# We need to set the GOOGLE_API_KEY and GOOGLE_CSE_ID
# environment variables for this to work.
# They are loaded implicitly by the wrapper.

# Initialize the Google Search Wrapper
search_wrapper = GoogleSearchAPIWrapper()


def google_search(query: str):
    return search_wrapper.results(query, num_results=10)

google_search_tool = Tool(
    name="google_search",
    description="A wrapper around Google Search. Use this to search for any information on the internet. Useful for when you need to answer questions about current events or find up-to-date information.",
    func=google_search
)


# Let's test it out!
if __name__ == "__main__":
    print("--- Testing the Google Search Tool ---")
    query = "What were the top 5 tech news stories yesterday?"
    results = google_search_tool.run(query)
    print(f"Results for '{query}':\n")
    for i, item in enumerate(results):
        print(f"Title: {item.get('title', 'N/A')}")
        print(f"Snippet: {item.get('snippet', 'N/A')}\n")