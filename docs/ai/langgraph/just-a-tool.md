# The Google Search Tool

## 1. Why would a LLM need to search anything ?

Large Language Models like Gemini are incredibly powerful. They have been trained on a vast amount of text and can reason, write, and summarize like nothing we've ever seen. They have one fundamental limitation: **they are stuck in time.**

!!! note
    An LLM's knowledge is frozen at the point its training data was collected. It has no access to events that happened yesterday, real-time stock prices, or the current weather. Think of it as a brilliant historian locked in a library filled with books that are all a year old. They can tell you everything about the past, but they can't tell you today's news.

This is where a Search **Tool** come in.

## 2. But what is a tool ?

A tool is simply a function that the LLM can decide to call to get information or perform an action in the "real world." By giving our agent a `Google Search` tool, we are giving it a window to the live, up-to-the-minute internet.



An agent can be equipped with any number of tools to interact with the world. Few examples : 

<div class="grid" markdown>

<span class="hover-trigger">Query a Database<span class="hover-box">Connect to a SQL database to answer questions about company sales data.</span></span>
{ .card }

<span class="hover-trigger">Access API<span class="hover-box">Use a weather API or a Financial API for stocks or Connect to other Agents or access any Product via the API</span></span>
{ .card }

<span class="hover-trigger">Execute Code<span class="hover-box">Run Python code in a sandboxed environment to perform calculations or data analysis.</span></span>
{ .card }

<span class="hover-trigger">Possibilities are endless<span class="hover-box">think what tool can aid to LLM capability and make life / goal easy for LLM</span></span>
{ .card }
</div>


## 3. Building the Google Search Tool

Let's write some code. We'll create a simple Python script that defines our Google Search tool using the libraries we installed earlier.

Create a new file in your project directory called `search_tool.py` and add the following code.

```python title="search_tool.py"
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
```

## 4. Breaking Down the Code


1.  **`load_dotenv()`**: This function is our starting point. It reads the `.env` file we created earlier and securely loads our `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` into the environment, making them available to the LangChain wrappers without hardcoding them.

2.  **`GoogleSearchAPIWrapper()`**: We initialize the wrapper class from `langchain-google-community`. This class handles all the backend complexity of making authenticated requests to the Google Custom Search API.

3.  **`def google_search(query: str):`**: This is the most important change in our new script! Instead of using the default `.run` method, we've created our own custom function called `google_search`.
    *   This function takes a `query` string as input.
    *   It calls `search_wrapper.results(query, num_results=10)`. Unlike the `.run()` method which returns a simple string, the `.results()` method returns a **list of dictionaries**. Each dictionary contains structured data about a search result, like its `title`, `link`, and `snippet`.
    *   This approach gives us much more control over the data format and allows us to specify parameters like `num_results=10` on the fly.

4.  **`Tool(...)`**: We define our `Tool` object as before, but with one key difference:
    *   `name`: `google_search`, a simple name for the agent to use.
    *   `description`: A clear explanation of what the tool does and when to use it. This is crucial for the agent's reasoning process.
    *   `func=google_search`: We now pass our **custom function** to the tool. When the agent decides to use `google_search`, it will execute our function, which in turn calls the API and returns the structured list of results.


5.  **`if __name__ == "__main__":`**: Our test block has also been updated to reflect the new output format.
    *   It calls `google_search_tool.run(query)`, which executes our custom function.
    *   Because our function returns a list of dictionaries, we can now iterate through the `results` and cleanly print the `title` and `snippet` for each item. This demonstrates the power of getting structured data back from our tools.

!!! info "A Reusable Building Block"
    One of the best parts about structuring our code this way is that `search_tool.py` now acts as a self-contained module. The `google_search_tool` object we created is not trapped inside this file; we can import it into any other part of our application.

    For example, in our main agent file (which we'll create soon!), we can simply write:
    ```python
    from search_tool import google_search_tool

    # Now we can use the tool in our agent graph!
    tools = [google_search_tool]
    ```
    This approach keeps our project organized and scalable. Our tools are defined in one place, and our agent's logic is defined in another.

## 5. Running the Tool

If you run the script directly from your terminal, the `if __name__ == "__main__:"` block will execute.

```bash
python search_tool.py
```

You should see an output similar to this (the news will, of course, be different!):

```text
--- Testing the Google Search Tool ---
Results for 'What were the top 5 tech news stories yesterday?':

Title: Tech News | Today's Latest Technology News | Reuters
Snippet: A European climate research institute is using artificial intelligence (AI) to more accurately predict the probability of wildfires across the globe. Gabe ...

Title: Google News
Snippet: News · News · Sign in. For you. Headlines Local U.S.. World Business Technology ... Top stories. Los Angeles Times. Camp Pendleton Marines deployed to L.A.; after ...

Title: Top news of the week - Science News, Physics News, Technology ...
Snippet: Top news stories for the week 23. Sunday, Jun 08. Apple under pressure to ... Tech Xplore is a part of Science X network. With global reach of over 5 ...
```
Success! We've created a standalone, functional tool that can fetch real-time information from the internet.



---

Right now, our tool is just a Python object in a script. It's not yet part of an agent. In the next section, we'll take this tool and integrate it into our very first **LangGraph State Graph**. This is where the magic really begins.