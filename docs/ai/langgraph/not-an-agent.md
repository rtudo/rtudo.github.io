# Not an Agent: Our First Call to Gemini

With our environment set up and our keys safely stored, it's time to write some code!

But hold on. Before we jump into the complexities of graphs, nodes, and edges, let's take one small, crucial step. We're going to write a very simple script to make a direct call to the Gemini model.

Think of this as a **sanity check**. Its only job is to confirm that everything we just set up is working correctly. Can our code find the API key? Can it connect to Google's service? Can it get a response back? Let's find out.

## 1. The "Not-an-Agent" Script

In your project directory (`langgraph-google-tutorial`), create a new file named `not_an_agent.py` and add the following code.

```python title="not_an_agent.py"
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()

# Initialize the ChatGoogleGenerativeAI client
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Invoke the model with a simple prompt
result = llm.invoke("what is the capital of India?")

# Print the content of the response
print(result.content)
```

## 2. Breaking Down the Code

This script is intentionally straightforward. Let's walk through it line by line.

1.  `from dotenv import load_dotenv`: This line imports the function that knows how to find and read our `.env` file.
2.  `from langchain_google_genai import ChatGoogleGenerativeAI`: Here, we import the specific class from the LangChain library that allows us to communicate with Google's Gemini chat models.
3.  `load_dotenv()`: This executes the function, loading our `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` into the environment so our script can use them.  [ we will validating GOOGLE_CSE_ID in later sections ]
4.  `llm = ChatGoogleGenerativeAI(...)`: This is the core of our script. We create an *instance* of the model client. We're telling LangChain, "Hey, get ready to talk to the `gemini-2.0-flash` model."
5.  `result = llm.invoke(...)`: This is the action! The `.invoke()` method sends our prompt ("what is the capital of India?") directly to the Gemini API. It then waits for a response.
6.  `print(result.content)`: The response from the model is an object. We access its main text content using `.content` and print it to our terminal.

## 3. Running the Script

Now, let's run it from our terminal. Make sure your virtual environment is still active (you should see `(venv)` in your prompt).

```bash
python not_an_agent.py
```

If everything is set up correctly, you should see a response printed to your console, something like this:

```text
The capital of India is **New Delhi**.
```

Awesome! A simple request and a simple answer. This confirms our API key is valid and our local environment is ready to go.

## 4. What Just Happened (and Why It's Not an Agent)

So, did we just build an agent? **No, not even close.**

What we did was make a single, direct API call. Our program had no ability to reason, no tools to use, and no capacity to take multiple steps. It simply passed a question to the LLM and printed the answer.

This is the fundamental building block. An **agent**, which we'll build next, is much more sophisticated. An agent can loop, make decisions, and use tools to enrich its responses.

Here's a visual comparison:

!!! info inline "Our Current Script"
    <div class="scaled-mermaid">
    ```mermaid
    graph TD
        A[User Prompt] --> B[Gemini LLM];
        B --> C[Final Answer];
    ```
    </div>

!!! info inline end "A True Agent (Coming Next)"
    <div class="scaled-mermaid">
    ```mermaid
    graph TD
        A[User Question] --> B{Agent Decides};
        B -- "Need Info" --> C[Tool: Search];
        C -- "Tool Result" --> B;
        B -- "Have Enough Info" --> D[Generate Final Answer];
    ```
    </div>

By successfully running this simple script, we've validated our setup and built the first block. Now we're ready for the exciting part: giving our program a brain with LangGraph.