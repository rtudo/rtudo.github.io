# Still Not an Agent: Building an Interactive Chatbot

Okay, our `not_an_agent.py` script worked perfectly. We sent a question to Gemini and got a correct answer back. Mission accomplished!

But it was a bit... static. The script ran once and then immediately exited. What if we want to have an ongoing conversation? Let's take the next logical step and turn our one-shot script into a simple, interactive chatbot. This will feel much more advanced, but as we'll see, it's still missing the secret ingredient that makes an agent truly "agentic."

## 1. The "Chatbot" Script

In your project directory, create a new file named `chatbot.py`. This script will build directly on our previous one, but this time we'll wrap our logic in a `while` loop.

```python title="chatbot.py"
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()

# Initialize the ChatGoogleGenerativeAI client
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chatbot():
    """
    A simple, interactive chatbot function that uses Gemini.
    """
    print("🤖 Chatbot is ready! Type 'exit' or 'quit' to end the conversation.")

    while True:
        # Get user input from the command line
        user_input = input("You: ")

        # Check if the user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye! 👋")
            break

        # If not exiting, invoke the model with the user's input
        result = llm.invoke(user_input)

        # Print the model's response
        print(f"Gemini: {result.content}")

# Run the chatbot if the script is executed directly
if __name__ == "__main__":
    chatbot()
```

## 2. Breaking Down the Code

Most of this code is familiar, but the new parts are what give it its conversational power.

1.  **`def chatbot():`**: We've wrapped our logic in a function. This is good practice and makes our code more organized and reusable.
2.  **`while True:`**: This creates an infinite loop. The code inside this block will run over and over again until we explicitly tell it to `break`. This is the engine of our continuous conversation.
3.  **`user_input = input("You: ")`**: This is a standard Python function that pauses the script and waits for the user to type something into the terminal and press Enter. Whatever the user types is stored in the `user_input` variable.
4.  **`if user_input.lower() in ["exit", "quit"]:`**: We need a way to escape our infinite loop! This line checks if the user's input (converted to lowercase) is either "exit" or "quit". If it is, the `break` statement is executed, and the `while` loop terminates.
5.  **`llm.invoke(user_input)`**: This is the same `.invoke()` call as before, but now, instead of a hardcoded question, we're passing the live input we just received from the user.

## 3. Running Our Chatbot

Let's see it in action. In your terminal (with the virtual environment active), run the new script:

```bash
python chatbot.py
```

Your terminal will come to life! You can now have a back-and-forth conversation with Gemini.

```text
🤖 Chatbot is ready! Type 'exit' or 'quit' to end the conversation.
You: what is the capital of India?
Gemini: The capital of India is New Delhi.
You: what is it famous for?
Gemini: New Delhi is famous for its rich history, impressive architecture, and as the political hub of India... (and so on)
You: exit
Exiting chatbot. Goodbye! 👋
```

## 4. The Crucial Distinction: Chatbot vs. Agent

This feels a lot more powerful, right? We have an interactive conversation! So... is *this* an agent?

The answer is still a firm **no**. This is a classic chatbot, and the distinction is vital to understanding the power of LangGraph.

!!! danger "The Illusion of Agency"
    A chatbot creates the illusion of agency because it's conversational. However, it's just a simple loop: **Input -> Process -> Output**. The program itself makes no decisions. It's a passive messenger that relays your question to the LLM and prints the answer.

An **agent**, on the other hand, is an active decision-maker. It has a goal and can choose from a set of tools to achieve that goal.

Let's visualize the difference.

!!! info inline "Our Chatbot's Flow"
    Our chatbot follows a simple, repetitive loop. It has no ability to deviate from this path.
    ```mermaid
    graph TD
        A(Start) --> B[Get User Input];
        B --> C{LLM Invocation};
        C --> D[Print LLM Output];
        D --> B;
    ```

!!! info inline end "A True Agent's Flow"
    An agent has branching logic. It can **decide** whether to call a tool or talk to the LLM. This decision-making is the core of what makes it an agent.
    ```mermaid
    graph TD
        A[User Question] --> B{Agent Decides};
        B -- "Need Info" --> C[Tool];
        C -- "Tool Result" --> B;
        B -- "Have Enough Info" --> D[Generate Final Answer];
    ```

The key takeaway is this: Our chatbot is a linear "pass-through." An agent is a cyclical system with a reasoning loop that can plan and execute a series of steps.

By building and understanding this chatbot, we've clearly defined what an agent *is not*. This sets the stage perfectly for our next step, where we will finally use LangGraph to build that reasoning loop and create our first true agent.