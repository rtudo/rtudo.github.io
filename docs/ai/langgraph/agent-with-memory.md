# Giving Your Agent Long-Term Memory

Our agent is efficient, but it's still... a bit forgetful. Every time we interact with it, it's a completely new, isolated execution. The agent has no memory of what we talked about just a moment ago. For a single task, that's fine. But for a real chatbot, that's a deal-breaker.

Note that here we are not talking about a single execution where as per previous code, langgraph state with all previous prompts is updated and passed for all subsequent inferences but only for that single execution.

Let's fix that and give our agent a proper memory.

## Introducing Checkpointers

In LangGraph, the mechanism for giving a graph memory across multiple runs is called a **checkpointer**.

!!! info "What is a Checkpointer?"
    A checkpointer is a component that automatically **saves** the state of your graph after an execution and **loads** it back in before the next one. It's the key to building conversational applications.

The simplest kind of checkpointer is the `MemorySaver`. It holds the conversation state in your computer's RAM.

### How `MemorySaver` Works

When you attach a `MemorySaver` to your graph, the `invoke`/`stream` process changes slightly:

1.  You provide a unique ID for your conversation (called a `thread_id`).
2.  LangGraph asks the `MemorySaver`: "Do you have any saved history for `thread_id`?"
3.  If yes, it loads that state and starts the graph execution from there.
4.  The graph runs as usual, adding new messages to the state.
5.  At the end of the run, LangGraph tells the `MemorySaver`: "Here is the final, updated state for `thread_id`. Please save it."

This simple save/load cycle ensures that context is passed between turns, creating a continuous conversation.

### In-Memory vs. Persistent Memory

<div class="grid cards" markdown>
!!! tip "In-Memory: `MemorySaver`"
    `MemorySaver` stores conversation history in a variable inside your running Python script. It's perfect for learning and for applications where memory only needs to last for a single session. If you restart your script, the memory is **gone**.

!!! success "Persistent: `SqliteSaver`, `PostgresSaver`"
    For production applications, you'd use a persistent checkpointer like `SqliteSaver`. It saves the state to a database file (or a full-fledged database like Postgres). This means you can stop your application, restart it days later, and the agent will remember the conversation perfectly.
</div>

## Implementing `MemorySaver`

Adding memory to our streamlined agent is surprisingly simple. We only need to change how we **compile** and **call** the graph. The agent's elegant logic, using the pre-built `ToolNode` and `tools_condition`, remains exactly the same.

Here is the complete code. Notice the two key additions: the `MemorySaver` and the `config` dictionary used when calling the graph.

```python title="smarter_agent_with_memory.py" hl_lines="13 53 57"
import os
import json
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver # 👈 Import the checkpointer

# --- Standard Setup ---
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
search = GoogleSearchAPIWrapper()
google_search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)
tools = [google_search_tool]
llm_with_tools = llm.bind_tools(tools)

# --- Node Definitions & Graph Assembly (Using Pre-built) ---
def chatbot(state: State):
    print("---CHATBOT---")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")


# --- Adding Memory to the Graph ---

# 1. Initialize the checkpointer
memory = MemorySaver() # 👈 Initialize the checkpointer

# 2. Compile the graph with the checkpointer
#    This is the only change to our graph definition!
graph = graph_builder.compile(checkpointer=memory) # 👈 Compile graph with the checkpointer

# 3. Define a unique ID for the conversation thread
#    This ID tells the checkpointer which conversation history to use.
config = {"configurable": {"thread_id": "user-123"}}


# --- Let's test the memory! ---

print("--- First interaction ---")
# Input must be a list of messages. We use HumanMessage.
user_input_1 = "Hi, my name is Alex."
for event in graph.stream({"messages": [("user", user_input_1)]}, config=config):
    # The event stream is the same as before
    for key, value in event.items():
        if key == "chatbot" and not value["messages"][-1].tool_calls:
            print(f"Assistant: {value['messages'][-1].content}")

print("\n" + "="*50 + "\n")

print("--- Second interaction ---")
# The agent should now remember the name "Alex" because we are using the same `config`
user_input_2 = "What did I say my name was?"
for event in graph.stream({"messages": [("user", user_input_2)]}, config=config):
    for key, value in event.items():
        if key == "chatbot" and not value["messages"][-1].tool_calls:
            print(f"Assistant: {value['messages'][-1].content}")

```

## Running the Code and Seeing the Result

When you run this script, you'll see the magic of the checkpointer in action.

```text
--- First interaction ---
---CHATBOT---
User: Hi, my name is Alex
Assistant: Hello Alex! It's nice to meet you. How can I help you today?

==================================================

--- Second interaction ---
---CHATBOT---
User: What did I say my name was?
Assistant: You told me your name is Alex.
```

Success! In the second interaction, the agent correctly recalled the name from the first. It didn't need to search or guess; the information was present in the conversation state that was automatically loaded by the `MemorySaver`. The only thing connecting the two separate `.stream()` calls was the `config` dictionary pointing to the same conversation thread.

And just like that, our smart and efficient agent is no longer forgetful! It's now a true conversationalist, ready to build upon past interactions.

!!! note "without MemorySaver"
    Without Checkpointer, agent won't know the previous conversations across the two different graph runs as in the example. Go ahead and give it a try by 
    updating agent code like below where we can compile the graph without the checkpointer
    ```
    #graph = graph_builder.compile(checkpointer=memory)
    graph = graph_builder.compile()
    ```