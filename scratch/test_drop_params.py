import os
import asyncio
from crewai import Agent, Crew, Task, LLM, Process
from crewai.tools import tool

@tool("Dummy Tool")
def dummy_tool(param: str) -> str:
    """A dummy tool description."""
    return f"Processed {param}"

try:
    local_llm = LLM(
        model="ollama/axis-llama",
        base_url="http://localhost:11434/v1"
    )
    # Force prompt-based tool calling
    local_llm.supports_function_calling = lambda: False
except Exception as e:
    print("LLM creation error:", e)
    local_llm = None

if local_llm:
    agent = Agent(
        role="Tester",
        goal="Test tool integration",
        backstory="You are testing tools.",
        llm=local_llm,
        tools=[dummy_tool],
        verbose=True
    )
    
    task = Task(
        description="Run dummy tool with input 'hello' and return the result.",
        expected_output="The dummy tool response.",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        res = crew.kickoff()
        print("Success:", res)
    except Exception as e:
        print("Crew kickoff failed:", e)
