from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from retriever import get_retriever

# Load environment variables
load_dotenv()

# 1. Define the State
# This is the "notebook" the agents pass back and forth
class AgentState(TypedDict):
    question: str
    context: str
    draft_summary: str
    feedback: str
    loop_count: int

# Initialize the Brains
researcher_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
fact_checker_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

# 2. Define the Researcher Node
def researcher(state: AgentState):
    print(f"\n--- 🕵️‍♂️ RESEARCHER (Loop {state['loop_count']}) ---")
    question = state["question"]
    
    # If this is the first loop, fetch the context from your local FAISS database
    if state["loop_count"] == 0:
        print("Fetching context from local PDF memory...")
        retriever = get_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        state["context"] = context
    else:
        context = state["context"]
        print("Using existing context and applying Fact-Checker feedback...")

    # Draft the summary
    if state["loop_count"] == 0:
        prompt = f"""You are a Researcher. Write a comprehensive summary answering the user's question based ONLY on the provided context. 
        Question: {question}
        Context: {context}"""
    else:
        prompt = f"""You are a Researcher. Revise your previous draft based on the Fact-Checker's feedback. 
        Do not add information outside the original context.
        Question: {question}
        Context: {context}
        Draft: {state['draft_summary']}
        Feedback: {state['feedback']}"""

    response = researcher_llm.invoke(prompt)
    state["draft_summary"] = response.content
    print("Draft complete.")
    return state

# 3. Define the Fact-Checker Node
def fact_checker(state: AgentState):
    print("\n--- 📋 FACT-CHECKER ---")
    print("Auditing draft against source text...")
    
    prompt = f"""You are a strict Fact-Checker. Compare the Draft Summary against the Source Context.
    Your ONLY job is to find claims in the draft that are NOT explicitly supported by the context.
    
    Source Context: {state['context']}
    Draft Summary: {state['draft_summary']}
    
    If the draft is perfectly accurate based on the context, reply ONLY with the word: APPROVE
    If there are errors or hallucinations, reply with: REJECT: [Explain exactly what is wrong]"""

    response = fact_checker_llm.invoke(prompt)
    feedback = response.content.strip()
    
    state["feedback"] = feedback
    state["loop_count"] += 1 # Increment the loop counter
    
    if feedback.startswith("APPROVE"):
        print("Verdict: APPROVED ✅")
    else:
        print(f"Verdict: REJECTED ❌ \nFeedback: {feedback}")
        
    return state

# 4. Define the Routing Logic
def route_workflow(state: AgentState):
    # Stop if approved, or if they've argued 3 times (to prevent infinite loops)
    if state["feedback"].startswith("APPROVE") or state["loop_count"] >= 4:
        return END
    else:
        return "researcher"

# 5. Assemble the Graph
print("Compiling Reflective-RAG Graph...")
workflow = StateGraph(AgentState)

# Add our nodes
workflow.add_node("researcher", researcher)
workflow.add_node("fact_checker", fact_checker)

# Set the starting point
workflow.set_entry_point("researcher")

# Connect Researcher to Fact-Checker
workflow.add_edge("researcher", "fact_checker")

# Connect Fact-Checker to the Routing Logic
workflow.add_conditional_edges("fact_checker", route_workflow)

# Compile it into a runnable application
app = workflow.compile()

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Ensure you have already run retriever.py with a sample.pdf to build the DB!
    initial_state = {
        "question": "What is the main topic of this document?",
        "context": "",
        "draft_summary": "",
        "feedback": "",
        "loop_count": 0
    }
    
    print("\n🚀 Starting Reflective-RAG Workflow...")
    # Stream the output so we can watch the agents work
    final_output = None
    for output in app.stream(initial_state):
        final_output = output 
    
    print("\n🎉 --- FINAL APPROVED SUMMARY --- 🎉")
    # The output is a dictionary keyed by the last node that ran (e.g., {"fact_checker": {state}})
    last_node = list(final_output.keys())[0]
    final_draft = final_output[last_node]["draft_summary"]
    
    print(final_draft)
    print("\n--------------------------------------")