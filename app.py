import streamlit as st
import os
import tempfile
from retriever import build_vector_db
from graph import app as agent_workflow

# Page Configuration
st.set_page_config(page_title="Reflective-RAG Assistant", page_icon="🧠", layout="centered")

st.title("🧠 Reflective-RAG Research Assistant")
st.markdown("Upload a document and let the **Researcher (Gemini)** and **Fact-Checker (Llama 3)** debate to find the perfect answer.")

# Sidebar for Document Upload
with st.sidebar:
    st.header("1. Upload Memory")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing document and building local memory..."):
            # Save the uploaded file temporarily so our retriever can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Build the FAISS database
            build_vector_db(tmp_file_path)
            
            # Clean up the temporary file
            os.remove(tmp_file_path)
            
        st.success("✅ Database built! Agents are ready.")

# Main Interface for Questions
st.header("2. Ask the Agents")
question = st.text_input("What would you like to know about this document?")

if st.button("Start Research Loop") and question:
    if not uploaded_file:
        st.error("Please upload a PDF in the sidebar first!")
    else:
        # Set up the initial state for LangGraph
        initial_state = {
            "question": question,
            "context": "",
            "draft_summary": "",
            "feedback": "",
            "loop_count": 0
        }
        
        # Create a placeholder in the UI for a loading spinner
        with st.status("Agents are analyzing the text...", expanded=True) as status:
            st.write("🕵️‍♂️ Researcher is reading the vector database...")
            
            # Run the LangGraph workflow
            final_output = None
            for output in agent_workflow.stream(initial_state):
                # We can print updates to the UI based on which node just ran
                node_name = list(output.keys())[0]
                if node_name == "researcher":
                    st.write("🕵️‍♂️ Researcher drafted a summary. Sending to Fact-Checker...")
                elif node_name == "fact_checker":
                    st.write("📋 Fact-Checker is auditing the draft...")
                
                final_output = output
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
        
        # Extract and display the final approved summary
        last_node = list(final_output.keys())[0]
        final_draft = final_output[last_node]["draft_summary"]
        
        st.subheader("🎉 Final Approved Summary")
        st.write(final_draft)