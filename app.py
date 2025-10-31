import streamlit as st
from core.rag_pipeline import ServiceDeskRAG

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gen AI Service Desk",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- 2. LOAD RAG PIPELINE ---
# We use st.cache_resource to load the RAG model only ONCE.
# This is a critical performance optimization.
@st.cache_resource
def load_rag_pipeline():
    """Loads the RAG pipeline into Streamlit's cache."""
    print("Loading RAG pipeline for Streamlit...")
    try:
        pipeline = ServiceDeskRAG()
        return pipeline
    except Exception as e:
        # If the API key is missing, ServiceDeskRAG will raise an error
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.stop() # Stop the app if the pipeline fails to load

rag_pipeline = load_rag_pipeline()
if rag_pipeline is None:
    # If pipeline is None, the error is already shown.
    st.stop()


# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🤖  Gen AI Service Desk")
    st.markdown(
        "This chatbot is a demo of a **RAG (Retrieval-Augmented Generation)** system. "
        "It answers questions based *only* on a trusted knowledge base, "
        "preventing hallucinations and ensuring data privacy."
    )
    st.markdown("---")
    st.markdown("### 💡 Ask me about:")
    st.info(
        "✓ VPN Troubleshooting\n"
        "✓ Password Resets\n"
        "✓ Setting up a Printer"
    )
    st.markdown("---")
    st.markdown("Built by **Diyag Gijo**")

# --- 4. CHAT INTERFACE ---
st.title("🤖 Gen AI Service Desk")
st.caption("I am an AI assistant powered by Google Gemini and a local vector store.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you with your IT questions today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("My VPN won't connect..."):
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the "brain" (RAG pipeline)
            response, sources = rag_pipeline.query(prompt)
            
            # Display the answer
            st.markdown(response)
            
            # **THE STANDOUT FEATURE**: Display the sources
            if sources:
                st.markdown("---")
                st.subheader("Sources Used:")
                for source_file in set(sources): # Use set() to avoid duplicates
                    st.caption(f"Source: `{source_file}`")
    
    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

