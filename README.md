# E-Commerce FAQ Bot

A retrieval-augmented generation (RAG) assistant designed to automate customer support for online retailers. The system handles queries related to product catalogues, return policies, and shipping logistics using an agentic workflow built with LangGraph.

## Core Functionality

The assistant uses a multi-node graph to process user inquiries:

*   **Intelligent Routing**: Automatically determines if a query requires knowledge base retrieval, a calculation tool, or a response from conversation history.
*   **Contextual Retrieval**: Utilizes a local vector store (NumPy-based) to find relevant policies or product details.
*   **Dynamic Timeline Tool**: Calculates estimated delivery dates and return windows based on the current date.
*   **Faithfulness Evaluation**: Includes a self-correction loop that scores responses based on how well they adhere to the retrieved context, preventing hallucinations.

## Tech Stack

*   **LLM**: Llama 3.3 70B (via Groq)
*   **Frameworks**: LangGraph, LangChain
*   **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
*   **Frontend**: Streamlit
*   **Database**: In-memory NumPy vector collection

## Project Structure

*   `agent.py`: The core agentic logic and graph definition.
*   `capstone_streamlit.py`: The Streamlit web interface and session management.
*   `day13_capstone.ipynb`: Development notebook for testing components and logic.
*   `.env`: Environment configuration for API keys.

## Setup Instructions

### Prerequisites

*   Python 3.10+
*   Groq API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Satyam5167/Capstone-ECOMMERCE_POLICY_BOT.git
   cd Capstone-ECOMMERCE_POLICY_BOT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Create a `.env` file in the root directory and add your Groq API key:
   ```text
   GROQ_API_KEY=your_api_key_here
   ```

### Running the Application

To start the Streamlit interface:
```bash
streamlit run capstone_streamlit.py
```

## How it Works

The system implements a state machine where the user's question passes through a series of nodes:
1. **Memory**: Prepares conversation history.
2. **Router**: Decides the next step (Retrieve, Tool, or Skip).
3. **Retrieval**: Fetches relevant snippets from the knowledge base.
4. **Tool**: Generates real-time data (e.g., shipping estimates).
5. **Answer**: Synthesizes a response using the provided context.
6. **Evaluation**: Checks the answer for faithfulness. If the score is too low, it triggers a retry.
