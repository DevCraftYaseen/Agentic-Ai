# Imports -----------------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# ------------------------------------------------------------------------------------------------

# Load ENV
load_dotenv()

# Create a Model
gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Page Header
st.header("Research Tool")

# Input Variables --------------------------------------------------------------------------------

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need",
"BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical",
"Code-Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# ------------------------------------------------------------------------------------------------

# Prompt Template -------------------------------------------------------------------------------

template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1. Mathematical Details:
        - Include relevant mathematical equations if present in the paper.
        Explain the mathematical concepts using simple, intuitive code snippets where
        applicable
    2. Analogies:
        - Use relatable analogies to simplify complex ideas.
        If certain information is not available in the paper, respond with: "Insufficient
        information available" instead of guessing.
        Ensure the summary is clear, accurate, and aligned with the provided style and length.
    """,
    input_variables=['paper_input', 'style_input', 'length_input']
)

# ------------------------------------------------------------------------------------------------

# Prompt Placeholders filling -------------------------------------------------------------------
if not paper_input and style_input and length_input:
        st.warning("Please select all the required fields for summarization.")
else:
   prompt = template.invoke(
       {     
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
       }
   )

# ------------------------------------------------------------------------------------------------

# Action Button and Showcasing results ---------------------------------------------------------

if st.button("Summarize"):
    if not prompt:
        st.warning("Please enter a prompt before summarizing.")
    else:
        result = gemini.invoke(prompt)
        st.write(result.content)
        
# ------------------------------------------------------------------------------------------------
