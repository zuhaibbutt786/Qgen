import streamlit as st
import pdfplumber
from transformers import pipeline

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def generate_questions(text, generator):
    """Generate questions from text using a Transformer pipeline."""
    questions = []
    # Split text into smaller segments to avoid exceeding model limits
    segments = text.split("\n")
    for segment in segments:
        if segment.strip():
            try:
                result = generator(f"answer: {segment}")
                questions.append(result[0]['generated_text'])
            except Exception as e:
                st.warning(f"Error generating question for segment: {segment[:50]}... {e}")
    return questions

# Load the model pipeline (cached for efficiency)
@st.cache_resource
def load_generator():
    return pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap")

def main():
    st.title("PDF to Question Generator")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        st.info("Extracting text from the uploaded PDF...")
        pdf_text = extract_text_from_pdf(uploaded_file)

        if pdf_text:
            st.text_area("Extracted Text", pdf_text, height=300)

            if st.button("Generate Questions"):
                st.info("Generating questions. This may take some time...")
                generator = load_generator()
                questions = generate_questions(pdf_text, generator)

                st.subheader("Generated Questions")
                for i, question in enumerate(questions):
                    st.markdown(f"**Q{i+1}:** {question}")
        else:
            st.warning("No text could be extracted from the PDF.")

if __name__ == "__main__":
    main()
