import streamlit as st
import requests

# Title of the app
st.title("Resume Management System")

# Section for uploading PDFs
st.header("Upload PDF Resume")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Upload the PDF to FastAPI backend
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    response = requests.post("http://localhost:8000/upload_pdf/", files=files)

    if response.status_code == 200:
        st.success("PDF content successfully uploaded and stored in Qdrant.")
    else:
        st.error("Failed to upload PDF content.")

# Section for searching technology names
st.header("Search for Persons by Technology")

# Text input for technology names
technologies_input = st.text_input("Enter technology names (comma-separated)")

if st.button("Search"):
    if technologies_input:
        technologies = [tech.strip() for tech in technologies_input.split(",")]
        # print(technologies)
        response = requests.post("http://localhost:8000/search/", json={"technologies": technologies})

        if response.status_code == 200:
            result = response.json()
            unique_names = result.get("unique_names", [])
            if unique_names:
                st.write("Persons associated with the technologies:")
                st.write(", ".join(unique_names))
            else:
                st.write("No persons found for the given technologies.")
        else:
            st.error("Failed to retrieve data from the backend.")
    else:
        st.warning("Please enter at least one technology name.")
