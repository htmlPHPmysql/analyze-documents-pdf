import streamlit as st
import pdfplumber

def get_pdf_text(pdf_document):
    text = ""
    for pdf in pdf_document:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text


def main():
    st.set_page_config(page_title="Ask question about yuor PDFs")
    st.header("PDF Document Assistant")
    st.chat_input("Ask question about yuor PDF...")
    with st.sidebar:
        st.subheader("Yuor documents")
        pdf_documents = st.file_uploader(
            "Upload your documents and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            if not pdf_documents:
                st.error("Upload at least one doc")
                return

            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_documents)
                st.write(raw_text)

                # get the text chunks

                # create vector store

                # create conversation chian
                st.success(f"Documents {len(pdf_documents)} processed successfully")

if __main__ == '__main__':
    main()