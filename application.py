import streamlit as st


def main():
    st.set_page_config(page_title="Ask question about yuor PDFs")
    st.header("PDF Document Assistant")
    st.chat_input("Ask question about yuor PDF...")
    with st.sidebar:
        st.subheader("Yuor documents")
        
