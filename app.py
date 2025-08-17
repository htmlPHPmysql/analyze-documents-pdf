import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS



def get_pdf_text( pdf_document ):
    text = ""
    for pdf in pdf_document:
        with pdfplumber.open( pdf ) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get_text_chunks( text ):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text( text )
    return chunks

def get_vectorstore( text_chunks ):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(
        texts = text_chunks,
        embedding = embeddings
    )
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config( page_title="Ask question about yuor PDFs" )
    st.header( "PDF Document Assistant" )
    st.chat_input( "Ask question about yuor PDF..." )
    with st.sidebar:
        st.subheader( "Your documents" )
        pdf_documents = st.file_uploader(
            "Upload your documents and click on 'Process'", accept_multiple_files=True
        )
        if st.button( "Process" ):
            if not pdf_documents:
                st.error( "Upload at least one doc" )
                return

            with st.spinner( "Processing" ):
                # get pdf text
                raw_text = get_pdf_text( pdf_documents )                
                if not raw_text:
                    st.error( "No text found in the uploaded documents." )
                    return
                else:
                    st.success( "Text extracted successfully!" )
                    # st.write( raw_text )
                
                # get the text chunks
                text_chunks = get_text_chunks( raw_text )
                # st.write( text_chunks )

                # create vector store
                vectorstore = get_vectorstore( text_chunks )

                # create conversation chian
                st.success( f"Documents {len( pdf_documents )} processed successfully" )

if __name__ == '__main__':
    main()