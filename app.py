import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAIError # Імпортуємо клас помилки для кращої обробки

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
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(
            texts = text_chunks,
            embedding = embeddings
        )
        return vectorstore
    except OpenAIError as e:
        st.error(f"Помилка при створенні векторної бази даних: {e}")
        return None

def get_conversation_chain( vectorstore ):
    try:
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(
            memory_key = "chat_history",
            return_messages = True,
            input_key = "question"
        )
        retriever = vectorstore.as_retriever(
            search_kwargs = {
                "k": 6  # number of documents to retrieve
            }
        )

        prompt = PromptTemplate.from_template(
            """
            Below are document fragments and chat history.
            Answer the user's question ONLY based on the following document fragments and chat history.

            If the answer is NOT in the documents, respond only with:
            "Unfortunately, I could not find the answer to this question in the provided documents."

            Do not use any knowledge from outside the documents.
            Do not guess. Do not create an answer if you are not certain based on the documents.

            Chat history:
            {chat_history}

            Context from documents:
            {context}

            User's question.:
            {question}

            Answer:
            """
        )
        conversation_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: memory.load_memory_variables( {} )[ "chat_history" ]
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        st.session_state.memory = memory
        return conversation_chain
    except OpenAIError as e:
        st.error(f"Помилка при створенні ланцюжка розмови: {e}")
        return None

def handle_userinput( user_question ):
    if "conversation" not in st.session_state or not st.session_state.conversation:
        st.error( "Please process your documents first." )
        return

    try:
        response = st.session_state.conversation.invoke( user_question )
        st.session_state.memory.save_context(
            {
                "question": user_question
            },
            {
                "output": response
            }
        )
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )
        st.session_state.chat_history.append(
            {
                "role": "ai",
                "content": response
            }
        )
        for message in st.session_state.chat_history:
            with st.chat_message( message[ "role" ] ):
                st.markdown( message[ "content" ] )
    except OpenAIError as e:
        st.error(f"Помилка під час обробки запиту: {e}")

def main():
    load_dotenv()
    st.set_page_config( page_title="Ask question about yuor PDFs" )

    # Initialize session state for conversation
    if "conversation" not in st.session_state:
        st.session_state.convesation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = None

    st.header( "PDF Document Assistant" )
    user_question = st.chat_input( "Ask question about yuor PDF..." )
    if user_question:
        handle_userinput( user_question )

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

                # get the text chunks
                text_chunks = get_text_chunks( raw_text )
                
                # create vector store
                vectorstore = get_vectorstore( text_chunks )
                if vectorstore is None:
                    return

                # create conversation chain
                st.session_state.conversation = get_conversation_chain( vectorstore )
                if st.session_state.conversation is None:
                    return

                st.success( f"Documents {len( pdf_documents )} processed successfully" )
                st.session_state.chat_history = []

if __name__ == '__main__':
    main()