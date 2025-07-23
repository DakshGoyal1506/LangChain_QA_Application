import os
import streamlit as st 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv


def load_document(file_path: str):
    print(f"Loading {file_path!r}")
    lower = file_path.lower()
    
    if lower.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif lower.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif lower.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return loader.load()

def load_from_wikipedia(query, lang="en", load_max_docs=2):
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

def chunk_data(data, chunk_size = 256, chunk_overlap = 64):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  
    vector_store = Chroma.from_documents(chunks, embeddings) 
    return vector_store

def ask_and_get_answer(vector_store, q, k=3):

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.invoke(q)
    return answer

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000000 * 0.02

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":

    # load_dotenv(find_dotenv(), override=True)

    # st.image('img.png')
    st.subheader('LLM Question-Answering Application')

    with st.sidebar:
        api_key = st.text_input('OpenAI API Key: ', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        chunk_overlap = int(chunk_size * 0.2)
        k = st.number_input('k', min_value=1, max_value=20, value=2, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')
    
    q = st.text_input('Ask a question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)

            st.text_area('LLM Answer: ', value=answer['result'])

            st.divider()

            if 'history' not in st.session_state:
                st.session_state.history = ''

            value = f'Q: {q} \nA: {answer["result"]}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            st.text_area(label='Chat History', value=h, key='history', height=400)

# run the app: streamlit run ./app.py