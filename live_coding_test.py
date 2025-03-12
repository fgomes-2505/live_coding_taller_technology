"""
This script creates a AI Agent that uses RAG techniques for improving its
answers
"""

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from pathlib import Path
import os
from langchain_core.documents import Document



def vector_database():
    """
    This function creates a vector database for using in RAG
    It returns the vector database
    """

    # Reading CSV
    df = pd.read_csv('customer_service_conversations.csv')

    # Creating List
    documents = df['Customer Message'].to_list()

    # HuggingFace Parameters
    model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    # Define Embedding Function
    hf = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    # Creating Index for Vector Database
    index = faiss.IndexFlatL2(
        len(hf.embed_query("hello world"))
    )

    # Creating Vector Database
    vector_store = FAISS(
        embedding_function=hf,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    # Adding Dcuments
    ids = [i for i in range(len(documents))]
    document_1 = Document(page_content=documents[0])
    document_2 = Document(page_content=documents[1])
    document_3 = Document(page_content=documents[2])
    document_4 = Document(page_content=documents[3])
    document_5 = Document(page_content=documents[4])

    documents = [document_1, document_2, document_3, document_4, document_5]
    vector_store.add_documents(documents=documents, ids=ids)
    return vector_store

def qa_chain(model, query, vector_store):
    """
    This function creates a question-answer chain that uses a vector store for
    enhancing the model answer
    """
    # Getting OpenAI API Key
    BASE_DIR = Path(__file__).resolve().parent.parent
    load_dotenv(str(BASE_DIR) + '/')
    api_key = os.getenv("API_KEY")
    os.environ['OPENAI_API_KEY'] = api_key

    # Loading Model
    llm = ChatOpenAI(
        model=model,
        verbose=True
    )

    # Retrieving Information RAG
    retriever = vector_store.as_retriever(fetch_k=3)

    system_prompt = """
    You are someone who answers the client's questions. Answer kindly and
    softly, caring about what the user wants from you.
    Answer the questions considering  the context provided.
    Contexto: {context}
    """

    # Gerando Lista Mensagem Contendo System Prompt e Pergunta Usuario
    messages = [('system', system_prompt)]
    messages.append(('human', '{input}'))
    prompt = ChatPromptTemplate.from_messages(messages)

    # Criando Question Answer Chain
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # Criando Retrieval Chain
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    # Obtendo Resposta
    try:
        response = chain.invoke({'input': query})
        return response
    except Exception as e:
        return e

if __name__ == '__main__':
    question = input('What is you question')
    rag = vector_database()
    answer = qa_chain('gpt-4o-mini', question, rag)
    print(answer)