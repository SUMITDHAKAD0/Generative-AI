import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter, CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceHubEmbeddings
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    # context_entities_recall,
    context_precision,
    context_recall,
    context_relevancy,
    faithfulness
  )
from datasets import Dataset
from ragas import evaluate
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context


class Data_Ingestion:
  def __init__(self, path:str, split_tech='recursive', chunk_size=1000, chunk_overlap=100):
    self.path = path
    self.split_tech = split_tech
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

  def data_load(self):

    if self.path.split('.')[1].lower() == 'pdf':
      print('pdf')
      pdf_loader = PyPDFLoader(self.path)
      text = pdf_loader.load()

    if self.path.split('.')[1].lower() == 'txt':
      print('txt')
      txt_loader = TextLoader(self.path)
      text = txt_loader.load()

    if self.path.split('.')[1].lower() == 'docx':
      print('docs')
      doc_loader = Docx2txtLoader(self.path)
      text = doc_loader.load()

    return text

  def splitter(self, texts):

    if self.split_tech.lower() == 'recursive':
      print('rec')
      text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap)
      documents = text_splitter.split_documents(texts)


    if self.split_tech.lower() == 'character':
      print('char')
      text_splitter = CharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap)
      documents = text_splitter.split_documents(texts)

    return documents


class Embading_DB:
  def __init__(self):
    self.embedding = 'openai'
    self.vector_db = 'chroma'
    self.embd_model = 'sentence-transformers/all-MiniLM-L6-v2'

  def create_embedding(self):
    if self.embedding == 'hugging':
      print('hugging')
      embedding = HuggingFaceHubEmbeddings(model= self.embd_model)

    if self.embedding == 'openai':
      print('openai')
      embedding = OpenAIEmbeddings()

    return embedding

  def create_vectordb(self, document, embedding):
    if self.vector_db == 'chroma':
      print('chroma')
      db = Chroma.from_documents(document, embedding, persist_directory="/content/")

    if self.vector_db == 'faiss':
      print('faiss')
      db = FAISS.from_documents(document, embedding)
      db.save_local("faiss_index")
      print('done')

    return db

class RAG:

  def __init__(self, db, prompt, model_name, chain_type, temprature):
    self.retriever = db.as_retriever()
    self.prompt = prompt
    self.model_name = model_name
    self.chain_type = chain_type
    self.temprature = temprature

  def create_model(self):
    if self.model_name == 'openai':
      print('openai')
      model = ChatOpenAI(temperature = self.temprature)

    if self.model_name == 'llama':
      pass

    return model


  def doc_stuff_chain(self, model):
    if self.chain_type == 'retriver':
      print('doc_stuff_chain')
      document_chain = create_stuff_documents_chain(model, self.prompt)
      retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

    return retrieval_chain, self.retriever

class Evaluation_RAG:
  def __init__(self, test_size):
    self.test_size = test_size

  def single_evatuation(self, query):
    print('inside single evaluation function')

    response = chain.invoke({"input" : query})
    contexts = [context.page_content for context in response["context"]]

    answers = [response['answer']]
    question = [query]
    context = [[contexts[0]]]

    response_dataset = Dataset.from_dict({
          "question" : question,
          "answer" : answers,
          "contexts" : context
      })

    score = evaluate(response_dataset, metrics=[faithfulness, context_relevancy, answer_relevancy])
    return score, response['answer']

  def document_evatuation(self, document, chain):
    print('inside document evaluation function')

    # generator with openai models
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    # generate testset
    testset = generator.generate_with_langchain_docs(document, test_size=self.test_size, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
    eval_dataset = testset.to_pandas()
    test_questions = eval_dataset["question"].values.tolist()
    test_groundtruths = eval_dataset["ground_truth"].values.tolist()

    answers = []
    contexts = []

    for question in test_questions:
      response = chain.invoke({"input" : question})
      answers.append(response["answer"])
      contexts.append([context.page_content for context in response["context"]])

    response_dataset = Dataset.from_dict({
          "question" : test_questions,
          "answer" : answers,
          "contexts" : contexts,
          "ground_truth" : test_groundtruths
      })

    score = evaluate(response_dataset, metrics=[faithfulness, context_relevancy, answer_relevancy, answer_correctness, answer_similarity, context_precision, context_recall])
    return score.to_pandas()



# Function to save uploaded files to corresponding folders based on file type
def save_uploaded_files(uploaded_files, folder_path):
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1]
        file_path = os.path.join(folder_path, file_extension)
        os.makedirs(file_path, exist_ok=True)
        with open(os.path.join(file_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    return file_path  # Return the folder path

# Adding an appropriate title for the test website
st.title("AiEnsured RAG Evaluation")
st.sidebar.markdown("RAG Evaluation")

file_type = st.sidebar.selectbox('Select your file Type?', ['pdf', 'txt', 'word'])
split_tech = st.sidebar.selectbox('Select your file Type?', ['recursive', 'character'])
chunk_size = st.sidebar.number_input('Enter Chunk Size', value=1000)
chunk_overlap = st.sidebar.number_input('Enter Chunk Overlap Size', value=100)
evaluation_type = st.sidebar.selectbox('Evaluation Type', ['single', 'overall_doc'], index=0)



uploaded_files = st.file_uploader("Upload your Files and Click on the Submit & Process Button", accept_multiple_files=False)

if uploaded_files:
    # Save the uploaded files to corresponding folders based on file type
    folder_path = save_uploaded_files([uploaded_files], "uploaded_files")

    # Display confirmation message
    st.write("File has been saved successfully.")
    # st.write(uploaded_files)

    path = ""
    if uploaded_files.name.split(".")[-1] == "pdf":
        path = os.path.join(folder_path, uploaded_files.name)
    elif uploaded_files.name.split(".")[-1] == "docx":
        path = os.path.join(folder_path, uploaded_files.name)
    elif uploaded_files.name.split(".")[-1] == "txt":
        path = os.path.join(folder_path, uploaded_files.name)
    else:
        st.write('Invalid file format')

    # st.write(path)

    # Create an instance of Data_Ingestion and call the appropriate method
    obj = Data_Ingestion(path, split_tech, chunk_size, chunk_overlap)
    text = obj.data_load()
    doc = obj.splitter(text)

    # Create an instance of Embading_DB and call the appropriate method
    emb_obj = Embading_DB()
    emb = emb_obj.create_embedding()
    db = emb_obj.create_vectordb(doc, emb)

    prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context.
            Think step by step before providing a detailed answer.
            I will tip you $1000 if the user finds the answer helpful.
            <context>
            {context}
            </context>
            Question: {input}""")

    rag_obj = RAG(db, prompt, model_name='openai', chain_type='retriver', temprature=0.6)
    model = rag_obj.create_model()
    chain, context = rag_obj.doc_stuff_chain(model)

    ev_obj = Evaluation_RAG(5)
    if evaluation_type == "overall_doc":
        result = ev_obj.document_evatuation(doc, chain)
        st.write("DataFrame:")
        st.dataframe(result)

    if evaluation_type == "single":
        if chain is not None:
          key = 0
          while True:
              query = st.text_input("Ask your Query", key='key' + str(key))

              if len(query) == 0:
                  print('Ask something')
              else:
                  # response = chain.invoke({"input": query})
                  score, answer = ev_obj.single_evatuation(query)
                  st.write('query : ', query, '\nAnswer :', answer, 'Scores : ', score)
                  key +=1

        else:
          print('Chain not found')
