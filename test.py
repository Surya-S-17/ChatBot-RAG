from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] ='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'
raw_text = ''
embeddings = HuggingFaceEmbeddings()
#if there is multiple document,execute this block
'''for file in os.listdir('uploads'):
    pdfreader = PdfReader(os.path.join('uploads',file))


    from typing_extensions import Concatenate
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content'''

#if there is only one document,mention the ppath and execute this block
'''pdfreader = PdfReader('path_to_document')


    from typing_extensions import Concatenate
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content'''




text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = HuggingFaceEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)
chain = load_qa_chain(HuggingFaceHub(repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1'), chain_type="stuff")  

def get_Chat_response(chat):
    query = chat
    docs = document_search.similarity_search(query)
    a=chain.run(input_documents=docs, question=query)
    output= (a[a.index('Helpful Answer:')+16:])
    return output
print(get_Chat_response('What is the outcome of this research paper'))