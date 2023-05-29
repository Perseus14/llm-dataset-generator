from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb

class ChromaDB:

    docs = []

    chroma_client = chromadb.Client()

    def __init__(self, doc_list, chunk_size=1000) -> None:
        
        self.chunk_size = chunk_size    
        self.collection = self.chroma_client.create_collection(name="dataset")
        doc_id = 0
        for file in doc_list:
            if f'{file}'.endswith('.pdf'):
                doc = PyPDFLoader( f'{file}').load()
            elif f'{file}'.endswith('.txt'):
                doc = TextLoader( f'{file}').load()
            
            self.docs.append({'doc': doc[0], 'id': doc_id})
            doc_id+=1

        doc_splitter = RecursiveCharacterTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap  = 100
            )

        texts = []  # List to store the chunked texts
        ids = []
        metadatas = []
        
        # Split documents into chunks
        for doc in self.docs:
            doc_text = doc['doc'].page_content
            text_chunks = doc_splitter.create_documents([doc_text])
            texts+=[text_chunk.page_content for text_chunk in text_chunks]
            metadatas+=[{'doc_id': doc['id']}]*len(text_chunks)
            
            
        ids = [str(text_id) for text_id,_ in enumerate(texts)]
        assert len(metadatas)==len(ids)==len(texts)
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas)

    def query(self, query):

        results = self.collection.query(
        query_texts=[query],
            n_results=3
        )
        return results['documents'][0]
    
    def count(self):
        return self.collection.count()
    
    def get(self, ids):
        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"]
        )
        return results