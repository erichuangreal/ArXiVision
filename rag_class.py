import numpy as np
import hashlib
import json
import re
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from pathlib import Path

EMBEDDING_MODEL = "text-embedding-3-small"


class RAGClass:
    def __init__(self, data_path, persist_directory="chroma_store") :
        self.data_path = Path(data_path)
        self.persist_directory = Path(persist_directory)
        self.chunk_size = None
        self.chunk_overlap = None
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.embeddings = None
        self.result = None
        
        self.paper_metadata = {}
        
    def load_documents(self):
        self.documents = []
        self.paper_metadata = {}
        metadata_files = list(self.data_path.rglob("metadata.json"))
        
        if not metadata_files:
            raise FileNotFoundError(
            f"No metadata.json files found under: {self.data_path.resolve()}"
            )
        
        for metadata_path in metadata_files:
            batch_folder = metadata_path.parent
            # Load metadata for this batch
            with open(metadata_path, "r", encoding="utf-8") as f:
                batch_metadata = json.load(f)
            metadata_lookup = {}

            for metadata in batch_metadata:
                paper_id = Path(metadata["local_pdf_path"]).stem
                metadata_lookup[paper_id] = metadata
            for txt_file in batch_folder.glob("*.txt"):
                paper_id = txt_file.stem
                # Store full metadata ONCE, will access after relevant chunks are found
                if paper_id in metadata_lookup:
                    self.paper_metadata[paper_id] = metadata_lookup[paper_id]
                text = txt_file.read_text(encoding="utf-8")
                
                # Splitting and labelling pg numbers
                parts = re.split(r"--- PAGE (\d+) ---", text)
                
                for i in range(1, len(parts), 2):
                    page_number = int(parts[i])
                    page_text = parts[i + 1].strip()
                    if not page_text:
                        continue

                    document = Document(
                        page_content=page_text,
                        metadata={
                            "paper_id": paper_id,
                            "page_number": page_number
                        }
                    )

                    self.documents.append(document)
        print(f"Loaded {len(self.documents)} documents.")
        print(f"Loaded metadata for {len(self.paper_metadata)} papers.")

        return self.documents
    
    def split_documents(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split documents into {len(self.text_chunks)} chunks.")

    def fingerprint(self):
        # Identifies the corpus, chunking, and model behind a store, so a stale
        # one is rebuilt rather than silently serving outdated chunks.
        chunk_digests = sorted(
            hashlib.sha256(
                (chunk.page_content + repr(sorted(chunk.metadata.items()))).encode("utf-8")
            ).hexdigest()
            for chunk in self.text_chunks
        )
        header = f"{EMBEDDING_MODEL}|{self.chunk_size}|{self.chunk_overlap}|{len(chunk_digests)}"
        return hashlib.sha256(
            (header + "".join(chunk_digests)).encode("utf-8")
        ).hexdigest()

    def load_vectorstore(self, fingerprint_path, fingerprint):
        # Returns a stored vectorstore only if it matches and is non-empty.
        if not fingerprint_path.exists():
            return None
        if fingerprint_path.read_text(encoding="utf-8").strip() != fingerprint:
            print("Corpus, chunking, or embedding model changed. Rebuilding.")
            return None

        vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings
        )
        if not vectorstore.get(limit=1)["ids"]:
            print("Stored vectorstore is empty. Rebuilding.")
            return None
        return vectorstore

    def create_vectorstore(self, rebuild=False):
        if not self.text_chunks:
            raise ValueError("No chunks to embed. Call split_documents() first.")

        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        fingerprint = self.fingerprint()
        fingerprint_path = self.persist_directory / "fingerprint.txt"

        if not rebuild:
            self.vectorstore = self.load_vectorstore(fingerprint_path, fingerprint)
            if self.vectorstore is not None:
                print(f"Loaded vectorstore from {self.persist_directory} (no re-embedding).")
                return self.vectorstore

        # Drop the collection rather than the files: Chroma caches an open client
        # per directory, and deleting the database under it turns it readonly.
        # Cleared first so a crash mid-build cannot leave a matching fingerprint.
        fingerprint_path.unlink(missing_ok=True)
        if (self.persist_directory / "chroma.sqlite3").exists():
            Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            ).delete_collection()
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        print(f"Embedding {len(self.text_chunks)} chunks into {self.persist_directory}...")
        self.vectorstore = Chroma.from_documents(
            self.text_chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        fingerprint_path.write_text(fingerprint, encoding="utf-8")
        print("Vectorstore created with embeddings.")
        return self.vectorstore

    def setup_retriever(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        def retrieve_with_metadata(inputs):
            query = inputs["input"]

            docs = base_retriever.invoke(query)

            for doc in docs:
                paper_id = doc.metadata["paper_id"]

                metadata = self.paper_metadata.get(
                    paper_id,
                    {}
                )

                doc.metadata["title"] = metadata.get(
                    "title",
                    "Unknown"
                )

                doc.metadata["arxiv_id"] = metadata.get(
                    "arxiv_id",
                    "Unknown"
                )

                authors = metadata.get("authors", [])

                doc.metadata["authors"] = ", ".join(authors)

            return docs
        self.retriever = RunnableLambda(retrieve_with_metadata)
        print("Retriever set up from vectorstore.")
        print("Retriever details:", base_retriever)
        return self.retriever
    
    def setup_qa_chain(self):
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")
        llm = ChatOpenAI(
            model="gpt-5-nano"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Answer the user's question using ONLY the retrieved context.

                Do not use prior knowledge.

                For every factual claim:
                - It must be supported by the retrieved context.
                - Cite the corresponding source with page numbers.

                If a claim cannot be supported by the retrieved context,
                do not include it.

                If the retrieved context is insufficient to answer the question,
                say that the retrieved papers do not contain enough information.

                Never invent sources, citations, authors, page numbers, or results.
                
                Retrieved context:
                {context}
                """
            ),
            ("human", "{input}")
        ])
        
        document_prompt = PromptTemplate.from_template(
            """
            Paper: {title}
            Authors: {authors}
            Page: {page_number}
            arXiv ID: {arxiv_id}

            Content:
            {page_content}
            """
        )

        # QA chain is set up below to connect llm and retriever
        document_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, document_chain)
        print("QA chain set up.")
        return self.qa_chain
    
    def answer_query(self, query: str):
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        
        
        response = self.qa_chain.invoke({"input": query})
        self.result = response["answer"]
        print("Query:", query, "\nAnswer:", self.result)
        return self.result

    # using cosine similarity to test system accuracy
    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)

        return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )
    
    def evaluate(self, queries: list, ground_truths: list):
        # Determines the system's accuracy with sample queries and ground truths
        if len(queries) != len(ground_truths):
            raise ValueError("Queries and ground truths must be of the same length.")
        
        total_sim = 0
        correct = 0
        for idx, (query, truth) in enumerate(zip(queries, ground_truths)):
            response = self.qa_chain.invoke({"input": query})
            answer = response["answer"]
            
            truth_embedding = self.embeddings.embed_query(truth)
            answer_embedding = self.embeddings.embed_query(answer)
            
            similarity = self.cosine_similarity(
                truth_embedding,
                answer_embedding
            )
            
            print(
            "Query:", idx + 1,
            "\nExpected:", truth,
            "\nModel Answer:", answer,
            "\nSimilarity:", similarity
            )
            total_sim += similarity
            
            threshold = 0.80
            if similarity >= threshold:
                correct += 1
        length = len(queries) 
        accuracy = correct / length
        avg_sim = total_sim / length
        print("Avg similiarity: ", f"{avg_sim * 100:.2f}%")
        print(correct, " / ", length, " passed\n", f"Accuracy: {accuracy * 100:.2f}%")
        return avg_sim
    