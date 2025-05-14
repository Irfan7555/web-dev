from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import (CharacterTextSplitter, RecursiveCharacterTextSplitter)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector
import tempfile
import os
import yaml
import shutil
from langchain.schema import Document
from enum import Enum
from typing import List
import boto3
from botocore.client import Config
import psycopg2
from typing import List

connection_string = os.getenv("DATABASE_URL")

class EmbeddingModel(Enum):
    OPEN_AI = "text-embedding-3-small"


class ChunkingStrategy(Enum):
    SEMANTIC_PERCENTILE = "semantic_percentile"
    SEMANTIC_INTERQUARTILE = "semantic_interquartile"
    SEMANTIC_STD_DEV = "semantic_std_dev"
    TEXT_SPLIT = "text_splitter"
    RECURSIVE = "recursive"




def load_settings(settings_path="settings.yml"):
    """Load settings from a YAML file."""
    try:
        with open(settings_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Error loading settings file: {e}")


def load_pdf_with_pages(file):
    """Load a PDF file and extract content with page numbers."""
    suffix = os.path.splitext(file.name)[1]
    temp_file_path = None

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            # Save the uploaded file to the temporary file
            shutil.copyfileobj(file, temp_file)
            temp_file.flush()
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Extract content and page numbers
        pages = []
        for doc in documents:
            page_number = doc.metadata.get('page', 0) + 1  # Convert to 1-based indexing
            pages.append({
                "page_content": doc.page_content,
                "page_number": page_number
            })

        return {
            "filename": file.name,
            "pages": pages
        }
    finally:
        # Attempt to delete the file, but handle the case where it might still be in use
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                # Log the error but don't crash
                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")
                # On Windows, files may be deleted later when no longer in use


def load_docx_with_pages(file):
    """Load a DOCX file and extract content with page numbers."""
    suffix = os.path.splitext(file.name)[1]
    temp_file_path = None

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            # Save the uploaded file to the temporary file
            shutil.copyfileobj(file, temp_file)
            temp_file.flush()

        # The file is now closed properly after the with block

        # Use Docx2txtLoader to load DOCX content
        loader = Docx2txtLoader(temp_file_path)
        documents = loader.load()

        # Extract content and page numbers
        pages = []
        for i, doc in enumerate(documents):
            pages.append({
                "page_content": doc.page_content,
                "page_number": i + 1  # 1-based page numbering
            })

        return {
            "filename": file.name,
            "pages": pages
        }
    finally:
        # Attempt to delete the file, but handle the case where it might still be in use
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                # Log the error but don't crash
                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")


def initialize_embedding_model(embedding_model):
    if embedding_model.startswith("text"):  # OpenAI models
        embeddings = OpenAIEmbeddings(model=embedding_model)
        return embeddings


def create_chunking(
        chunking_type,
        documents,
        chunk_size=2000,
        chunk_overlap=100,
        percentile=90,
        interquartile_range_factor=1.5,
        standard_deviation_factor=3,
):
    if chunking_type == "text_splitter":
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunked_documents = text_splitter.split_documents(documents)
    elif chunking_type == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunked_documents = text_splitter.split_documents(documents)
    elif chunking_type == "semantic_percentile":
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile",
                                           breakpoint_threshold_amount=percentile)
        chunked_documents = semantic_chunker.split_documents(documents)
    elif chunking_type == "semantic_interquartile":
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="interquartile",
                                           breakpoint_threshold_amount=percentile)
        chunked_documents = semantic_chunker.split_documents(documents)
    elif chunking_type == "semantic_std_dev":
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="standard_deviation",
                                           breakpoint_threshold_amount=percentile)
        chunked_documents = semantic_chunker.split_documents(documents)
    else:
        raise ValueError(
            "Invalid chunking type. Choose from 'text_splitter', 'recursive', 'semantic_percentile', 'semantic_interquartile', or 'semantic_std_dev'."
        )

    print(len(chunked_documents))
    return chunked_documents


def process_and_chunk_file(
        file,
        db_name=None,
        embedding_model=None,
        splitter_type=None,
        tab=None,
        chunk_size=1000,  # Not used but kept for compatibility
        chunk_overlap=200,  # Not used but kept for compatibility
):
    try:
        # Ensure the FAISS_Index directory exists
        os.makedirs("FAISS_Index", exist_ok=True)

        file_name = file.name.lower()

        # Load the file with page metadata
        if file_name.endswith('.pdf'):
            file_data = load_pdf_with_pages(file)
        elif file_name.endswith('.docx'):
            file_data = load_docx_with_pages(file)
        else:
            raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

        # Convert the loaded data with page metadata into LangChain document format
        total_pages = len(file_data["pages"])
        documents = []
        for page in file_data["pages"]:
            documents.append(
                Document(
                    page_content=page["page_content"],
                    metadata={
                        "page": page["page_number"] - 1,
                        "source": file_data["filename"],
                        "total_pages": total_pages
                    }
                )
            )

        # Initialize semantic text splitter
        embedding_model= initialize_embedding_model(embedding_model)

        chunks  = create_chunking(chunking_type=splitter_type, documents=documents)

        if tab == "To FAISS":
            # FAISS logic
            index_path = f"FAISS_Index/{db_name}_FAISS" if db_name else f"FAISS_Index/default_FAISS"

            if os.path.exists(index_path) and os.path.isdir(index_path):
                # Load the existing FAISS store
                existing_vectorstore = FAISS.load_local(index_path, embedding_model,
                                                        allow_dangerous_deserialization=True)
                existing_vectorstore.add_documents(chunks)
                existing_vectorstore.save_local(index_path)
            else:
                # Create new FAISS index
                vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
                vectorstore.save_local(index_path)



        elif tab == "From S3":
            collection_name = db_name

            vectorstore = PGVector(
                embeddings=embedding_model,
                collection_name=collection_name,
                connection=connection_string,
                use_jsonb=True
            )

            try:
                vectorstore.from_existing_index(
                    embedding=embedding_model,
                    collection_name=collection_name,
                    connection=connection_string
                )

                # Add documents to the existing collection

                vectorstore.add_documents(chunks)



            except Exception as e:

                print(f"Collection '{collection_name}' does not exist. Creating a new one. Error: {e}")
                vectorstore.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    collection_name=collection_name,
                    connection=connection_string,
                    use_jsonb=True
                )

        return {
            "status": "success",
            "chunk_count": len(chunks),
        }
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


def get_postgres_conn():
    return psycopg2.connect(connection_string)

# Function to list all PGVector collections
def list_pgvector_collections():
    conn = get_postgres_conn()
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'langchain_pg_collection'
            );
        """)
        collection_table_exists = cursor.fetchone()[0]

        if collection_table_exists:
            cursor.execute("""
                SELECT name, uuid
                FROM langchain_pg_collection
                ORDER BY name;
            """)
            collections = cursor.fetchall()
        else:
            collections = []

    conn.close()
    return collections

def create_retriever(collection_name, connection_string, embedding_function):
    """Create a retriever connected to the specified PGVector collection."""
    retriever = PGVector(
        connection=connection_string,
        collection_name=collection_name,
        embeddings=embedding_function,
    )
    return retriever


def search_documents(retriever, query, top_k=4):
    """Search for relevant documents with similarity scores."""
    results_with_scores = retriever.similarity_search_with_score(query, k=top_k)

    # Normalize score to a scale of 0-1 (optional, but can be useful)
    normalized_results = []
    for doc, score in results_with_scores:
        normalized_score = 1 / (1 + float(score))  # Normalizing the score (optional)
        doc.metadata['score'] = normalized_score
        normalized_results.append(doc)

    return normalized_results

def delete_pgvector_collection(collection_name: str) -> str:
    """Delete a PGVector collection from 'langchain_pg_collection' and vectors table."""
    try:
        if not connection_string:
            raise Exception("Database connection string not found in environment variables.")

        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()

        # First, delete from langchain_pg_collection to remove tracking
        cursor.execute("""
            DELETE FROM langchain_pg_collection
            WHERE name = %s
            RETURNING *;
        """, (collection_name,))
        result = cursor.fetchone()

        if result:
            collection_id = result[0]
            # Now delete associated vectors from langchain_pg_embedding
            cursor.execute("""
                DELETE FROM langchain_pg_embedding
                WHERE collection_id = %s;
            """, (collection_id,))
            conn.commit()
            return f"Collection '{collection_name}' and its embeddings deleted successfully."
        else:
            return f"Collection '{collection_name}' not found."

    except Exception as e:
        return f"Error while deleting collection: {str(e)}"

    finally:
        if conn:
            conn.close()


def retrieve_documents(path: str, tab: str, embedding_model, query: str, k: int = 4) -> List[Document]:
    if tab == "From FAISS":
        try:
            vectorstore = FAISS.load_local(folder_path=path, embeddings=embedding_model,
                                           allow_dangerous_deserialization=True)
            docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)

            for doc, score in docs_and_scores:
                normalized_score = 1 / (1 + float(score))
                doc.metadata['score'] = normalized_score

            return [doc for doc, _ in docs_and_scores]
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")

    if tab == "From Postgres":
        try:
            collection_name = path

            # Create retriever
            retriever = create_retriever(
                collection_name=collection_name,
                connection_string=connection_string,
                embedding_function=embedding_model
            )

            # Perform search
            results = search_documents(retriever, query, top_k=k)

            return results

        except Exception as e:
            raise Exception(f"Error retrieving documents from Postgres: {str(e)}")


def get_retriever(path: str, embedding_model, k=16, search_type = "mmr", ):
    try:
        vectorstore = FAISS.load_local(folder_path=path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_type=search_type, k=k,
                                              fetch_k = k*2)
        return retriever
    except Exception as e:
        raise Exception(f"Error getting retriever: {str(e)}")
