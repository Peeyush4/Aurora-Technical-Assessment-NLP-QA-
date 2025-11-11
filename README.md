# Aurora-Technical-Assessment-NLP-QA




For the vector store, I chose ChromaDB's PersistentClient to create a simple, file-based database. This is ideal for a self-contained take-home project as it requires no separate server setup. For a full-scale production environment with multiple users, I would run ChromaDB as a dedicated server and have the FastAPI app connect via the HttpClient to manage concurrency and allow for independent scaling of the API and the database.