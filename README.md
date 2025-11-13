# Aurora-Technical-Assessment-NLP-QA




For the vector store, I chose ChromaDB's PersistentClient to create a simple, file-based database. This is ideal for a self-contained take-home project as it requires no separate server setup. For a full-scale production environment with multiple users, I would run ChromaDB as a dedicated server and have the FastAPI app connect via the HttpClient to manage concurrency and allow for independent scaling of the API and the database.


## Future works
If this API were scaled to production with live, streaming user messages, I would integrate an observability platform like LangSmith. This would be critical for evaluating answer quality, tracking costs, and monitoring for model drift as new types of user questions emerge.