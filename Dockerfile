# 1. Use a lightweight Python base image
FROM python:3.13.9

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy just the requirements file first to leverage Docker caching
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
# This includes main.py, qa_system.py, the data/ folder, 
# and the chroma_db/ folder you generated.
COPY . .

# 6. Expose the port your app will run on
EXPOSE 8000

# 7. The command to start your FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]