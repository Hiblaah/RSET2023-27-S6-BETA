# Cloud-Storage-phone-cluster (PocketCluster)

## Overview
PocketCluster is a distributed, zero-trust cloud storage system that leverages old or unused mobile devices as storage nodes. Instead of relying on centralized data centers, the system encrypts, chunks, and distributes files across a trusted cluster of mobile phones. 

A unique feature of PocketCluster is its implementation of **Semantic Search (RAG)** over encrypted files. The system generates high-level semantic embeddings of file contents during upload, allowing users to query their files using natural language without the system ever having to decrypt the file data during the search process.

## Key Features
- **Zero-Trust Mobile Storage**: The storage nodes (mobile devices) only ever receive encrypted chunks of data. They cannot read, reconstruct, or search the actual file content.
- **Secure File Distribution**: Files are encrypted, split into chunks, and distributed with replication across multiple connected phones to ensure redundancy.
- **Semantic Search**: Natural language search capability that operates on lightweight semantic summaries and embeddings, preserving privacy while enabling powerful discovery.
- **Real-Time Cluser Dashboard**: A home page providing a comprehensive view of connected devices, global storage usage, and overall cluster health.

## System Architecture
### 1. Upload Flow
1. **Semantic Summarization**: Upon file selection, the controller generates a semantic summary of the content to capture meaning (not the whole file).
2. **Embedding Generation**: The summary is embedded into a vector space.
3. **Encryption & Chunking**: The actual file is encrypted, split into distinct chunks, and distributed to online mobile storage nodes.
4. **Metadata Storage**: The controller stores the embedding and chunk-to-node metadata securely.

### 2. Download Flow
1. **Semantic Search**: The user's search query is converted into an embedding and matched against file summaries.
2. **Chunk Retrieval**: Once a file is identified, the controller looks up the metadata and requests specific encrypted chunks from the storage nodes.
3. **Reconstruction**: The controller collects all chunks, decrypts them, reconstructs the original file, and delivers it to the user. Storage nodes are oblivious to this decryption process.

## Tech Stack
- **Backend**: Python, FastAPI, WebSockets, SQLAlchemy, MySQL
- **Storage Node/Client (Android App)**: Kotlin, Jetpack Compose, ONNX Runtime, Hugging Face Tokenizers (AI/Embeddings), Room Database

## Getting Started

### Prerequisites
- Python 3.9+
- Android Studio (for Client App)
- MySQL Database

### Setting up the Backend
1. Navigate to the `backend` directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up the schema in your MySQL database according to `backend/app/schema.sql` (update DB URI in configuration as needed).
5. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Running the Android Storage Node / Client
1. Open Android Studio.
2. Select **Open** and choose the `PhoneClusterApp` directory (e.g., `Cloud-Storage-phone-cluster/PhoneClusterApp`).
3. Sync the project with Gradle files.
4. Ensure you have an Android device or emulator running.
5. Build and run the project from Android Studio.