## Global Mobile Reviews Sentiment – FastAPI + Streamlit + Docker

This project provides **mobile reviews sentiment classification** using:

- **FastAPI** backend (`main.py`) exposing a `/predict` endpoint.
- **Streamlit** frontend (`app.py`) with an attractive UI for single-review analysis.
- **Hybrid ONNX + SentenceTransformer** model stored in the `model/` folder.
- **Dockerfile** for containerized deployment and easy hosting.

---

### Show project app link 
https://mobile-reviews-sentiment-analysis.streamlit.app/


### 1. Project structure

- **`main.py`**: FastAPI app + model loading and `predict_sentiment` function.
- **`app.py`**: Streamlit UI that imports and uses `predict_sentiment`.
- **`model/`**: Contains `hybrid_sentiment.onnx` (and optionally `.pth`).
- **`requirements.txt`**: Python dependencies.
- **`Dockerfile`**: Image for running Streamlit app in a container.

---

### 2. Local setup (without Docker)

```bash
git clone <your-repo-url>.git
cd "Global Mobile Reviews"
python -m venv .venv
source .venv/Scripts/activate  # on Windows Git Bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Run FastAPI backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API test example (in another terminal):

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"This phone is amazing, battery life is great!\"}"
```

#### Run Streamlit UI

```bash
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

---

### 3. Docker usage

#### Build image

From the project root:

```bash
docker build -t mobile-reviews-sentiment:latest .
```

#### Run container

```bash
docker run -p 8501:8501 mobile-reviews-sentiment:latest
```

Then open `http://localhost:8501` in your browser.

---

### 4. Deploying with GitHub + Docker registry

1. **Push code to GitHub** – include all project files and the `model/` folder (or host the model elsewhere and download it at build time).
2. **Create a Docker registry** – e.g. Docker Hub or GitHub Container Registry.
3. **Build and push manually** from your machine:

```bash
docker tag mobile-reviews-sentiment:latest YOUR_DOCKER_USERNAME/mobile-reviews-sentiment:latest
docker push YOUR_DOCKER_USERNAME/mobile-reviews-sentiment:latest
```

4. Deploy the image on your preferred platform (e.g. Azure Container Apps, AWS ECS, Cloud Run, etc.).

---

### 5. Deploying on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On Streamlit Cloud:
   - Create a new app.
   - Point to your GitHub repo.
   - Set **Main file path** to `app.py`.
3. Ensure `requirements.txt` is present (this project already has it).
4. Deploy – Streamlit will install dependencies and run `app.py`.

---

### 6. Notes and customization

- The **UI** is defined in `app.py` – you can tweak colors, layout, and text there.
- To change the model, update:
  - The encoder in `main.py` (`SentenceTransformer("all-MiniLM-L6-v2")`).
  - The ONNX path and input tensor name if your new ONNX model differs.
- For **batch predictions** or more analytics (e.g., sentiment distribution charts), extend `app.py` to accept multiple reviews at once and visualize them.


