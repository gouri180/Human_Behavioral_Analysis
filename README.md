# Human Behaviour Analysis using Qwen2.5-VL

<div align="center">

# Human Behaviour Analysis using Qwen2.5-VL

Analyze human activities, actions, and object interactions from video sequences using a Vision Language Model.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Qwen2.5--VL-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## Project Overview

This project demonstrates how Vision Language Models (VLMs) can be used to understand human behavior from videos.

Instead of sending an entire video to a model, the system extracts representative frames and sends them to a FastAPI server powered by **Qwen2.5-VL-3B-Instruct**. The model analyzes the temporal sequence of frames and generates a natural language description of the observed activity.

Example output:

> "A person enters the room, places a bag on a desk, sits down, and begins working on a laptop."

---

## Features

* Video upload through Streamlit
* Human activity understanding
* Behavior description generation
* Object interaction analysis
* FastAPI REST API
* Qwen2.5-VL integration
* Remote access through ngrok
* Adjustable frame extraction interval
* Lightweight client-server architecture

---

## System Architecture

```text
┌─────────────────────────────┐
│      Uploaded Video         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Streamlit Frontend     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ OpenCV Frame Extraction     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Base64 Encoded Frames       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ FastAPI Server              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Qwen2.5-VL-3B-Instruct      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Behaviour Description       │
└─────────────────────────────┘
```

---

## Technology Stack

### Backend

* FastAPI
* Uvicorn
* Transformers
* PyTorch
* Qwen2.5-VL-3B-Instruct

### Frontend

* Streamlit

### Computer Vision

* OpenCV
* Pillow

### Deployment

* ngrok

---

## Project Structure

```text
human-behaviour-analysis/
│
├── server.py
├── client.py
├── app.py
├── requirements.txt
├── README.md
│
├── sample_videos/
│   └── sample.mp4
│
└── screenshots/
    ├── upload.png
    ├── preview.png
    └── result.png
```

---

## Hardware Requirements

### Minimum

* NVIDIA GPU with 12 GB VRAM
* 16 GB RAM
* CUDA-compatible GPU

### Recommended

* RTX 3060 12 GB
* RTX 4070
* RTX 4080
* NVIDIA A100

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/human-behaviour-analysis.git

cd human-behaviour-analysis
```

### Create Virtual Environment

Windows:

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the FastAPI Server

Start the server:

```bash
python server.py
```

Expected output:

```text
Loading model: Qwen/Qwen2.5-VL-3B-Instruct

Model loaded successfully.

INFO: Uvicorn running on http://0.0.0.0:8000
```

Verify:

```bash
curl http://localhost:8000/
```

Expected response:

```json
{
  "status": "running",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct"
}
```

---

## Exposing the Server using ngrok

Authenticate ngrok:

```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

Create a tunnel:

```bash
ngrok http 8000
```

Example:

```text
Forwarding

https://example.ngrok-free.app
```

Copy the generated public URL.

---

## Configure Streamlit

Open `app.py` and update:

```python
BASE_URL = "https://example.ngrok-free.app"
```

The application automatically creates:

```python
ANALYZE_URL = f"{BASE_URL}/analyze"
```

---

## Running the Streamlit Application

Launch Streamlit:

```bash
streamlit run app.py
```

Expected output:

```text
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

Open:

```text
http://localhost:8501
```

---

## Application Workflow

### Step 1

Start FastAPI:

```bash
python server.py
```

### Step 2

Expose the API:

```bash
ngrok http 8000
```

### Step 3

Update the ngrok URL inside `app.py`

### Step 4

Launch Streamlit:

```bash
streamlit run app.py
```

### Step 5

Upload a video

### Step 6

Configure:

* Frame Interval
* Maximum Frames

### Step 7

Click:

```text
Analyze Behaviour
```

### Step 8

View the generated behavior description

---

## API Endpoints

### Health Check

#### Request

```http
GET /
```

#### Response

```json
{
  "status": "running",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct"
}
```

---

### Behaviour Analysis

#### Request

```http
POST /analyze
```

#### Body

```json
{
  "frames": [
    "base64_frame_1",
    "base64_frame_2"
  ]
}
```

#### Response

```json
{
  "description": "A person walks into a room and begins working on a laptop."
}
```

---

## Example Use Cases

### Surveillance Analytics

* Activity monitoring
* Scene understanding
* Human behavior analysis

### Workplace Monitoring

* Employee activity understanding
* Human-object interaction analysis

### AI Demonstrations

* Vision Language Models
* Multimodal AI
* Video understanding

### Research Projects

* Action recognition
* Behavior understanding
* Video reasoning

---

## Future Improvements

* Real-time webcam analysis
* Multi-person behavior understanding
* YOLO integration
* Person tracking
* Temporal action recognition
* Structured JSON output
* Video summarization

---

## Results

Example generated output:

```text
A person enters a room, places a backpack on a table,
sits down, opens a laptop, and begins working.
```

---

## License

MIT License

---

## Acknowledgements

* Qwen Team
* Hugging Face
* FastAPI
* Streamlit
* OpenCV
* ngrok

---

## Author

**Gourinath**

Machine Learning Engineer | Computer Vision | Generative AI | Multimodal AI

This project demonstrates practical application of Vision Language Models for human behavior understanding using a scalable client-server architecture.
