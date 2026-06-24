# Human Behaviour Analysis using Qwen2.5-VL

## Overview

This project demonstrates human behaviour understanding from videos using the Qwen2.5-VL-3B-Instruct Vision Language Model.

The system follows a client-server architecture:

- A Streamlit application provides the user interface.
- OpenCV extracts key frames from uploaded videos.
- Frames are encoded as Base64 strings and sent to a FastAPI server.
- Qwen2.5-VL analyzes the sequence of frames.
- The server returns a natural language description of the observed actions and behaviour.

---

## Features

- Video upload through Streamlit
- Human behaviour analysis
- Action recognition
- Object interaction understanding
- FastAPI REST API
- Qwen2.5-VL-3B-Instruct integration
- ngrok support for remote deployment
- Configurable frame extraction settings

---

## System Architecture

```text
                     ┌────────────────────┐
                     │   Uploaded Video   │
                     └──────────┬─────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Streamlit Frontend  │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Frame Extraction (CV2)  │
                  └──────────┬──────────────┘
                             │
                             ▼
                ┌─────────────────────────────┐
                │ Base64 Encoded Video Frames │
                └──────────┬──────────────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │ FastAPI Server│
                   └───────┬───────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Qwen2.5-VL-3B-Instruct  │
              └──────────┬──────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │ Behaviour Description Text │
            └────────────────────────────┘
