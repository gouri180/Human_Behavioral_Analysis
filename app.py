
"""
Human Behaviour Analysis Dashboard

This Streamlit application provides a user interface for
analyzing human behaviour from videos using a remote
FastAPI server powered by Qwen2.5-VL.

Workflow
--------
1. User uploads a video.
2. Frames are extracted at configurable intervals.
3. Frames are encoded as Base64 strings.
4. Frames are sent to the FastAPI server.
5. Qwen2.5-VL analyzes the sequence of frames.
6. Behaviour description is displayed in the UI.

Requirements
------------
- Streamlit
- OpenCV
- Requests

Usage
-----
streamlit run app.py
"""

import cv2
import base64
import tempfile
import requests
import streamlit as st

# ============================================================
# Configuration
# ============================================================

# Public FastAPI server URL exposed through ngrok
BASE_URL = "include your ngrok server url"

# Analysis endpoint
ANALYZE_URL = f"{BASE_URL}/analyze"

# Required header when using ngrok free tier
HEADERS = {
    "ngrok-skip-browser-warning": "true"
}

# Default frame extraction settings
FRAME_INTERVAL_SECONDS = 1
MAX_FRAMES = 8

# ============================================================
# Streamlit Application Configuration
# ============================================================

st.set_page_config(
    page_title="Human Behaviour Analysis",
    page_icon="🎥",
    layout="wide"
)

st.title("Human Behaviour Analysis")

st.markdown(
    """
Upload a video and analyze human actions, object interactions,
and overall activity using the Qwen2.5-VL Vision Language Model.
"""
)

# ============================================================
# Sidebar Configuration
# ============================================================

with st.sidebar:

    st.header("Settings")

    FRAME_INTERVAL_SECONDS = st.slider(
        label="Frame Interval (Seconds)",
        min_value=1,
        max_value=5,
        value=1,
        help="Extract one frame every N seconds."
    )

    MAX_FRAMES = st.slider(
        label="Maximum Frames",
        min_value=4,
        max_value=20,
        value=8,
        help="Maximum number of frames sent for analysis."
    )

    st.divider()

    st.subheader("Server Status")

    if st.button("Check Server Connection"):

        try:

            response = requests.get(
                BASE_URL + "/",
                headers=HEADERS,
                timeout=30
            )

            st.success(
                f"Connected successfully "
                f"({response.status_code})"
            )

            try:
                st.json(response.json())
            except Exception:
                st.write(response.text[:500])

        except Exception as e:

            st.error(
                f"Connection failed: {e}"
            )

# ============================================================
# Frame Extraction Utility
# ============================================================

def extract_frames(video_path: str):
    """
    Extract frames from a video at a configurable interval.

    Parameters
    ----------
    video_path : str
        Path to the uploaded video file.

    Returns
    -------
    list[str]
        List of Base64 encoded image frames.
    """

    frames = []

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise ValueError(
            "Unable to determine video FPS."
        )

    frame_interval = int(
        fps * FRAME_INTERVAL_SECONDS
    )

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_count % frame_interval == 0:

            success, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if success:

                encoded_frame = base64.b64encode(
                    buffer.tobytes()
                ).decode("utf-8")

                frames.append(encoded_frame)

            if len(frames) >= MAX_FRAMES:
                break

        frame_count += 1

    cap.release()

    return frames

# ============================================================
# Video Upload Section
# ============================================================

uploaded_file = st.file_uploader(
    label="Upload Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file:

    st.subheader("Video Preview")

    # Center video preview
    left_col, center_col, right_col = st.columns(
        [1, 3, 1]
    )

    with center_col:

        st.video(
            uploaded_file,
            format="video/mp4"
        )

    st.write("")

    if st.button(
        label="Analyze Behaviour",
        use_container_width=True
    ):

        try:

            # ====================================================
            # Save Uploaded Video Temporarily
            # ====================================================

            with st.spinner(
                "Extracting frames..."
            ):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as tmp:

                    tmp.write(
                        uploaded_file.read()
                    )

                    temp_video_path = tmp.name

                frames = extract_frames(
                    temp_video_path
                )

            st.success(
                f"Successfully extracted "
                f"{len(frames)} frames."
            )

            payload = {
                "frames": frames
            }

            # ====================================================
            # Send Frames to API
            # ====================================================

            with st.spinner(
                "Analyzing behaviour..."
            ):

                response = requests.post(
                    ANALYZE_URL,
                    json=payload,
                    headers=HEADERS,
                    timeout=300
                )

            st.write(
                f"Server Response Code: "
                f"{response.status_code}"
            )

            # ====================================================
            # Display Results
            # ====================================================

            if response.status_code == 200:

                result = response.json()

                st.divider()

                st.subheader(
                    "Behaviour Analysis Result"
                )

                if "description" in result:

                    st.success(
                        result["description"]
                    )

                elif "error" in result:

                    st.error(
                        result["error"]
                    )

                else:

                    st.json(result)

            else:

                st.error(
                    f"Server Error "
                    f"({response.status_code})"
                )

                st.code(
                    response.text
                )

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )

