import os

import requests
import streamlit as st
from PIL import Image


DEFAULT_BACKEND_URL = "http://127.0.0.1:5000/api/predict"
BACKEND_URL = os.environ.get("AGRISCAN_BACKEND_URL", DEFAULT_BACKEND_URL)


def predict_with_backend(uploaded_file):
    uploaded_file.seek(0)
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    response = requests.post(BACKEND_URL, files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def show_prediction(result: dict) -> None:
    prediction = result["prediction"]
    confidence = float(result["confidence"])
    display_prediction = "Healthy" if "Healthy" in prediction else prediction

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {display_prediction}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(min(max(confidence / 100, 0.0), 1.0))

    if prediction == "Unknown":
        st.warning("Unknown leaf condition detected. The model confidence is below 60% or the image is outside known classes.")
    elif "Healthy" in prediction:
        st.success("Healthy rice leaf detected.")
    else:
        st.error(f"Disease detected: {prediction}")

    st.subheader("Disease Information")
    st.write(f"**Cause:** {result['cause']}")
    st.write(f"**Treatment suggestion:** {result['treatment']}")


def main() -> None:
    st.set_page_config(
        page_title="AgriScan - Rice Leaf Disease Detector",
        layout="centered",
    )

    st.title("AgriScan - Rice Leaf Disease Detector")

    uploaded_file = st.file_uploader(
        "Upload a close-up rice leaf image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload a close-up rice leaf image to begin detection.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Analyze Disease"):
        with st.spinner("Sending image to Flask backend..."):
            try:
                result = predict_with_backend(uploaded_file)
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to Flask backend. Start it with `python app.py`, "
                    "then run this Streamlit app again."
                )
                return
            except requests.exceptions.RequestException as exc:
                st.error(f"Backend request failed: {exc}")
                return

        if "error" in result:
            st.error(result["error"])
            return

        show_prediction(result)


if __name__ == "__main__":
    main()
