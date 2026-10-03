import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from pathlib import Path
import urllib.request

st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Face Recognition Attendance")
st.caption("AI-based Student Attendance System")

# -------------------------------------------------
# MODEL FILES
# -------------------------------------------------

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)


@st.cache_resource
def download_models():

    if not YUNET_PATH.exists():
        urllib.request.urlretrieve(
            YUNET_URL,
            str(YUNET_PATH)
        )

    if not SFACE_PATH.exists():
        urllib.request.urlretrieve(
            SFACE_URL,
            str(SFACE_PATH)
        )

    return True


@st.cache_resource
def load_face_models():

    download_models()

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (320, 320),
        0.85,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        str(SFACE_PATH),
        ""
    )

    return detector, recognizer


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "students" not in st.session_state:
    st.session_state.students = {}

if "attendance" not in st.session_state:
    st.session_state.attendance = []


# -------------------------------------------------
# GET FACE EMBEDDING
# -------------------------------------------------

def get_face_embedding(image, detector, recognizer):

    if image is None:
        return None

    h, w = image.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # Use largest detected face
    faces = sorted(
        faces,
        key=lambda x: x[2] * x[3],
        reverse=True
    )

    face = faces[0]

    aligned = recognizer.alignCrop(
        image,
        face
    )

    feature = recognizer.feature(
        aligned
    )

    return feature


# -------------------------------------------------
# REGISTER STUDENT
# -------------------------------------------------

st.subheader("👤 Register Student")

student_name = st.text_input(
    "Student Name",
    placeholder="Enter student name"
)

student_photo = st.file_uploader(
    "Upload one clear face photo",
    type=["jpg", "jpeg", "png"]
)


if st.button(
    "Register Student",
    type="primary"
):

    if not student_name.strip():

        st.warning(
            "Please enter student name."
        )

    elif student_photo is None:

        st.warning(
            "Please upload a face photo."
        )

    else:

        try:

            with st.spinner(
                "Loading face recognition model..."
            ):

                detector, recognizer = (
                    load_face_models()
                )

            img = Image.open(
                student_photo
            ).convert("RGB")

            img_array = np.asarray(img)

            img_array = cv2.cvtColor(
                img_array,
                cv2.COLOR_RGB2BGR
            )

            embedding = get_face_embedding(
                img_array,
                detector,
                recognizer
            )

            if embedding is None:

                st.error(
                    "❌ No clear face detected. "
                    "Please upload a front-facing photo."
                )

            else:

                name = student_name.strip()

                st.session_state.students[
                    name
                ] = embedding

                st.success(
                    f"✅ {name} registered successfully!"
                )

        except Exception as e:

            st.error(
                "Face recognition error."
            )

            st.exception(e)


# -------------------------------------------------
# REGISTERED STUDENTS
# -------------------------------------------------

if st.session_state.students:

    st.write(
        f"**Registered students: "
        f"{len(st.session_state.students)}**"
    )

    with st.expander(
        "View Registered Students"
    ):

        for name in st.session_state.students:

            st.write(
                "•",
                name
            )


st.divider()


# -------------------------------------------------
# MARK ATTENDANCE
# -------------------------------------------------

st.subheader("📸 Mark Attendance")

camera = st.camera_input(
    "Take student's photo"
)


if camera is not None:

    if not st.session_state.students:

        st.warning(
            "Register a student first."
        )

    else:

        try:

            with st.spinner(
                "Recognizing face..."
            ):

                detector, recognizer = (
                    load_face_models()
                )

            img = Image.open(
                camera
            ).convert("RGB")

            img_array = np.asarray(img)

            img_array = cv2.cvtColor(
                img_array,
                cv2.COLOR_RGB2BGR
            )

            h, w = img_array.shape[:2]

            detector.setInputSize(
                (w, h)
            )

            _, faces = detector.detect(
                img_array
            )

            if faces is None or len(faces) == 0:

                st.error(
                    "❌ No face detected."
                )

            else:

                marked = False

                for face in faces:

                    aligned = recognizer.alignCrop(
                        img_array,
                        face
                    )

                    query_feature = (
                        recognizer.feature(
                            aligned
                        )
                    )

                    best_name = None
                    best_score = -1

                    for name, known_feature in (
                        st.session_state.students.items()
                    ):

                        score = (
                            recognizer.match(
                                known_feature,
                                query_feature,
                                cv2.FaceRecognizerSF_FR_COSINE
                            )
                        )

                        if score > best_score:

                            best_score = score
                            best_name = name

                    # SFace cosine similarity threshold
                    if best_score >= 0.363:

                        today = datetime.now().strftime(
                            "%Y-%m-%d"
                        )

                        already_marked = any(
                            row["Name"] == best_name
                            and row["Date"] == today
                            for row in st.session_state.attendance
                        )

                        if already_marked:

                            st.info(
                                f"ℹ️ {best_name} "
                                "already present today."
                            )

                        else:

                            now = datetime.now()

                            st.session_state.attendance.append(
                                {
                                    "Name": best_name,
                                    "Time": now.strftime(
                                        "%H:%M:%S"
                                    ),
                                    "Date": today
                                }
                            )

                            st.success(
                                f"✅ Attendance marked: "
                                f"{best_name}"
                            )

                        marked = True

                        break

                if not marked:

                    st.error(
                        "❌ Face not recognized."
                    )

        except Exception as e:

            st.error(
                "Face recognition error."
            )

            st.exception(e)


st.divider()


# -------------------------------------------------
# ATTENDANCE RECORDS
# -------------------------------------------------

st.subheader("📋 Attendance Records")

if st.session_state.attendance:

    df = pd.DataFrame(
        st.session_state.attendance
    )

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download Attendance CSV",
        csv,
        "Attendance.csv",
        "text/csv"
    )

else:

    st.info(
        "No attendance records yet."
  )
