import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Face Recognition Attendance System")
st.write("Automatic attendance using Python & Computer Vision")

if "known_faces" not in st.session_state:
    st.session_state.known_faces = []

if "attendance" not in st.session_state:
    st.session_state.attendance = []

st.subheader("1️⃣ Register Students")

files = st.file_uploader(
    "Upload student face images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if files:
    st.session_state.known_faces = []

    for file in files:
        image = np.array(Image.open(file).convert("RGB"))
        encodings = face_recognition.face_encodings(image)

        if encodings:
            name = file.name.rsplit(".", 1)[0]
            st.session_state.known_faces.append(
                (name, encodings[0])
            )

    st.success(
        f"{len(st.session_state.known_faces)} student(s) registered."
    )

st.divider()

st.subheader("2️⃣ Take Attendance")

camera = st.camera_input("Take a photo")

if camera:
    image = np.array(Image.open(camera).convert("RGB"))

    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)

    if not encodings:
        st.warning("No face detected. Please try again.")

    elif not st.session_state.known_faces:
        st.warning("Please upload student face images first.")

    else:
        recognized = []

        known_names = [
            item[0] for item in st.session_state.known_faces
        ]

        known_encodings = [
            item[1] for item in st.session_state.known_faces
        ]

        for encoding in encodings:
            distances = face_recognition.face_distance(
                known_encodings, encoding
            )

            best_index = np.argmin(distances)

            if distances[best_index] < 0.50:
                name = known_names[best_index]

                today = datetime.now().strftime("%Y-%m-%d")

                already_marked = any(
                    x["Name"] == name and x["Date"] == today
                    for x in st.session_state.attendance
                )

                if not already_marked:
                    st.session_state.attendance.append({
                        "Name": name,
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Date": today
                    })

                recognized.append(name)

        if recognized:
            st.success(
                "Attendance marked: " +
                ", ".join(set(recognized))
            )
        else:
            st.error("Face not recognized.")

st.divider()

st.subheader("📋 Attendance Records")

if st.session_state.attendance:
    df = pd.DataFrame(st.session_state.attendance)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download Attendance CSV",
        df.to_csv(index=False),
        "Attendance.csv",
        "text/csv"
    )
else:
    st.info("No attendance records yet.")
