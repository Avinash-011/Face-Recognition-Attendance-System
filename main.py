import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

st.title("📸 Face Attendance")

if "students" not in st.session_state:
    st.session_state.students = {}

name = st.text_input("Student Name")

photo = st.camera_input("📷 Take Student Photo")

if st.button("Register Student"):

    if not name:
        st.warning("Enter student name")

    elif not photo:
        st.warning("Take a photo first")

    else:
        image = Image.open(photo).convert("RGB")
        image.thumbnail((800, 800))

        img = np.array(image)

        faces = face_recognition.face_locations(img)

        if len(faces) == 0:
            st.error("❌ No face found")

        elif len(faces) > 1:
            st.error("❌ Only one face allowed")

        else:
            encoding = face_recognition.face_encodings(
                img, faces
            )[0]

            st.session_state.students[name.strip()] = encoding

            st.success(
                f"✅ {name} registered successfully!"
            )
