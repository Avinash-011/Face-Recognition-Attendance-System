import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Attendance", page_icon="📸")

st.title("📸 Face Attendance System")

if "students" not in st.session_state:
    st.session_state.students = {}

st.subheader("👤 Register Student")

name = st.text_input("Student Name")

photo = st.camera_input("📷 Take Photo")

if photo and st.button("Register Student", type="primary"):

    if not name.strip():
        st.warning("Enter student name.")
        st.stop()

    image = Image.open(photo).convert("RGB")
    image.thumbnail((800, 800))

    img = np.array(image)

    faces = face_recognition.face_locations(
        img,
        model="hog"
    )

    if len(faces) == 0:
        st.error("❌ No face found.")

    elif len(faces) > 1:
        st.error("❌ Only one face allowed.")

    else:
        encoding = face_recognition.face_encodings(
            img,
            faces
        )[0]

        st.session_state.students[name.strip()] = encoding

        st.success(
            f"✅ {name} registered successfully!"
        )

        st.image(
            image,
            caption="Registered Photo",
            width=250
        )

st.divider()

st.subheader("👥 Registered Students")

for name in st.session_state.students:
    st.write("•", name)
