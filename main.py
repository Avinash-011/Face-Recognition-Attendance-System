import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Face Attendance",
    page_icon="📸"
)

st.title("📸 Face Recognition Attendance")
st.caption("AI-based Student Attendance System")


# =========================
# LOAD FACE RECOGNITION
# =========================

@st.cache_resource
def load_face_recognition():
    import face_recognition
    return face_recognition


# =========================
# SESSION DATA
# =========================

if "students" not in st.session_state:
    st.session_state.students = {}

if "attendance" not in st.session_state:
    st.session_state.attendance = []


# =========================
# REGISTER STUDENT
# =========================

st.subheader("👤 Register Student")

student_name = st.text_input(
    "Student Name",
    placeholder="Enter name"
)

student_photo = st.file_uploader(
    "Upload one clear face photo",
    type=["jpg", "jpeg", "png"]
)

if st.button("Register Student", type="primary"):

    if not student_name:
        st.warning("Enter student name.")

    elif not student_photo:
        st.warning("Upload a photo.")

    else:

        face_recognition = load_face_recognition()

        img = Image.open(student_photo).convert("RGB")

        # Very small image = less memory
        img.thumbnail((500, 500))

        img_array = np.asarray(img)

        # Fast CPU face detection
        locations = face_recognition.face_locations(
            img_array,
            model="hog",
            number_of_times_to_upsample=0
        )

        if len(locations) == 0:

            st.error("❌ No face found.")

        elif len(locations) > 1:

            st.error(
                "❌ Use a photo containing only one face."
            )

        else:

            encoding = face_recognition.face_encodings(
                img_array,
                locations,
                num_jitters=1,
                model="small"
            )[0]

            st.session_state.students[
                student_name.strip()
            ] = encoding

            st.success(
                f"✅ {student_name} registered!"
            )


# =========================
# STUDENTS
# =========================

if st.session_state.students:

    st.write(
        f"**Registered students:** "
        f"{len(st.session_state.students)}"
    )

    with st.expander("View Students"):

        for name in st.session_state.students:
            st.write("•", name)


st.divider()


# =========================
# ATTENDANCE
# =========================

st.subheader("📸 Mark Attendance")

camera = st.camera_input(
    "Take student's photo"
)

if camera:

    if not st.session_state.students:

        st.warning(
            "Register a student first."
        )

    else:

        face_recognition = load_face_recognition()

        img = Image.open(camera).convert("RGB")

        # Reduce camera image
        img.thumbnail((500, 500))

        img_array = np.asarray(img)

        locations = face_recognition.face_locations(
            img_array,
            model="hog",
            number_of_times_to_upsample=0
        )

        if not locations:

            st.error("❌ No face detected.")

        else:

            encodings = face_recognition.face_encodings(
                img_array,
                locations,
                num_jitters=1,
                model="small"
            )

            names = list(
                st.session_state.students.keys()
            )

            known_faces = list(
                st.session_state.students.values()
            )

            marked = False

            for encoding in encodings:

                distances = face_recognition.face_distance(
                    known_faces,
                    encoding
                )

                best_index = int(
                    np.argmin(distances)
                )

                if distances[best_index] < 0.50:

                    name = names[best_index]

                    today = datetime.now().strftime(
                        "%Y-%m-%d"
                    )

                    already_marked = any(
                        row["Name"] == name
                        and row["Date"] == today
                        for row in st.session_state.attendance
                    )

                    if already_marked:

                        st.info(
                            f"ℹ️ {name} already present today."
                        )

                    else:

                        st.session_state.attendance.append({
                            "Name": name,
                            "Time": datetime.now().strftime(
                                "%H:%M:%S"
                            ),
                            "Date": today
                        })

                        st.success(
                            f"✅ Attendance marked: {name}"
                        )

                    marked = True

            if not marked:

                st.error(
                    "❌ Face not recognized."
                )


# =========================
# ATTENDANCE RECORDS
# =========================

st.divider()

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

    csv = df.to_csv(index=False)

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
