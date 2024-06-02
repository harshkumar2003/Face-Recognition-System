import tkinter as tk
import customtkinter as ctk
from tkinter import simpledialog, messagebox
import cv2
import os
import face_recognition
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.configure(bg='#2E4053')

        # Initialize variables
        self.dataset_path = "dataset"
        self.video_capture = cv2.VideoCapture(0)
        self.person_name = None

        # Create UI elements
        title_frame = ctk.CTkFrame(root, fg_color="#2E4053")
        title_frame.pack(pady=25)

        label = ctk.CTkLabel(title_frame, text="Face Recognition System", font=("Helvetica", 25, "bold"), text_color='#F7F9F9')
        label.pack()

        button_frame = ctk.CTkFrame(root, fg_color="#2E4053")
        button_frame.pack(pady=20)

        register_button = ctk.CTkButton(button_frame, text="Register Person", command=self.start_registration, font=("Helvetica", 14), fg_color='#5DADE2', text_color='white')
        register_button.pack(pady=10)

        verify_button = ctk.CTkButton(button_frame, text="Verify Person", command=self.verify_person, font=("Helvetica", 14), fg_color='#5DADE2', text_color='white')
        verify_button.pack(pady=10)

        capture_button = ctk.CTkButton(button_frame, text="Capture Image", command=self.capture_image, font=("Helvetica", 14), fg_color='#5DADE2', text_color='white')
        capture_button.pack(pady=10)

        # Initialize video feed display
        self.video_feed_label = ctk.CTkLabel(root, fg_color="#2E4053")
        self.video_feed_label.pack()

        # Start displaying live video feed
        self.show_video_feed()

    def show_video_feed(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Convert frame from BGR to RGB for displaying in tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to fit the label size
            rgb_frame = cv2.resize(rgb_frame, (640, 480))

            # Convert RGB frame to PIL format and then to Tkinter PhotoImage
            img = Image.fromarray(rgb_frame)
            img = ImageTk.PhotoImage(image=img)

            # Update label with the new frame
            self.video_feed_label.img = img
            self.video_feed_label.configure(image=img)

        # Schedule the next update after 10 ms
        self.video_feed_label.after(10, self.show_video_feed)

    def start_registration(self):
        self.person_name = simpledialog.askstring("Register Person", "Enter person's name:")
        if self.person_name:
            messagebox.showinfo("Information", f"Registration started for {self.person_name}. "
                                                "Please click 'Capture Image' when ready.")

    def capture_image(self):
        if self.person_name:
            ret, frame = self.video_capture.read()
            if ret:
                # Convert frame to RGB (face_recognition library uses RGB format)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame to a standard size (optional)
                rgb_frame = cv2.resize(rgb_frame, (640, 480))

                # Save the captured image in the dataset folder under the person's name
                person_dir = os.path.join(self.dataset_path, self.person_name)
                os.makedirs(person_dir, exist_ok=True)
                image_path = os.path.join(person_dir, f"{self.person_name}_{len(os.listdir(person_dir)) + 1}.jpg")
                cv2.imwrite(image_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

                messagebox.showinfo("Information", f"Image captured and saved for {self.person_name}.")
                self.person_name = None  # Reset person name after capturing image

    def verify_person(self):
        ret, frame = self.video_capture.read()

        if not ret:
            messagebox.showerror("Error", "Failed to capture image from camera.")
            return

        # Convert frame to RGB (face_recognition library uses RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            messagebox.showinfo("Info", "No faces detected in the captured image.")
            return

        # Compute face encodings for the captured image
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            messagebox.showinfo("Info", "No face encodings found in the captured image.")
            return

        # Take the first face encoding (assuming one face per verification)
        face_encoding = face_encodings[0]

        # Load known face encodings and names
        known_face_encodings = []
        known_face_names = []

        for person_name in os.listdir(self.dataset_path):
            person_dir = os.path.join(self.dataset_path, person_name)
            if os.path.isdir(person_dir):  # Only process directories
                for image_file in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_file)

                    # Load image from file
                    known_image = face_recognition.load_image_file(image_path)

                    # Compute face encodings for the known image
                    known_face_encoding = face_recognition.face_encodings(known_image)
                    if known_face_encoding:
                        known_face_encodings.append(known_face_encoding[0])
                        known_face_names.append(person_name)

        # Compare face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Find the best match (the smallest distance)
        best_match_index = None
        if matches:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
        else:
            name = "Unknown"

        messagebox.showinfo("Result", f"Match found, Person: {name}")

    def __del__(self):
        # Release the camera when the object is deleted
        if self.video_capture.isOpened():
            self.video_capture.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

