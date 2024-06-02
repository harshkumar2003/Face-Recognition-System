# import face_recognition
# import cv2
# import pickle

# # Load known encodings from file
# with open('encodings.pickle', 'rb') as f:
#     known_encodings = pickle.load(f)

# # Only use the encodings for person1
# person1_encodings = known_encodings.get('person1', [])

# if not person1_encodings:
#     print("No encodings found for person1.")
#     exit()

# # Initialize the video capture
# video_capture = cv2.VideoCapture(0)

# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break

#     # Convert the frame from BGR to RGB (since face_recognition uses RGB)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Find all the faces and face landmarks in the frame
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')

#     # Loop through each detected face
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         matches = face_recognition.compare_faces(person1_encodings, face_encoding)
#         name = "Unknown"

#         # Check if a match was found
#         if any(matches):
#             name = "person1"
#             print("Match found: person1")
#         else:
#             print("No match found")

#         # Draw rectangle and label
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     # Display the resulting image
#     cv2.imshow('Video', frame)

#     # Hit 'q' on the keyboard to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()

import face_recognition
import cv2
import pickle

# Function to capture and encode faces for registration
def register_face():
    video_capture = cv2.VideoCapture(0)  # Open the camera

    while True:
        ret, frame = video_capture.read()  # Capture frame-by-frame
        if not ret:
            print("Failed to capture image from camera.")
            break
        
        # Convert the frame from BGR to RGB (since face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face landmarks in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')

        # If at least one face is detected, encode and store it
        if len(face_encodings) > 0:
            # Assume only one face is registered per image for simplicity
            new_face_encoding = face_encodings[0]

            # Save the new encoding to the dataset (encodings.pickle)
            # Assuming 'person1' is the identifier for the new face
            with open('encodings.pickle', 'rb') as f:
                known_encodings = pickle.load(f)

            if 'person1' not in known_encodings:
                known_encodings['person1'] = []

            known_encodings['person1'].append(new_face_encoding)

            with open('encodings.pickle', 'wb') as f:
                pickle.dump(known_encodings, f)

            print("Face registered successfully.")
            break  # Exit loop after registering one face

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Load known encodings from file
with open('encodings.pickle', 'rb') as f:
    known_encodings = pickle.load(f)

# Only use the encodings for person1
person1_encodings = known_encodings.get('person1', [])

if not person1_encodings:
    print("No encodings found for person1. Register your face first.")
    register_face()
    exit()

# Initialize the video capture for live recognition
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB (since face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face landmarks in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')

    # Loop through each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(person1_encodings, face_encoding)
        name = "Unknown"

        # Check if a match was found
        if any(matches):
            name = "person1"
            print("Match found: person1")
        else:
            print("No match found")

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
