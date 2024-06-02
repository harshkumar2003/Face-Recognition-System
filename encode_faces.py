import face_recognition
import os
import pickle

def encode_faces(dataset_path):
    encodings = {}
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            encodings[person] = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encodings[person].append(face_encodings[0])
    return encodings

dataset_path = 'dataset'
encodings = encode_faces(dataset_path)

# Save encodings to a file
with open('encodings.pickle', 'wb') as f:
    pickle.dump(encodings, f)

print("Encodings saved to encodings.pickle")
