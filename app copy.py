import face_recognition
import cv2
import os

def load_known_faces(known_faces_dir):
    known_faces = []
    known_names = []
    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)
    return known_faces, known_names

def main():


    try:
        image = face_recognition.load_image_file("known_faces/WIN_20250101_11_37_23_Pro.jpg")
        print("Image loaded successfully.")
    except Exception as e:
        print(f"Error loading image: {e}")

        
    # Ruta de las imágenes conocidas
    known_faces_dir = "known_faces"  # Carpeta con imágenes organizadas por carpetas con nombres de personas
    known_faces, known_names = load_known_faces(known_faces_dir)


    # Inicializar cámara
    video_capture = cv2.VideoCapture(0)
    
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("No se puede acceder a la cámara.")
            break

        # Cambiar colores de BGR (OpenCV) a RGB (face_recognition)
        rgb_frame = frame[:, :, ::-1]

        # Localizar y codificar rostros en el cuadro actual
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Comparar el rostro detectado con los conocidos
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            name = "Desconocido"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            # Dibujar un rectángulo y etiqueta
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar la imagen
        cv2.imshow("Reconocimiento Facial", frame)

        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar cámara y cerrar ventanas
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
