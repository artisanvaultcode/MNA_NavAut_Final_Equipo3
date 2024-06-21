import threading
from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model('behavioural_cloning_model.keras')


def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]  # Retornar solo los canales RGB


def preprocess_image(image):
    image = cv2.resize(image, (200, 66))
    image = image / 255.0 - 0.5  # Normalizar la imagen
    return np.expand_dims(image, axis=0)


threshold = 0.01


def adjust_steering_angle(angle):
    # Verificar si el ángulo supera el umbral positivo o negativo
    if angle > threshold:
        # Multiplicar por 10 y limitar a 0.25
        return min(angle * 10, 0.25)
    elif angle < -threshold:
        # Multiplicar por 10 y limitar a -0.25
        return max(angle * 10, -0.25)
    else:
        # Si el ángulo está dentro del umbral, simplemente retornar el ángulo original
        return angle


def main():
    robot = Car()
    driver = Driver()

    camera = robot.getDevice("camera")
    camera.enable(int(robot.getBasicTimeStep()))
    # Create keyboard instance
    keyboard = Keyboard()
    keyboard.enable(int(robot.getBasicTimeStep()))
    max_speed = 30
    min_speed = 10
    current_speed = max_speed
    driver.setCruisingSpeed(current_speed)
    steering_angle = [0]  # Usamos una lista para permitir modificaciones dentro del hilo

    while robot.step() != -1:
        image = get_image(camera)
        processed_image = preprocess_image(image)

        # Calcular el ángulo de dirección de manera asincrónica
        def prediction_thread():
            angle = model.predict(processed_image)[0][0]
            print(f"Raw predicted steering angle: {angle:.4f}")
            steering_angle[0] = adjust_steering_angle(angle)

        threading.Thread(target=prediction_thread).start()

        driver.setSteeringAngle(steering_angle[0])

        key = keyboard.getKey()
        if key == Keyboard.UP:
            current_speed = max_speed
        elif key == Keyboard.DOWN:
            current_speed = min_speed
        driver.setCruisingSpeed(current_speed)


if __name__ == "__main__":
    main()
