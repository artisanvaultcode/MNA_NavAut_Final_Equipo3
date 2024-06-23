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
    image = image / 255.0 - 0.5
    return np.expand_dims(image, axis=0)


def main():
    robot = Car()
    driver = Driver()

    camera = robot.getDevice("camera")
    camera.enable(int(robot.getBasicTimeStep()))

    # Create keyboard instance
    keyboard = Keyboard()
    keyboard.enable(int(robot.getBasicTimeStep()))

    current_speed = 50
    driver.setCruisingSpeed(current_speed)

    range_sensor = robot.getDevice("range-finder")
    range_sensor.enable(int(robot.getBasicTimeStep()))

    max_distance_threshold = 20  # Distancia máxima de umbral en metros
    min_distance_threshold = max_distance_threshold / 2  # Distancia mínima de umbral

    cruising_speed = 30  # Velocidad normal cuando no hay obstáculos
    reduced_speed = 15  # Velocidad reducida para mantener distancia

    max_angle = 0.25
    min_angle = -0.25

    while robot.step() != -1:
        manual_control = False


        if not manual_control:
            image = get_image(camera)
            processed_image = preprocess_image(image)
            normalized_angle = model.predict(processed_image)[0][0]
            print(f"Raw predicted steering angle: {normalized_angle:.4f}")

            # Desnormalizar el ángulo
            angle = normalized_angle # ((normalized_angle + 1) / 2) * (max_angle - min_angle) + min_angle
            steering_angle = angle

        print(f"Setting steering angle: {steering_angle:.4f}")
        driver.setSteeringAngle(steering_angle)

        sensor_data = range_sensor.getRangeImage()
        min_distance = np.min(sensor_data)  # Encontrar la distancia mínima detectada

        if min_distance < min_distance_threshold:
            print(f"Vehículo o peatón demasiado cerca a {min_distance:.2f} metros, deteniendo vehículo.")
            driver.setCruisingSpeed(0)
        elif min_distance < max_distance_threshold:
            print(f"Vehículo o peatón detectado a {min_distance:.2f} metros, reduciendo velocidad.")
            driver.setCruisingSpeed(reduced_speed)
        else:
            driver.setCruisingSpeed(cruising_speed)


if __name__ == "__main__":
    main()
