from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import csv

# Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

# Display image
def display_image(display, image):
    image_rgb = np.dstack((image, image, image,))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# File operations
def save_image_data(file_name, angle):
    with open('image_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([file_name, angle])

# Main function
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(int(robot.getBasicTimeStep()))

    # Processing display
    display_img = Display("display_image")

    # Create keyboard instance
    keyboard = Keyboard()
    keyboard.enable(int(robot.getBasicTimeStep()))

    # Create image directory
    image_dir = "captured_images"
    os.makedirs(image_dir, exist_ok=True)

    # Check if CSV file exists to append header only if file does not exist
    if not os.path.exists('image_data.csv'):
        with open('image_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Steering Angle'])

    # Initial settings
    max_speed = 50  # Maximum speed
    min_speed = 10  # Minimum speed
    current_speed = min_speed  # Start at minimum speed
    driver.setCruisingSpeed(current_speed)
    fast_capture_interval = 5  # Fast capture when key pressed
    slow_capture_interval = 50  # Slow capture when no key pressed
    capture_interval = slow_capture_interval  # Initial capture interval
    counter = 0

    while driver.step() != -1:
        if counter % capture_interval == 0:
            image = get_image(camera)
            grey_image = greyscale_cv2(image)
            display_image(display_img, grey_image)
            current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")[:-3]
            file_name = f"{image_dir}/{current_datetime}.png"
            camera.saveImage(file_name, 100)
            save_image_data(file_name, driver.getSteeringAngle())
            print(f"Image saved: {file_name} with angle: {driver.getSteeringAngle()}")

        counter += 1

        # Keyboard handling for speed and turning
        key = keyboard.getKey()
        if key == Keyboard.UP:
            current_speed = max_speed
            driver.setCruisingSpeed(current_speed)
            capture_interval = fast_capture_interval
        elif key == Keyboard.DOWN:
            current_speed = min_speed
            driver.setCruisingSpeed(current_speed)
            capture_interval = slow_capture_interval
        elif key == Keyboard.RIGHT:
            driver.setSteeringAngle(0.25 if current_speed == min_speed else 0.1)
            capture_interval = fast_capture_interval
        elif key == Keyboard.LEFT:
            driver.setSteeringAngle(-0.25 if current_speed == min_speed else -0.1)
            capture_interval = fast_capture_interval
        else:
            driver.setSteeringAngle(0)  # Reset steering angle when not turning

if __name__ == "__main__":
    main()
