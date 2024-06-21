"""camera_pid controller."""
"""
Actividad 2.1 - Detección de carriles usando transformada de Hough
- A01018289 Renata Díaz Barreiro Castro
- A1260437 Rodia Zuriel Tejeda Moreno
- A0 Esteban Sánchez Retamoza
- A0 Miguel
"""


from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

#Display image 
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 60

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)

#-------------------
# Preprocesamiento de imágen compuesta por:
# 1. Conversión de color
# 2. Filtro Gaussiano
# 3. Canny Edge Detector
# 4. Extracción de ROI
def process_image_for_lane_detection(image):
    # Convertir a escala de grises
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar filtro Gaussiano con un kernel de 5x5 para suavizar la imagen
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # Aplicar detección de bordes con Canny
    edges = cv2.Canny(blur_img, 50, 150)

    # Definición de ROI
    #Nota: El área de interes formada por un 100% del largo por un 20% del ancho fue el que mejores resultados dió
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height * 0.8),
        (width, height * 0.8),
        (width, height),
        (0, height),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    # Extraer la región de interés de la imagen procesada
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

# Transformación de Hough para detectar líneas
def detect_lines(image):
    rho = 2
    theta = np.pi/180
    threshold = 10
    min_line_len = 10 # Valor bajo debido a las lineas cortas del carril con linea intermitente
    max_line_gap = 30 # Valor alto que mejor funcionó debido a las lineas discontinuas intermitentes que marcan el carril central
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    return lines

# Dibujar líneas encontradas sobre la imagen
def draw_lines(image, lines):
    alpha = 1
    beta = 1
    gamma = 1
    img_lines = np.zeros_like(image)
    if lines is not None: # Protección en caso de que no se hayan encontrado lineas
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2) # Crear las líneas de la línea
    img_lane_lines = cv2.addWeighted(image, alpha, img_lines, beta, gamma) # Colocar línea
    print("Total de líneas encontradas: " + str(len(img_lane_lines)))
    return img_lane_lines

# Calcular ángulo de dirección
# Se probaron distíntas estrategias, donde la mejor fue extraer el ángulo de la línea más prominente y buscar que el vehículo lo siguiera
# Otras estratégias probadas:
# - Basado en promedio de todas las líneas
# - Buscando centrar el vehículo entre línas del lado izquierdo y líneas del lado derecho
# - Basado en promedio de todas las líneas
def calculate_steering_angle(lines):
    if lines is None or len(lines) == 0:
        print("No se dectaron líneas, siguiendo recto.")
        return 0

    longest_line = None
    max_length = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_length:
                max_length = length
                longest_line = (x1, y1, x2, y2)

    if longest_line is None:
        print("No se encontró línea más larga, siguiendo recto..")
        return 0

    # Extraer las coordenadas de la línea más larga
    x1, y1, x2, y2 = longest_line
    # Calcular el ángulo de la línea más larga
    line_angle = np.arctan2(y2 - y1, x2 - x1)

    # Ajustar el ángulo para alinear la línea verticalmente
    # Haciendo que el ángulo de la línea sea cero respecto al vehículo
    desired_angle = -line_angle
    print("Se detectó línea más larga, direccionando a ángulo: " + str(desired_angle))

    # Realizar clip para garantizar que se utiliza un ángulo aceptable para Webots
    return np.clip(desired_angle, -1.0, 1.0)


#-------------------
# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        #----
        # Procesamiento de imagen
        processed_image = process_image_for_lane_detection(image)
        # Detección de líneas
        lines = detect_lines(processed_image)

        # Desplegar entrada de cámara con las líneas detectadas
        line_image = draw_lines(greyscale_cv2(image), lines)
        display_image(display_img, line_image)

        # Cálculo del ángulo de dirección
        angle = calculate_steering_angle(lines)
        driver.setSteeringAngle(angle)

        # Velocidad constante
        driver.setCruisingSpeed(speed)



if __name__ == "__main__":
    main()