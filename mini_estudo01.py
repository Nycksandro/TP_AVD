from Experimento import Experimento
import cv2

exp = Experimento()
exp.preparar_dataset()

def sobel_completo(img, ksize=3):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(magnitude)

filtros = {
            "Canny":{
            "func": cv2.Canny,
            "params": {'threshold1': 110, 'threshold2': 190, 'apertureSize': 3, "L2gradient": True}
            },
            "Laplacian": {
            "func": cv2.Laplacian, 
            "params": {"ddepth": cv2.CV_8U, "ksize": 3, "scale": 2}
            },
            "Sobel": { # Referência das funções e os seus parâmetros
            "func": cv2.Sobel, 
            "params": {"ddepth": cv2.CV_64F, "dx": 1, "dy": 0, "ksize": 3, "scale": 1, "borderType": cv2.BORDER_DEFAULT}
            }
}

exp.mini_estudo01(filtros, 3)
