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
    "Sobel": {
        "func": sobel_completo,   # função auxiliar abaixo
        "params": {"ksize": 3}
    },
    "Laplacian": {
        "func": cv2.Laplacian,
        "params": {"ddepth": cv2.CV_8U, "ksize": 3}
    }
}

exp.mini_estudo01(filtros, 3)