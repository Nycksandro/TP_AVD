from Experimento import Experimento
import cv2

exp = Experimento()
exp.preparar_dataset()

filtros = { "Sobel": { # Referência das funções e os seus parâmetros
        "func": cv2.Sobel, 
        "params": {"ddepth": cv2.CV_8U, "dx": 1, "dy": 0, "ksize": 3}
        },
        "Laplacian": {
            "func": cv2.Laplacian, 
            "params": {"ddepth": cv2.CV_8U, "ksize": 3}
        }
}

exp.mini_estudo01(filtros, 3)