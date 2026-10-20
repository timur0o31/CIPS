import cv2 as cv
import time
import os

def sobel_native(img):
    sobel_x = [[-1, 0, 1],[-2,0,2],[-1,0,1]]
    sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
    height, width = img.shape
    res = img.copy()
    res.fill(0)
    for y in range(1, height-1):
        for x in range(1, width-1):
            G_x = 0
            G_y = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    pixel = int(img[y + j, x + i])
                    G_x += pixel * sobel_x[j + 1][i + 1]
                    G_y += pixel * sobel_y[j + 1][i + 1]
            grad = (G_x**2 + G_y**2)**0.5
            res[y,x] = min(int(grad),255)
    return res

def sobel_opencv(img):
    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    magnitude = cv.magnitude(grad_x, grad_y)
    return cv.convertScaleAbs(magnitude)

def main():
    name = "img.png"
    image_path = f"images/{name}"
    os.makedirs("results/", exist_ok=True)
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    t0 = time.perf_counter()
    res_cv = sobel_opencv(img)
    t_cv = time.perf_counter() - t0
    t1 = time.perf_counter()
    res_native = sobel_native(img)
    t_native = time.perf_counter() - t1
    cv.imwrite(f"results/sobel_opencv_{name}", res_cv)
    cv.imwrite(f"results/sobel_native_{name}", res_native)
    print(f"OpenCV: {t_cv:.6f} секунд")
    print(f"Native: {t_native:.6f} секунд")
    print(f"Ускорение OpenCV в {t_native / t_cv:.1f} раз")

if __name__ == "__main__":
    main()