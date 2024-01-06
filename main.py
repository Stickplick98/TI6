import cv2 as cv
import numpy as np

img = cv.imread("C:\\TraitementImages3eme\\Images\\vaisseaux.jpg",cv.IMREAD_COLOR)
img_planete = cv.imread("C:\\TraitementImages3eme\\Images\\planete.jpg",cv.IMREAD_COLOR)

image_gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, binary_image = cv.threshold(image_gris, 20, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("thresh",binary_image)

contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

min_size2 = 500
max_size2 = 10000

filtered_contours = [cnt for cnt in contours if min_size2 < cv.contourArea(cnt) < max_size2]

result_image = np.zeros_like(image_gris)

cv.drawContours(result_image, filtered_contours, -1, (255), thickness=cv.FILLED)

image_couleur = img.copy()
image_couleur[result_image == 0] = 0

img_with_contours = image_couleur.copy()
cv.drawContours(img_with_contours, filtered_contours, -1, (0,0,255), thickness=1)

combined_image = np.where(result_image[:, :, None] != 0, image_couleur, img_planete)
combined_image2 = np.where(result_image[:, :, None] != 0, img_with_contours, img_planete)

cv.imshow("Image planete", img_planete)
cv.imshow("Image base", img)
cv.imshow("Image couleur", image_couleur)
cv.imshow("Image concatenée", combined_image)

cv.imshow("test", combined_image2)

cv.imwrite("Synthese.png", combined_image)
cv.imwrite("Synthese2.png", combined_image2)

cv.waitKey(0)
cv.destroyAllWindows()