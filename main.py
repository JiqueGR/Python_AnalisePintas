import numpy as np
import cv2
import colorgram
from PIL import Image
from matplotlib import pyplot as plt

def redimensionar(imagem):
    redimensionar = True
    while redimensionar:
        if imagem.shape[0] > 500 or imagem.shape[1] > 500:
            imagem = cv2.resize(imagem, (int(imagem.shape[1] * 0.9), int(imagem.shape[0] * 0.9)))
        else:
            redimensionar = False
    return imagem

def imagemGrabCut(imagem):
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    retorno, thresh = cv2.threshold(imagemCinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contorno, hierarquia = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_contorno = max(contorno, key=cv2.contourArea)
    x, y, largura, altura = cv2.boundingRect(max_contorno)
    mascara = np.zeros(imagem.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    retangulo = (x, y, largura, altura)
    cv2.grabCut(imagem, mascara, retangulo, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mascara2 = np.where((mascara == 2) | (mascara == 0), 0, 1).astype('uint8')
    imagemGC = imagem * mascara2[:, :, np.newaxis]
    return imagemGC

def analisaBordaContorno(imagem):
    imagem = cv2.medianBlur(imagem, 5)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    retorno, thresh = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("Contorno", thresh)
    contorno, hierarquia = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_contorno = max(contorno, key=cv2.contourArea)
    epsilon = 0.1 * cv2.arcLength(max_contorno, True)
    aproximacao = cv2.approxPolyDP(max_contorno, epsilon, True)
    areaAproximacao = cv2.contourArea(aproximacao)
    areaContorno = cv2.contourArea(max_contorno)
    if areaAproximacao > 0 and areaContorno / areaAproximacao >= 0.8 and areaContorno / areaAproximacao <= 1.6:
        print("Contorno arredondado de acordo com a função contorno.")
    else:
        print("Contorno não arredondado de acordo com a função contorno.")

def analisarBordaPixel(imagem):
    imagem = cv2.medianBlur(imagem, 5)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagemAux = np.zeros((int(imagem.shape[0]), int(imagem.shape[1]), 4), dtype=np.uint8)
    retorno, thresh = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contorno, hierarquia = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_contorno = max(contorno, key=cv2.contourArea)
    elipse = cv2.fitEllipseDirect(max_contorno)
    cv2.drawContours(imagemAux, [max_contorno], 0, (0, 0, 255), -1)
    cv2.ellipse(imagemAux, elipse, (255, 255, 255), -1)
    cv2.imshow("Pixel", imagemAux)
    altura = imagemAux.shape[0]
    largura = imagemAux.shape[1]
    contador = 0
    for y in range(altura):
        for x in range(largura):
            valor_vermelho = imagemAux[y, x, 2]
            valor_verde = imagemAux[y, x, 1]
            valor_azul = imagemAux[y, x, 0]
            if valor_vermelho == 255 and valor_verde != 255 and valor_azul != 255:
                contador = contador + 1
    if (contador * 100) / (altura * largura) < 2:
        print("Contorno arredondado de acordo com a função pixel.")
    else:
        print("Contorno não arredondado de acordo com a função pixel.")

def analisarCorRGB(cores):
    dentro_do_parametro = 0
    fora_do_parametro = 0
    descartado = 0
    texto = "\nAnalise de cor da pinta:"
    for indice, cor in enumerate(cores[1:], start=1):
        r, g, b = cor.rgb
        proportion = cor.proportion
        if ((r >= 50 and r <= 160) and (g >= 30 and g <= 120) and (b >= 20 and g <= 80)):
            texto += "\nCor {}, está dentro do parâmetro.".format(indice)
            dentro_do_parametro += 1
        elif (proportion >= 0.05):
            texto += "\nCor {}, não está dentro do parâmetro.".format(indice)
            fora_do_parametro += 1
        else:
            texto += "\nCor {},possui poucos pixels e foi descartado.".format(indice)
            descartado += 1
    categorias = ["Dentro do Parâmetro", "Fora do Parâmetro", "Descartado"]
    contagens = [dentro_do_parametro, fora_do_parametro, descartado]
    plt.bar(categorias, contagens)
    plt.xlabel("Condição")
    plt.ylabel("Contagem")
    plt.title("Análise de Cores RGB")
    plt.show()
    print(texto)

def analisarCorHistograma(segmento, imagemGC):
    hist1 = cv2.calcHist([segmento], [0], None, [256], [1, 255])
    hist2 = cv2.calcHist([imagemGC], [0], None, [256], [1, 255])
    chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    if(chi_square < 0.02):
        print("\nComparação de histograma: Cor da pinta dentro do intervalo esperado.")
    else:
        print("\nComparação de histograma: Cor da pinta fora do intervalo esperado.")
    plt.figure(figsize=(8, 4))
    plt.plot(hist1, color='blue', label='Segmento')
    plt.plot(hist2, color='red', label='ImagemGC')
    plt.title("Comparação")
    plt.xlabel("Intensidade")
    plt.ylabel("Pixels")
    plt.legend()
    plt.show()

def main():
    imagem = cv2.imread("maligna.jpg")
    imagem = redimensionar(imagem)
    cv2.imshow("Imagem Original", imagem)

    analisarBordaPixel(imagem)
    analisaBordaContorno(imagem)

    imagemGC = imagemGrabCut(imagem)
    imagemGCRGB = imagemGrabCut(imagem)
    imagemGCRGB = cv2.cvtColor(imagemGCRGB, cv2.COLOR_BGR2RGB)
    image_pil_RGB = Image.fromarray(imagemGCRGB)
    colors = colorgram.extract(image_pil_RGB, 4)
    analisarCorRGB(colors)

    minimo = (20,20,20)
    maximo = (60,255,255)
    mascara = cv2.inRange(imagemGC, minimo, maximo)
    segmento = cv2.bitwise_and(imagem, imagem, mask = mascara)

    analisarCorHistograma(segmento, imagemGC)


    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break

main()