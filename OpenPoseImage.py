import cv2
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--image_file", default="single.jpeg", help="Input image")

args = parser.parse_args()

#DEFINDO MODELO
MODE = "COCO"

if MODE is "COCO":
    #Acessando rquitetura da rede neural
    protoFile = "coco/pose_deploy_linevec.prototxt"
    #//Armazena os pesos do modelo treinado
    weightsFile = "coco/pose_iter_440000.caffemodel"
    #18 pontos de corpos
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

#Entrada da imagem
frame = cv2.imread(args.image_file)
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
#Pegando argumentos da linha de comando
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Usando dispositivo de CPU")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Usando dispositivo de GPU ")

t = time.time()
# dimensões de imagem de entrada para a rede
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("tempo gasto pela rede : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

#Lista vazia para armazenar os pontos-chave detectados
points = []

for i in range(nPoints):
    #confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    #mapa de confiança da parte do corpo correspondente.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Dimensione o ponto para caber na imagem original
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Adiciona o ponto à lista se a probabilidade for maior que o limite
        points.append((int(x), int(y)))
    else :
        points.append(None)

# Desenhar Esqueleto
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

#SAIDA/EXIBIR RESULTADO
cv2.imshow('Saida-Pontos-chave-Corpo', frameCopy)
cv2.imshow('Saida-esqueleto', frame)


cv2.imwrite('./result/Saida-Pontos-chave-Corpo.jpg', frameCopy)
cv2.imwrite('./result/Saida-esqueleto.jpg', frame)

print("Tempo total gasto : {:.3f}".format(time.time() - t))

cv2.waitKey(0)

