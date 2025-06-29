#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;


//DEFINDO MODELO
#define COCO

// Mapeia os 18 pontos do modelo COCO para os 12 usados nesta versão leve
static const int BODY_PART_MAP[12] = {
    0,   // nariz → cabeca
    1,   // pescoco/peito
    5,   // ombro esquerdo
    2,   // ombro direito
    6,   // cotovelo esquerdo
    3,   // cotovelo direito
    7,   // pulso esquerdo
    4,   // pulso direito
    12,  // joelho esquerdo
    13,  // tornozelo esquerdo
    9,   // joelho direito
    10   // tornozelo direito
};

// Conexões do esqueleto para 12 pontos
static const int POSE_PAIRS[11][2] = {
    {0,1},   // Cabeca → Peito
    {1,2},   // Peito → Ombro esquerdo
    {1,3},   // Peito → Ombro direito
    {2,4},   // Ombro esquerdo → Cotovelo esquerdo
    {3,5},   // Ombro direito → Cotovelo direito
    {4,6},   // Cotovelo esquerdo → Pulso esquerdo
    {5,7},   // Cotovelo direito → Pulso direito
    {1,8},   // Peito → Joelho esquerdo
    {8,9},   // Joelho esquerdo → Tornozelo esquerdo
    {1,10},  // Peito → Joelho direito
    {10,11}  // Joelho direito → Tornozelo direito
};

//Acessando rquitetura da rede neural
string protoFile = "coco/pose_deploy_linevec.prototxt";
//Armazena os pesos do modelo treinado
string weightsFile = "coco/pose_iter_440000.caffemodel";
// Total de pontos utilizados
const int nPoints = 12;
//
int main(int argc, char **argv)
{

    cout << "USO : ./OpenPose <imageFile> " << endl;
    cout << "USO : ./OpenPose <imageFile> <device>" << endl;
    
    string device = "cpu";
   //Entrada da imagem
    string imageFile = "./images/koue.png";
    //Pegando argumentos da linha de comando
    if (argc == 2)
    {   
      if((string)argv[1] == "gpu")
        device = "gpu";
      else 
      imageFile = argv[1];
    }
    else if (argc == 3)
    {
        imageFile = argv[1];
        if((string)argv[2] == "gpu")
            device = "gpu";
    }

 // Definindo as dimensões da imagem de entrada(altura e largura)
    int inWidth = 256;
    int inHeight = 256;
    float thresh = 0.1;    
 //ler img
    Mat frame = imread(imageFile);
    Mat frameCopy = frame.clone();
    int frameWidth = frame.cols;
    int frameHeight = frame.rows;


    double t = (double) cv::getTickCount();
    //Lê a rede na memória
    Net net = readNetFromCaffe(protoFile, weightsFile);
    
    //Usando dispositivo GPU
    if (device == "cpu")
    {
        cout << "Usando dispositivo de CPU" << endl;
        net.setPreferableBackend(DNN_TARGET_CPU);
    }
    else if (device == "gpu")
    {
        cout << "Usando dispositivo GPU" << endl;
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    }
   
   // Prepara o quadro para ser alimentado na rede
    Mat inpBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
  // Define o objeto preparado como o blob de entrada da rede
    net.setInput(inpBlob);


 //Fazer previsões e analisar pontos-chave
 //O método forward para a classe DNN no OpenCV faz uma passagem direta pela rede(metodo que faz previsão)
    Mat output = net.forward();

    int H = output.size[2];
    int W = output.size[3];

//ENCONTRA PONTO-CHAVE DE CORPO
    // encontra a posição das partes do corpo
    vector<Point> points(nPoints);//forma pontos
    for (int n=0; n < nPoints; n++)
    {
        // Mapa de probabilidade da parte do corpo correspondente.
        Mat probMap(H, W, CV_32F, output.ptr(0, BODY_PART_MAP[n]));

        Point2f p(-1,-1);
        Point maxLoc;
        double prob;
        minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
        if (prob > thresh)
        {
            p = maxLoc;
            p.x *= (float)frameWidth / W ;
            p.y *= (float)frameHeight / H ;

            circle(frameCopy, cv::Point((int)p.x, (int)p.y), 4, Scalar(0,0,255), -1);
         cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1); 

        }
        points[n] = p;
    }
    
    
//DESENHAR ESQUELETO

    int nPairs = sizeof(POSE_PAIRS)/sizeof(POSE_PAIRS[0]);

    for (int n = 0; n < nPairs; n++)
    {
        // procura 2 partes do corpo
        Point2f partA = points[POSE_PAIRS[n][0]];
        Point2f partB = points[POSE_PAIRS[n][1]];

        if (partA.x<=0 || partA.y<=0 || partB.x<=0 || partB.y<=0)
            continue;

        line(frame, partA, partB, Scalar(0,255,0), 3, LINE_AA);
        circle(frame, partA, 5, Scalar(0,0,255), FILLED);
        circle(frame, partB, 5, Scalar(0,0,255), FILLED);

    }

//SAIDA/EXIBIR RESULTADO
    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    cout << "Tempo gasto = " << t << endl;
    imshow("Saida-Pontos-chave-Corpo", frameCopy);
    imwrite("./result/Saida-Pontos-chave-Corpo.jpg", frameCopy);

    imshow("Saida-esqueleto", frame);
    imwrite("./result/Saida-esqueleto.jpg", frame);
    
    waitKey();

    return 0;
}
