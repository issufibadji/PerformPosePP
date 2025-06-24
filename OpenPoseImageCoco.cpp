#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;


//DEFINDO MODELO
#define COCO
#ifdef COCO
const int POSE_PAIRS[17][2] = 
{   
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10},
    {1,11}, {11,12}, {12,13},
    {1,0}, {0,14},
    {14,16}, {0,15}, {15,17}
};

//Acessando rquitetura da rede neural
string protoFile = "coco/pose_deploy_linevec.prototxt";
//Armazena os pesos do modelo treinado
string weightsFile = "coco/pose_iter_440000.caffemodel";
//18 pontos de corpos
int nPoints = 18;
#endif
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
    int inWidth = 368;
    int inHeight = 368;
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
        Mat probMap(H, W, CV_32F, output.ptr(0,n));

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

        line(frame, partA, partB, Scalar(0,255,255), 6);
        circle(frame, partA, 8, Scalar(0,0,255), -1);
        circle(frame, partB, 8, Scalar(0,0,255), -1);

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
