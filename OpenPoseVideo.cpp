#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
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

int main(int argc, char **argv)
{

    cout << "USAGE : ./OpenPose <videoFile> " << endl;
    cout << "USAGE : ./OpenPose <videoFile> <device>" << endl;
    
    string device = "cpu";
    //Entrada de Video
    string videoFile = "sample_video.mp4";

    //Pegando argumentos da linha de comando
    if (argc == 2)
    {   
      if((string)argv[1] == "gpu")
        device = "gpu";
      else 
      videoFile = argv[1];
    }
    else if (argc == 3)
    {
        videoFile = argv[1];
        if((string)argv[2] == "gpu")
            device = "gpu";
    }
    // Definindo as dimensões de video  de entrada(altura e largura)
    int inWidth = 368;
    int inHeight = 368;
    float thresh = 0.01;    
    //ler Video
    cv::VideoCapture cap(videoFile);

    if (!cap.isOpened())
    {
        cerr << "Não foi possível conectar à câmera" << endl;
        return 1;
    }
    
    Mat frame, frameCopy;
    int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    
    VideoWriter video("simple.avi",VideoWriter::fourcc('M','J','P','G'), 10, Size(frameWidth,frameHeight));
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

    double t=0;
    while( waitKey(1) < 0)
    {       
        double t = (double) cv::getTickCount();

        cap >> frame;
        frameCopy = frame.clone();
        // Prepara o quadro para ser alimentado na rede
        Mat inpBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
        // Define o objeto preparado como o blob de entrada da rede
        net.setInput(inpBlob);

        //Fazer previsões e analisar pontos-chave
        //O método forward para a classe DNN no OpenCV faz uma passagem direta pela rede(metodo que faz previsão)
        Mat output = net.forward();

        int H = output.size[2];
        int W = output.size[3];

        ///ENCONTRA PONTO-CHAVE DE CORPO
        // encontra a posição das partes do corpo
        vector<Point> points(nPoints);
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

                circle(frameCopy, cv::Point((int)p.x, (int)p.y), 8, Scalar(0,255,255), -1);
                cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 1.1, cv::Scalar(0, 0, 255), 2);
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

        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        cv::putText(frame, cv::format("Tempo gasto = %.2f sec", t), cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 50, 0), 2);
        imshow("Pontos-Chave", frameCopy);
        imshow("Saida-Esqueleto", frame);
        video.write(frame);
    }
    // Quando tudo estiver pronto, solte a captura de vídeo 
    cap.release();
    video.release();

    return 0;
}