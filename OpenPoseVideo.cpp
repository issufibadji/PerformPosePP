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
// Mapeia os 18 pontos originais do modelo COCO para 12 pontos simplificados
static const int BODY_PART_MAP[12] = {
    0,  // nariz -> cabeca
    1,  // pescoco/peito
    5,  // ombro esquerdo
    2,  // ombro direito
    6,  // cotovelo esquerdo
    3,  // cotovelo direito
    7,  // pulso esquerdo
    4,  // pulso direito
    12, // joelho esquerdo
    13, // tornozelo esquerdo
    9,  // joelho direito
    10  // tornozelo direito
};

// Conexões do esqueleto usando apenas 12 pontos
static const int POSE_PAIRS[11][2] = {
    {0,1}, {1,2}, {1,3},
    {2,4}, {3,5},
    {4,6}, {5,7},
    {1,8}, {8,9},
    {1,10}, {10,11}
};
//Acessando rquitetura da rede neural
string protoFile = "coco/pose_deploy_linevec.prototxt";
//Armazena os pesos do modelo treinado
string weightsFile = "coco/pose_iter_440000.caffemodel";
//Total de pontos utilizados
const int nPoints = 12;
#endif

int main(int argc, char **argv)
{

    cout << "USAGE : ./OpenPose <videoFile>" << endl;
    cout << "USAGE : ./OpenPose <videoFile> <device>" << endl;
    
    string device = "cpu";
    //Entrada de Video
    string videoFile = "0"; // webcam padrao

    //Pegando argumentos da linha de comando
    if (argc == 2)
    {
        if ((string)argv[1] == "gpu")
            device = "gpu";
        else
            videoFile = argv[1];
    }
    else if (argc == 3)
    {
        videoFile = argv[1];
        if ((string)argv[2] == "gpu")
            device = "gpu";
    }
    // Definindo as dimensões de video  de entrada(altura e largura)
    int inWidth = 256;
    int inHeight = 256;
    float thresh = 0.01;
    //ler Video
    cv::VideoCapture cap;
    if (videoFile == "0")
        cap.open(0);
    else
        cap.open(videoFile);

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

                circle(frameCopy, cv::Point((int)p.x, (int)p.y), 5, Scalar(0,255,255), FILLED);
                cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
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