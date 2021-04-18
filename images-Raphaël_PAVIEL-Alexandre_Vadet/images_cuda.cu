#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#define RGBSIZE 3
using namespace cv;


__global__ void noiretblanc( unsigned char * in, unsigned char * out, std::size_t colonnes, std::size_t lignes ) {
  auto colonne = blockIdx.x * blockDim.x + threadIdx.x;
  auto ligne = blockIdx.y * blockDim.y + threadIdx.y;

  if( colonne < colonnes && ligne < lignes ) {
    int pos = ligne*colonnes+colonne;
    int posG = pos*RGBSIZE;

    unsigned char r = in[posG];
    unsigned char b = in[posG+1];
    unsigned char g = in[posG+2];

    out[posG]=0.21f*r+0.71f*g+0.07f*b;
    out[posG+1]=0.21f*r+0.71f*g+0.07f*b;
    out[posG+2]=0.21f*r+0.71f*g+0.07f*b;
  }
}


__global__ void retourner( unsigned char * in, unsigned char * out, std::size_t colonnes, std::size_t lignes ) {
  auto colonne = blockIdx.x * blockDim.x + threadIdx.x;
  auto ligne = blockIdx.y * blockDim.y + threadIdx.y;

  if( colonne < colonnes && ligne < lignes ) {

    int pos = RGBSIZE * ( ligne * colonnes + colonne );
    int oppose = colonnes*lignes*RGBSIZE - pos;

    auto rbis = in[oppose];
    auto gbis = in[oppose + 1];
    auto bbis = in[oppose + 2];

    out[pos] = rbis;
    out[pos + 1] = gbis;
    out[pos + 2] = bbis;
  }
}

__global__ void detectionContours(unsigned char * in, unsigned char * out, std::size_t colonnes, std::size_t lignes) {
  auto colonne = blockIdx.x * blockDim.x + threadIdx.x;
  auto ligne = blockIdx.y * blockDim.y + threadIdx.y;

  if (ligne >= 1 && ligne < lignes - 1 && colonne >= 1 && colonne < colonnes - 1)
  {
      for (std::size_t i = 0; i < RGBSIZE; ++i)
      {
          unsigned char p_h = in[RGBSIZE * ((ligne - 1) * colonnes + colonne) + i];
          unsigned char p_g = in[RGBSIZE * (ligne * colonnes + colonne - 1) + i];
          unsigned char pixel = in[RGBSIZE * (ligne * colonnes + colonne) + i];
          unsigned char p_d = in[RGBSIZE * (ligne * colonnes + colonne + 1) + i];
          unsigned char p_b = in[RGBSIZE * ((ligne + 1) * colonnes + colonne) + i];

          int resultat = p_h + p_g + (-4*pixel) + p_d + p_b ;
          if (resultat > 255)
          {
            resultat = 255;
          }
          if (resultat < 0)
          {
            resultat = 0;
          }
          out[RGBSIZE * (ligne * colonnes + colonne) + i] = resultat;
      }
  }
}

__global__ void ameliorationNettete(unsigned char * in, unsigned char * out, std::size_t colonnes, std::size_t lignes) {
  auto colonne = blockIdx.x * blockDim.x + threadIdx.x;
  auto ligne = blockIdx.y * blockDim.y + threadIdx.y;

  if (ligne >= 1 && ligne < lignes - 1 && colonne >= 1 && colonne < colonnes - 1)
  {
      for (std::size_t i = 0; i < RGBSIZE; ++i)
      {
          unsigned char p_h = in[RGBSIZE * ((ligne - 1) * colonnes + colonne) + i];
          unsigned char p_g = in[RGBSIZE * (ligne * colonnes + colonne - 1) + i];
          unsigned char pixel = in[RGBSIZE * (ligne * colonnes + colonne) + i];
          unsigned char p_d = in[RGBSIZE * (ligne * colonnes + colonne + 1) + i];
          unsigned char p_b = in[RGBSIZE * ((ligne + 1) * colonnes + colonne) + i];

          int resultat = -p_h - p_g + (5*pixel) - p_d - p_b ;
          if (resultat > 255)
          {
            resultat = 255;
          }
          if (resultat < 0)
          {
            resultat = 0;
          }
          out[RGBSIZE * (ligne * colonnes + colonne) + i] = resultat;
      }
  }
}

__global__ void flou(unsigned char * in, unsigned char * out, std::size_t colonnes, std::size_t lignes) {
  auto colonne = blockIdx.x * blockDim.x + threadIdx.x;
  auto ligne = blockIdx.y * blockDim.y + threadIdx.y;

  if (ligne >= 1 && ligne < lignes - 1 && colonne >= 1 && colonne < colonnes - 1)
  {
      for (std::size_t i = 0; i < RGBSIZE; ++i)
      {
          unsigned char p_hg = in[RGBSIZE * ((ligne - 1) * colonnes + colonne - 1) + i];
          unsigned char p_h = in[RGBSIZE * ((ligne - 1) * colonnes + colonne) + i];
          unsigned char p_hd = in[RGBSIZE * ((ligne - 1) * colonnes + colonne + 1) + i];
          unsigned char p_g = in[RGBSIZE * (ligne * colonnes + colonne - 1) + i];
          unsigned char pixel = in[RGBSIZE * (ligne * colonnes + colonne) + i];
          unsigned char p_d = in[RGBSIZE * (ligne * colonnes + colonne + 1) + i];
          unsigned char p_bg = in[RGBSIZE * ((ligne + 1) * colonnes + colonne - 1) + i];
          unsigned char p_b = in[RGBSIZE * ((ligne + 1) * colonnes + colonne) + i];
          unsigned char p_bd = in[RGBSIZE * ((ligne + 1) * colonnes + colonne + 1) + i];

          int resultat = (p_hg + p_h + p_hd + p_g + pixel + p_d + p_bg + p_b + p_bd)/9;
          if (resultat > 255)
          {
            resultat = 255;
          }
          if (resultat < 0)
          {
            resultat = 0;
          }
          out[RGBSIZE * (ligne * colonnes + colonne) + i] = resultat;
      }
  }
}

int main()
{
  cv::Mat m_in = cv::imread("image.jpeg", cv::IMREAD_UNCHANGED );
  cv::Mat m_out = m_in;

  auto lignes = m_in.rows;
  auto colonnes = m_in.cols;

  unsigned char * matrice_out;
  unsigned char * matrice_in;
  cudaMalloc( &matrice_in, RGBSIZE*lignes * colonnes );
  cudaMalloc( &matrice_out, RGBSIZE*lignes * colonnes );

  cudaMemcpy( matrice_in, m_in.data, RGBSIZE * lignes * colonnes, cudaMemcpyHostToDevice );

  dim3 t( 32, 32 );
  dim3 b( ( colonnes - 1) / t.x + 1 , ( lignes - 1 ) / t.y + 1 );

  int i;
  std::cout << "Entrez le filtre que vous voulez appliquer (1: Noir et Blanc // 2: Retourner // 3: Detection contours // 4: Amélioration de la netteté // 5: Flouter) : ";
  std::cin >> i;

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  if( i == 1 )
  {
    noiretblanc<<< b, t >>>(matrice_in, matrice_out, colonnes, lignes);

    cudaMemcpy( m_out.data, matrice_out,RGBSIZE*lignes * colonnes, cudaMemcpyDeviceToHost );

    cv::imwrite( "./resultat/NoirEtBlanc_CUDA.jpeg", m_out );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tempsexecution;
    cudaEventElapsedTime(&tempsexecution, start, stop);
    std::cout << "Temps_NoirEtBlanc: " << tempsexecution << " millisecondes" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  else if( i == 2 )
  {
    retourner<<< b, t >>>(matrice_in, matrice_out, colonnes, lignes);

    cudaMemcpy( m_out.data, matrice_out,RGBSIZE*lignes * colonnes, cudaMemcpyDeviceToHost );

    cv::imwrite( "./resultat/Retourner_CUDA.jpeg", m_out );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tempsexecution;
    cudaEventElapsedTime(&tempsexecution, start, stop);
    std::cout << "Temps_Retourner: " << tempsexecution << " millisecondes" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  else if( i == 3 )
  {
    detectionContours<<< b, t >>>(matrice_in, matrice_out, colonnes, lignes);

    cudaMemcpy( m_out.data, matrice_out,RGBSIZE*lignes * colonnes, cudaMemcpyDeviceToHost );

    cv::imwrite( "./resultat/DetectionContours_CUDA.jpeg", m_out );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tempsexecution;
    cudaEventElapsedTime(&tempsexecution, start, stop);
    std::cout << "Temps_DetectionContours: " << tempsexecution << " millisecondes" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  else if( i == 4 )
  {
    ameliorationNettete<<< b, t >>>(matrice_in, matrice_out, colonnes, lignes);

    cudaMemcpy( m_out.data, matrice_out,RGBSIZE*lignes * colonnes, cudaMemcpyDeviceToHost );

    cv::imwrite( "./resultat/AmeliorationNettete_CUDA.jpeg", m_out );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tempsexecution;
    cudaEventElapsedTime(&tempsexecution, start, stop);
    std::cout << "Temps_AmeliorationNettete: " << tempsexecution << " millisecondes" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  else if( i == 5 )
  {
    flou<<< b, t >>>(matrice_in, matrice_out, colonnes, lignes);

    cudaMemcpy( m_out.data, matrice_out,RGBSIZE*lignes * colonnes, cudaMemcpyDeviceToHost );

    cv::imwrite( "./resultat/Flouter_CUDA.jpeg", m_out );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tempsexecution;
    cudaEventElapsedTime(&tempsexecution, start, stop);
    std::cout << "Temps_Flouter: " << tempsexecution << " millisecondes" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  else
  {
    std::cout << "Opération impossible" << std::endl;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  cudaFree( matrice_in);
  cudaFree( matrice_out);

  return 0;
}
