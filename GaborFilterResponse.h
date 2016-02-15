#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define Total_train_face 100
#define Total_probe_face 100
#define Height 311
#define Width 232
#define Filter_Num 40

#define N_ary 8
#define Remaining_Num 15

#define PI 3.1415926

double complex_modulus(double *t);
void GaborFilterResponse(double trainface[][Width],double Gabor_Respone[][Height][Width][2],double Mean_Value[][2]);

void Gabor_Respone_Mean(double *temp_mean,double Convolv[Height][Width][2]);
void GaborWavelet(int Row, int Column, double Kmax, double f, int u, int v, double Delt2, double GW[][Width][2]);
void convolv2_same(double x[][Width],double (*y)[Width][2],double (*z)[Width][2]);
