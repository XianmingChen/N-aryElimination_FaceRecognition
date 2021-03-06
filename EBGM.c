#include "cv.h"
#include "highgui.h"
#include "EBGM_FeatureVectors.h"
#include "EBGM_FaceComparison.h"

void read_image(char *filepath,double image[][Width])
{
	int i,j;
	CvScalar s;
	IplImage *img=cvLoadImage(filepath,0);
    for(i=0;i<img->height;i++)
	{
        for(j=0;j<img->width;j++)
		{
			s=cvGet2D(img,i,j); 
			image[i][j]=s.val[0]/255;
        }
    }
    cvReleaseImage(&img);  
}

void main()
{
	int i,j,k,m;
	//double trainface[Height][Width]={0.0};
	//double Gabor_Respone[Filter_Num][Height][Width][2]={0.0};

	//double Mean_Value[Filter_Num][2]={0.0};
	double Each_Feature_Vectors[500][41][2]={0.0};
	double Feature_Vectors[Total_train_face][500][41][2]={0.0};

	int train_feature_count[Total_train_face]={0};
	int each_feature_count=0;
	int probe_feature_count=0;

	int first_index;
	int Probe_count=0;

	int candidate_index[Total_train_face];

	char file_path[255];
	FILE *Gabor_file;
	double temp;
	int temp_count;

	double Active_Feature_Vectors[Total_train_face][500][41][2]={0.0};
	int Active_feature_count[Total_train_face]={0};
	int Active_trainface;
	int temp_candidate;

	FILE *record;
	int remainder;

	record=fopen("result_NE_EBGM.txt","a+");

	for (i=0;i<Total_train_face;i++)
	{
		printf("Training image %d...\n",i+1);

		//Read feature count from txt file
		sprintf(file_path,"Aligned_FERET/input/trainfaces/Count/%d.txt",i+1);
		Gabor_file=fopen(file_path,"r");
		fscanf(Gabor_file,"%ld",&temp_count);
		train_feature_count[i]=temp_count;
		fclose(Gabor_file);

		//Read Feature matrix from txt file
		sprintf(file_path,"Aligned_FERET/input/trainfaces/Feature/%d.txt",i+1);
		Gabor_file=fopen(file_path,"r");
		for (j=0;j<500;j++)
		{
			for (k=0;k<41;k++)
			{
				for (m=0;m<2;m++)
				{
					fscanf(Gabor_file,"%lf",&temp);
					Each_Feature_Vectors[j][k][m]=temp;
				}
			}
		}
		fclose(Gabor_file);

		memcpy(Feature_Vectors[i],Each_Feature_Vectors,500*41*2*8);
	}
	 
	printf("Begin to probe images...\n");
	for (i=0;i<Total_probe_face;i++)
	{
		//Read probe count from txt file
		sprintf(file_path,"Aligned_FERET/input/probefaces/Count/%d.txt",i+1);
		Gabor_file=fopen(file_path,"r");
		fscanf(Gabor_file,"%ld",&temp_count);
		probe_feature_count=temp_count;
		fclose(Gabor_file);

		//Read probe feature from txt file
		sprintf(file_path,"Aligned_FERET/input/probefaces/Feature/%d.txt",i+1);
		Gabor_file=fopen(file_path,"r");
		for (j=0;j<500;j++)
		{
			for (k=0;k<41;k++)
			{
				for (m=0;m<2;m++)
				{
					fscanf(Gabor_file,"%lf",&temp);
					Each_Feature_Vectors[j][k][m]=temp;
				}
			}
		}
		fclose(Gabor_file);

		//EBGM_FeatureVectors(Gabor_Respone,Mean_Value,&each_feature_count,Each_Feature_Vectors);
		//probe_feature_count=each_feature_count;
		//each_feature_count=0;


		for (j=0;j<Total_train_face;j++)
		{
			candidate_index[j]=j;
		}

		Active_trainface=Total_train_face;
		while (Active_trainface>Remaining_Num)
		{
			for (j=0;j<Active_trainface;j++)
			{
				temp_candidate=candidate_index[j];
				memcpy(&Active_Feature_Vectors[j],&Feature_Vectors[temp_candidate],500*41*2*sizeof(double));
				Active_feature_count[j]=train_feature_count[temp_candidate];
			}

			first_index=EBGM_FaceComparison(Active_trainface,Active_feature_count,Active_Feature_Vectors,
				probe_feature_count,Each_Feature_Vectors,candidate_index);
			
			remainder=Active_trainface%N_ary;
			if (remainder==0)
				Active_trainface=Active_trainface/N_ary;
			else
				Active_trainface=Active_trainface/N_ary+1;

			/*if(Active_trainface>=Remaining_Num*N_ary)
			{
				Active_trainface=Active_trainface/N_ary;
			}
			else if (Active_trainface>Remaining_Num)
			{
				Active_trainface=Remaining_Num;
			}
			else
			{
				Active_trainface=0;
			}*/
		}

		for (j=0;j<Active_trainface;j++)
		{
			temp_candidate=candidate_index[j];
			memcpy(&Active_Feature_Vectors[j],&Feature_Vectors[temp_candidate],500*41*2*sizeof(double));
			Active_feature_count[j]=train_feature_count[temp_candidate];
		}

		first_index=EBGM_FaceComparison(Active_trainface,Active_feature_count,Active_Feature_Vectors,
			probe_feature_count,Each_Feature_Vectors,candidate_index);

		if (first_index==i)
		{
			Probe_count++;
			printf("Image %d Probe successfully!\n",i+1);
		}
		else
		{
			printf("Image %d Probe failed!\n",i+1);
		}
	}
	printf("%d Images probe completed!\n",Total_probe_face);
	printf("Probe accuary= %f\n",(double)Probe_count/Total_probe_face);

	fprintf(record,"Probe accuary of %d images with %d -ary elimination is:= %f\n",Total_probe_face,N_ary,(double)Probe_count/Total_probe_face);
	fprintf(record,"\n");
	fclose(record);

	system("pause");
}