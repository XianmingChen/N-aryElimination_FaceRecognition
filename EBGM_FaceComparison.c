#include "EBGM_FaceComparison.h"
#include "EBGM_FeatureVectors.h"

void Exchange(double *A, double *B)
{
	double temp;
	temp=*A;
	*A=*B;
	*B=temp;
}

int random_num(int start, int end)
{
	int num;

	srand((unsigned)time(NULL)); 
	num=(rand()%(end-start))+start;
	return num;
}

int partition(double *Address, int start, int end)
{
	double pivot;
	int i,j;

	i=start-1;
	pivot=Address[end];

	for (j=start;j<end;j++)
	{
		if (Address[j]<=pivot)
		{
			i++;
			Exchange(&Address[i],&Address[j]);
		}
	}
	Exchange(&Address[end],&Address[i+1]);
	return i+1;
}

int randomized_partition(double *Address, int start, int end)
{
	int random_position;

	random_position=random_num(start,end);
	Exchange(&Address[random_position],&Address[end]);
	return partition(Address,start,end);
}

double randomized_selection(double *Address, int start, int end, int position)
{
	int random_position;
	int temp_rank;

	if (start==end)
		return Address[start];
	random_position=randomized_partition(Address,start,end);
	temp_rank=random_position-start+1;

	if (temp_rank>position+1)
		return randomized_selection(Address,start,random_position-1,position);
	else if (temp_rank<position+1)
		return randomized_selection(Address,random_position+1,end,position-temp_rank);
	else //if (pivot==position)
		return Address[random_position];
}

int search_index(double *Address, int length, double target_value)
{
	int i;
	for (i=0;i<length;i++)
	{
		if (target_value==Address[i])
		{
			return i;
		}
	}
	return -1;
}


int EBGM_FaceComparison(int total_trainface,int train_feature_count[], double train_gabor_feature[][500][41][2], 
						int probe_feature_count,double Probe_feature_Vectors[][41][2], int candidate_index[])
{
	int i,j,k,m;
	int MaxSimilarity_count[Total_train_face]={0};
	double OveralSimilarity[Total_train_face]={0.0};
	double temp_max_similarity=0.0;
	double point_peak_similarity=0.0;
	int max_similarity_index=0;

	double similarity=0.0;
	double similarity_matrix[500][Total_train_face]={0.0};
	double temp_MaxSimilarity_count=0.0,temp_feature_count=0.0;

	double thresh_distance=8.0;
	double probe_xcoordiante;
	double probe_ycoordiante;
	double train_xcoordiante;
	double train_ycoordiante;
	double distance=0.0;
	double last_similarity=0.0;
	double dotproduct=0.0;
	double train_feature_sqr=0.0;
	double probe_feature_sqr=0.0;
	int temp_index,real_index;
	int new_candidate_index[Total_train_face];
	double copy_OveralSimilarity[Total_train_face]={0.0};
	double temp_OveralSimilarity;

	double temp_Similarity;
	double copy_Similarity[Total_train_face]={0.0};
	double index_Similarity[Total_train_face]={0.0};


	for (i=0;i<probe_feature_count;i++)
	{
		probe_xcoordiante=Probe_feature_Vectors[i][0][0];
		probe_ycoordiante=Probe_feature_Vectors[i][0][1];
		for (j=0;j<total_trainface;j++)
		{
			last_similarity=0.0;
			for (k=0;k<train_feature_count[j];k++)
			{
				similarity=0.0;
				train_xcoordiante=train_gabor_feature[j][k][0][0];
				train_ycoordiante=train_gabor_feature[j][k][0][1];
				distance=sqrt((train_xcoordiante-probe_xcoordiante)*(train_xcoordiante-probe_xcoordiante)+(train_ycoordiante-probe_ycoordiante)*(train_ycoordiante-probe_ycoordiante));
				if (distance<=thresh_distance)
				{
					dotproduct=0.0;
					train_feature_sqr=0.0;
					probe_feature_sqr=0.0;
					for (m=1;m<41;m++)  //the first one is position information
					{
						dotproduct=dotproduct+sqrt(complex_modulus(train_gabor_feature[j][k][m]))*sqrt(complex_modulus(Probe_feature_Vectors[i][m]));
						train_feature_sqr = train_feature_sqr + complex_modulus(train_gabor_feature[j][k][m]);
						probe_feature_sqr = probe_feature_sqr + complex_modulus(Probe_feature_Vectors[i][m]);
					}
					similarity=dotproduct/sqrt(train_feature_sqr*probe_feature_sqr);
					if (similarity>last_similarity)
					{
						similarity_matrix[i][j]=similarity;
						last_similarity=similarity;
					}
				}
			}
		}
	}

	for (i=0;i<probe_feature_count;i++)
	{
		memcpy(copy_Similarity,similarity_matrix[i],total_trainface*sizeof(double));
		memcpy(index_Similarity,similarity_matrix[i],total_trainface*sizeof(double));
		for (j=total_trainface;j>total_trainface-1;j--)
		{
			temp_Similarity=randomized_selection(copy_Similarity,0,total_trainface-1,j-1);
			temp_index=search_index(index_Similarity,total_trainface,temp_Similarity);
			MaxSimilarity_count[temp_index]++;
		}
	}

	for (i=0;i<total_trainface;i++)
	{
		temp_MaxSimilarity_count=MaxSimilarity_count[i];
		temp_feature_count=train_feature_count[i];
		OveralSimilarity[i]= temp_MaxSimilarity_count/temp_feature_count;
	}

	memcpy(copy_OveralSimilarity,OveralSimilarity,Total_train_face*sizeof(double));

 	for (i=total_trainface;i>0;i--) //Only find the active rankings
	{
		temp_OveralSimilarity=randomized_selection(copy_OveralSimilarity,0,total_trainface-1,i-1);
		temp_index=search_index(OveralSimilarity,total_trainface,temp_OveralSimilarity);
		real_index=candidate_index[temp_index];
		new_candidate_index[total_trainface-i]=real_index;
	}
	memcpy(candidate_index,new_candidate_index,total_trainface*sizeof(int));

	max_similarity_index=new_candidate_index[0]; //The first one is the index of the max
	return max_similarity_index;
}