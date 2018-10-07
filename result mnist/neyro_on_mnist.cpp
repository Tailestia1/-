#include "fstream"
#include "iostream"
#include "math.h"
#include <conio.h>
#include <ctype.h>
#include <random>
using namespace std;
/*���� MNIST, 60��� �����������, 50-��� ��������, 10 - �������� ���������. 1������=1�������� 28*28�������� � ����� ������(��� ���������� �� ��������) 
1)����� �������� �� 48���, � 2��� ��������� ��� �������� �����������. ������ ��� ����� �������� �� � ������ ������
2)����� ������� ������� ����� ������� ���������, �� ��������� �� ������ ��� ����������� 10(��������)���������� � �������� �������
�� ������� ����������� ������ �������� RMS, adam, adagrad
3)�������� �������� ���������������� ������, ���������� ���������, ������������� ������� ����� �������������
*/
const int steps = 150;// steps - ������������ ���������� ���� ��������.
const int in_size = 784, meadle_size = 15, out_size = 10;//������� �����: ����, ����������, �������� (�������� ������� �� ������� � ������ ����� 784)
const int valid_size = 2000, test_size = 10000, base_size = 60000;
/* ����� ����������� �������. ������ ��������(��������, ���������) �������,
������ ����� �������������� ���������, ������� �������� ��������� ����������, ��� �� ���������� � ������ �������� ��� ���������*/
const double n = 0.001,// �������� ��������,
	gamma1 = 0.99, gamma = 0.999, eps = pow (10.0, -8),//  �������� ��������� ���������, ����������� �������� (RMS)
	beta1 = 0.9, beta2 = 0.999; //��������� ����������������� ��������� ��� ������ M D (Adam)
int valid_cur = base_size-valid_size-test_size, int_label[base_size], d_count=0, d_count_max=1; 
/*������ ������ ����� ���������. ���� ��������� � ������, �� ��������� ������ ���� ���: �������_������� �� ������� ���������, 
+������ ���������, ���� ���������� ����� ������ ������ ������� ����, �� ��� ��������� ����� ����� �����.
int_label - ������ ����� �������. �������� ��� ������ � ������������ ���������� �� ���������
������� ��� ��������� ����������, time - ������� ���������� ���������� ��� ���������� (����)*/
unsigned char input[base_size][28][28], label[base_size]; 
double int_input[base_size][in_size], weight1[meadle_size][in_size], weight2[out_size][meadle_size], b2[meadle_size], b3[out_size], w_start[meadle_size][in_size], w2_start[out_size][meadle_size],
/*��� ���� ���������, ���� ����� ������� ����� � �������, ���� �� �������� ���� �� �������,
�������� �� ������� ����, �������� �� ������� ����(������������ ��� ���������, ��� �� �����������, ���������� ����� ����)*/
sum1[meadle_size], sum2[out_size], act[meadle_size], out[out_size], answer[base_size][out_size],  b2_start[meadle_size], b3_start[out_size],
	/*sum1 - ������ ���� ��� �������� ����, sum2 - ������ ���� ��� ���������� ����, act - ������ �������� �������� (������� ����)
	out - �������� ������, answer - ������ ������� (������� ���������? 0 � 1)*/
  d_weight1[meadle_size][in_size], d_weight2[out_size][meadle_size], d_b3[out_size], d_b2[meadle_size], 
/*���������� ��������� ��� ����� � �������, ������� ����������-d_count*/
dE_g1[meadle_size][in_size], dE_g2[out_size][meadle_size], dE_b3[out_size], dE_b2[meadle_size],
/*������� ����������������� ��������� ��� ��������������� ����������*/
weight1_min[meadle_size][in_size], weight2_min[out_size][meadle_size], b2_min[meadle_size], b3_min[out_size], min_err=100,
 /*���� � ������ ��� ����������� ���������� ������*/
 dM_g1[meadle_size][in_size], dM_g2[out_size][meadle_size], dM_b3[out_size], dM_b2[meadle_size];
/*dM- ���������� ����������������� ��������� ��� ����������� ������ �������*/

double dE_rms_g1[meadle_size][in_size], dE_rms_g2[out_size][meadle_size], dE_rms_b3[out_size], dE_rms_b2[meadle_size];//��������� ������������ ������������� � ���� �������� 7
ifstream file_labels("train-labels.idx1-ubyte", ios::binary);
ifstream file("train-images.idx3-ubyte", ios::binary);

//-----������������� ����� � ������� (b)----
void Ini() 
{
    memset (b2, 0, sizeof(b2));
	memset (b3, 0, sizeof(b3));
    memset (answer, 0, sizeof(answer));
	memset (d_weight1, 0, sizeof(d_weight1));
    memset (d_weight2, 0, sizeof(d_weight2));
	memset (d_b2, 0, sizeof(d_b2));
	memset (d_b3, 0, sizeof(d_b3));
	memset (dE_g1, 0.1, sizeof(dE_g1));
    memset (dE_g2, 0.1, sizeof(dE_g2));
	memset (dE_b3, 0.1, sizeof(dE_b3));
    memset (dE_b2, 0.1, sizeof(dE_b2));
	
	memset (dM_g1, 0.01, sizeof(dE_g1));
    memset (dM_g2, 0.01, sizeof(dE_g2));
	memset (dM_b3, 0.01, sizeof(dE_b3));
    memset (dM_b2, 0.01, sizeof(dE_b2));
	/*typetech =7*/
	memset(dE_rms_g1, 0.1, sizeof(dE_rms_g1));
	memset(dE_rms_g2, 0.1, sizeof(dE_rms_g2));
	memset(dE_rms_b3, 0.1, sizeof(dE_rms_b3));
	memset(dE_rms_b2, 0.1, sizeof(dE_rms_b2));
	for (int i = 0; i < meadle_size; i++)//������������� ����� �� ������� ���� �� ������ � ������ �������� �� ���������� ����
	{
		b2[i] = 0.01 *  rand() / RAND_MAX -0.1;
		for (int j = 0; j < in_size; j++)
			weight1[i][j] = 0.1 *  rand() / RAND_MAX ;
	}
	for (int i = 0; i < out_size; i++)//������������� ����� �� ����������� ���� � ������, �������� � ������ �������� �� ������� ����
	{
		b3[i] =0.01 *  rand() / RAND_MAX -0.1;
		for (int j = 0; j < meadle_size; j++)
	        weight2[i][j] = 0.1 * rand() / RAND_MAX ;
	}
}

//-----���������� � ����������� ���� ����� �� mnist
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;
 
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
 
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

//���������� ���� ������
void Read_mnist() 
{
	if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        int labels_magic_number = 0;
        int number_of_items = 0;
        file_labels.read((char*)&labels_magic_number,sizeof(labels_magic_number)); 
        labels_magic_number = reverseInt(labels_magic_number);
        file_labels.read((char*)&number_of_items,sizeof(number_of_items));
        number_of_items = reverseInt(number_of_items);
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols = reverseInt(n_cols); 
        file.read((char*)&input,sizeof(input));
        file_labels.read((char*)&label,sizeof(label));
		n_cols = 28; n_rows = 28;//�������� �� ��� �� �� ��������� ��
        for (int i = 0; i < number_of_items; i++)
        {       
			int in_i = 0;
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c, in_i++)
                    int_input[i][in_i] = (double)input[i][r][c] / 255.0;
            }               
            int_label[i] = (int)label[i];
			answer[i][int_label[i]] = 1;
        }
		 file.close();
		 file_labels.close();
    }
	else
		cout << "Error of read mnist" << endl;
}

//-----������ ������ �������� ���� ������ ����� ��������� ��������� ����
void Set_Network(int g) 
{
    memset (sum1, 0, sizeof(sum1));//����� ����� � ������� � �������� �����
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//������ ��� ����� �������� ����������� ����
    {
        for (int i = 0; i < in_size; i++)
            sum1[t] += weight1[t][i] * int_input[g][i];//������������� ����� �� �������� ������� � ���� �������� � ������������ ������ ����������� ����
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t])));//���������� ������� ��������� �������. ����� = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// ���������� ����� �� ������� ����(���������� ��������� ����� ���������� �� ������� ���������� ����
			sum2[i] += weight2[i][t] * act[t];//����� ������� ��� ����� ������������� ������������ �� ���� ��������
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i])));//��������� �������� ���������� ����
	//��� ��������� �������� �������� ����� ������� � �������������� � ������
}

//���������� ��� ��������� ���������, ����� � ��������� ������������� ������ �������� (� ������� ���������� �������).
//���������� ���������� 4 ����, ��� ���� � � ��� ���� �����
void Set_Networkb2(int g) 
{
    memset (sum1, 0, sizeof(sum1));//����� ����� � ������� � �������� �����
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//������ ��� ����� �������� ����������� ����
    {
        for (int i = 0; i < in_size; i++)
            sum1[t] += weight1[t][i] * int_input[g][i];//������������� ����� �� �������� ������� � ���� �������� � ������������ ������ ����������� ����
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t]-gamma*dE_b2[t])));//���������� ������� ��������� �������. ����� = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// ���������� ����� �� ������� ����(���������� ��������� ����� ���������� �� ������� ���������� ����
			sum2[i] += weight2[i][t] * act[t];//����� ������� ��� ����� ������������� ������������ �� ���� ��������
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i])));//��������� �������� ���������� ����
	//��� ��������� �������� �������� ����� ������� � �������������� � ������
}
void Set_Networkb3(int g) 
{
    memset (sum1, 0, sizeof(sum1));//����� ����� � ������� � �������� �����
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//������ ��� ����� �������� ����������� ����
    {
        for (int i = 0; i < in_size; i++)
            sum1[t] += weight1[t][i] * int_input[g][i];//������������� ����� �� �������� ������� � ���� �������� � ������������ ������ ����������� ����
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t])));//���������� ������� ��������� �������. ����� = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// ���������� ����� �� ������� ����(���������� ��������� ����� ���������� �� ������� ���������� ����
			sum2[i] += weight2[i][t] * act[t];//����� ������� ��� ����� ������������� ������������ �� ���� ��������
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i]-gamma*dE_b3[i])));//��������� �������� ���������� ����
	//��� ��������� �������� �������� ����� ������� � �������������� � ������
}
void Set_Networkw1(int g) 
{
    memset (sum1, 0, sizeof(sum1));//����� ����� � ������� � �������� �����
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//������ ��� ����� �������� ����������� ����
    {
        for (int i = 0; i < in_size; i++)
			sum1[t] += (weight1[t][i]-gamma*dE_g1[t][i]) * int_input[g][i];//������������� ����� �� �������� ������� � ���� �������� � ������������ ������ ����������� ����
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t])));//���������� ������� ��������� �������. ����� = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// ���������� ����� �� ������� ����(���������� ��������� ����� ���������� �� ������� ���������� ����
			sum2[i] += weight2[i][t] * act[t];//����� ������� ��� ����� ������������� ������������ �� ���� ��������
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i])));//��������� �������� ���������� ����
	//��� ��������� �������� �������� ����� ������� � �������������� � ������
}
void Set_Networkw2(int g) 
{
    memset (sum1, 0, sizeof(sum1));//����� ����� � ������� � �������� �����
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//������ ��� ����� �������� ����������� ����
    {
        for (int i = 0; i < in_size; i++)
            sum1[t] += weight1[t][i] * int_input[g][i];//������������� ����� �� �������� ������� � ���� �������� � ������������ ������ ����������� ����
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t])));//���������� ������� ��������� �������. ����� = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// ���������� ����� �� ������� ����(���������� ��������� ����� ���������� �� ������� ���������� ����
			sum2[i] += (weight2[i][t] - gamma*dE_g2[i][t]) * act[t];//����� ������� ��� ����� ������������� ������������ �� ���� ��������
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i])));//��������� �������� ���������� ����
	//��� ��������� �������� �������� ����� ������� � �������������� � ������
}

//-----�������� �������� �� ������ ������������ ������
void Teach(int g, int type, int time)//type: 0 - ������� �������� 1 - ��� 2-���� 3-�������
{//������ ����������� RMS � ���������� �������
	double dw = 0, dw2 = 0;
	double m_estim=0, v_estim=0;//���������� ��� ��������� ����, ������ M D
	d_count++;

	switch (type)
	{
	case 0://���������� ��� ���������� �������
	{
		for (int h = 0; h < meadle_size; h++)//����� ����������� ����
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//��������� ����� ����������� ��� ���������� ����� � �������. dw2 ����� ������������ ��� b, dw2*input ��� ����� � ����������� ��
			//������� ������
			d_b2[h] += dw2;//���������� ��� b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//��� ����������� h ��������� ��������� bh, ������ ��� ���� �� ������� ���� ��������� � h*/
			for (int i = 0; i < in_size; i++)//���������� �� ���� ����� �������� ���� ��������� � ���������� �������� ����������� ����
			{
				dw = dw2 * int_input[g][i];
				d_weight1[h][i] += dw;//���������� ����� ��� ����������� ����
				if (d_count == d_count_max)
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//��������� ����.
					d_weight1[h][i] = 0;
				}
			}
		}
		//���������� �� �������� ���� (���� � ���� � ������ � ���)
		for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//�������� ��� ���������� b , dw=dw2*act(h)-��� �����
			d_b3[m] += dw2;//���������� ����� ���������
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-��������� ���� (���� ������), ��� ����� ��������� � ���������� m
			{
				dw = dw2 * act[h];//������� ����������� ��� wmh(� m �� h)
				d_weight2[m][h] += dw;
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//��������� ����.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 1://RMS
	{	
		for (int h = 0; h < meadle_size; h++)//����� ����������� ����
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//��������� ����� ����������� ��� ���������� ����� � �������. dw2 ����� ������������ ��� b, dw2*input ��� ����� � ����������� ��
			//������� ������
			dE_b2[h] = gamma * dE_b2[h] + (1 - gamma) * dw2 * dw2;//���������������� ��������� ��� ������
			d_b2[h] += dw2 / sqrt(dE_b2[h] + eps);//���������� ��� b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//��� ����������� h ��������� ��������� bh, ������ ��� ���� �� ������� ���� ��������� � h*/
			for (int i = 0; i < in_size; i++)//���������� �� ���� ����� �������� ���� ��������� � ���������� �������� ����������� ����
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] = gamma * dE_g1[h][i] + (1 - gamma) * dw * dw;//���������������� ��������� ��� ����������� ����
				d_weight1[h][i] += dw / sqrt(dE_g1[h][i] + eps);//���������� ����� ��� ����������� ����
				if (d_count == d_count_max)
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//��������� ����.
					d_weight1[h][i] = 0;
				}
			}
		}
		//���������� �� �������� ���� (���� � ���� � ������ � ���)
		for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//�������� ��� ���������� b , dw=dw2*act(h)-��� �����
			dE_b3[m] = gamma * dE_b3[m] + (1 - gamma) * dw2 * dw2;//���������������� ��������� ��� B �������� ����
			d_b3[m] += dw2 / sqrt(dE_b3[m] + eps);//���������� ����� ���������
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-��������� ���� (���� ������), ��� ����� ��������� � ���������� m
			{
				dw = dw2 * act[h];//������� ����������� ��� wmh(� m �� h)
				dE_g2[m][h] = gamma * dE_g2[m][h] + (1 - gamma) * dw * dw;
				d_weight2[m][h] += dw / sqrt(dE_g2[m][h] + eps);
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//��������� ����.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 2://adam
	{
		double x1, x2;
		x1 = 1 - pow(beta1, time);
		x2 = 1 - pow(beta2, time);
		//� ��������� ������������ ���������������� ��������� ��� �������� ����������� (�������� E_b, E_g � ������) � ��������� �� ����������� (1� �������)(������� ���. ����������)
		for (int h = 0; h < meadle_size; h++)//����� ����������� ����
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//��������� ����� ����������� ��� ���������� ����� � �������. dw2 ����� ������������ ��� b, dw2*input ��� ����� � ����������� �� ������� ������
			//���������� ������
			dE_b2[h] = beta1 * dE_b2[h] + (1 - beta1) * dw2 * dw2;//���������������� ��������� ��� ������
			dM_b2[h] = beta2 * dM_b2[h] + (1 - beta2) * dw2;//��������� ��� ����������� ������ �������
			if(time <=10)	m_estim = dM_b2[h] / x1;
			else m_estim= dM_b2[h];
			if (time <=100) v_estim = dE_b2[h] / x2;
			else v_estim= dE_b2[h];
			d_b2[h] += m_estim / sqrt(v_estim + eps);//���������� ��� b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//��� ����������� h ��������� ��������� bh, ������ ��� ���� �� ������� ���� ��������� � h
			for (int i = 0; i < in_size; i++)//���������� �� ���� ����� �������� ���� ��������� � ���������� �������� ����������� ����
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] = beta1 * dE_g1[h][i] + (1 - beta1) * dw * dw;//���������������� ��������� ��� ����������� ����
				dM_g1[h][i] = beta2 * dM_g1[h][i] + (1 - beta2) * dw;
				if (time <10) m_estim = dM_g1[h][i] / x1;
				else m_estim= dM_g1[h][i];
				if (time <100) v_estim = dE_g1[h][i] / x2;
				else v_estim= dE_g1[h][i];
				d_weight1[h][i] += m_estim / sqrt(v_estim + eps);//���������� ����� ��� ����������� ����
				if (d_count == d_count_max)
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//��������� ����.
					d_weight1[h][i] = 0;
				}
			}
		}
		//���������� �� �������� ���� (���� � ���� � ������ � ���)
		for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//�������� ��� ���������� b , dw=dw2*act(h)-��� �����
			dE_b3[m] = beta1 * dE_b3[m] + (1 - beta1) * dw2 * dw2;//���������������� ��������� ��� ������
			dM_b3[m] = beta2 * dM_b3[m] + (1 - beta2) * dw2;//��������� ��� ����������� ������ �������
			if (time <10) m_estim = dM_b3[m] / x1;
			else m_estim= dM_b3[m];
			if (time <100) v_estim = dE_b3[m] / x2;
			else v_estim= dE_b3[m];
			
			d_b3[m] += m_estim / sqrt(v_estim + eps);//���������� ��� b
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-��������� ���� (���� ������), ��� ����� ��������� � ���������� m
			{
				dw = dw2 * act[h];//������� ����������� ��� wmh(� m �� h)

				dE_g2[m][h] = beta1 * dE_g2[m][h] + (1 - beta1) * dw * dw;//���������������� ��������� ��� ����������� ����
				dM_g2[m][h] = beta2 * dM_g2[m][h] + (1 - beta2) * dw;
				if (time <10) m_estim = dM_g2[m][h] / x1;
				else m_estim= dM_g2[m][h];
				if (time <100) v_estim = dE_g2[m][h] / x2;
				else v_estim= dE_g2[m][h];
				
				d_weight2[m][h] += m_estim / sqrt(v_estim + eps);//���������� ����� ��� ����������� ����
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//��������� ����.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 3://adagrad
	{
		for (int h = 0; h < meadle_size; h++)//����� ����������� ����
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//��������� ����� ����������� ��� ���������� ����� � �������. dw2 ����� ������������ ��� b, dw2*input ��� ����� � ����������� ��
			//������� ������
			dE_b2[h] = dE_b2[h] + dw2 * dw2;//���������� ���������� �������� ���������
			d_b2[h] += dw2 / (sqrt(dE_b2[h]) + eps);//���������� ��� b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//��� ����������� h ��������� ��������� bh, ������ ��� ���� �� ������� ���� ��������� � h*/
			for (int i = 0; i < in_size; i++)//���������� �� ���� ����� �������� ���� ��������� � ���������� �������� ����������� ����
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] += dw * dw;//���������������� ��������� ��� ����������� ����
				d_weight1[h][i] += dw / (sqrt(dE_g1[h][i]) + eps);//���������� ����� ��� ����������� ����
				if (d_count == d_count_max)
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//��������� ����.
					d_weight1[h][i] = 0;
				}
			}
		}
		//���������� �� �������� ���� (���� � ���� � ������ � ���)
		for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//�������� ��� ���������� b , dw=dw2*act(h)-��� �����
			dE_b3[m] += dw2 * dw2;//���������������� ��������� ��� B �������� ����
			d_b3[m] += dw2 / (sqrt(dE_b3[m]) + eps);//���������� ����� ���������
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-��������� ���� (���� ������), ��� ����� ��������� � ���������� m
			{
				dw = dw2 * act[h];//������� ����������� ��� wmh(� m �� h)
				dE_g2[m][h] += dw * dw;
				d_weight2[m][h] += dw / (sqrt(dE_g2[m][h]) + eps);
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//��������� ����.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 4://momentum
	{
		for (int h = 0; h < meadle_size; h++)//����� ����������� ����
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//��������� ����� ����������� ��� ���������� ����� � �������. dw2 ����� ������������ ��� b, dw2*input ��� ����� � ����������� ��
			//������� ������
			dE_b2[h] = gamma * dE_b2[h] + n * dw2;//���������������� ��������� ��� ������
			d_b2[h] += dE_b2[h];//���������� ��� b
			if (d_count == d_count_max)
			{
				b2[h] -= d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//��� ����������� h ��������� ��������� bh, ������ ��� ���� �� ������� ���� ��������� � h
			for (int i = 0; i < in_size; i++)//���������� �� ���� ����� �������� ���� ��������� � ���������� �������� ����������� ����
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] = gamma * dE_g1[h][i] + n * dw;//���������������� ��������� ��� ����������� ����
				d_weight1[h][i] += dE_g1[h][i];//���������� ����� ��� ����������� ����
				if (d_count == d_count_max)
				{
					weight1[h][i] -= d_weight1[h][i] / (1.0 * d_count_max);//��������� ����.
					d_weight1[h][i] = 0;
				}
			}
		}
		//���������� �� �������� ���� (���� � ���� � ������ � ���)
		for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//�������� ��� ���������� b , dw=dw2*act(h)-��� �����
			dE_b3[m] = gamma * dE_b3[m] + n * dw2;//���������������� ��������� ��� B �������� ����
			d_b3[m] += dE_b3[m];//���������� ����� ���������
			if (d_count == d_count_max)
			{
				b3[m] -= d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-��������� ���� (���� ������), ��� ����� ��������� � ���������� m
			{
				dw = dw2 * act[h];//������� ����������� ��� wmh(� m �� h)
				dE_g2[m][h] = gamma * dE_g2[m][h] + n * dw;
				d_weight2[m][h] += dE_g2[m][h];
				if (d_count == d_count_max)
				{
					weight2[m][h] -= d_weight2[m][h] / (1.0 * d_count_max);//��������� ����.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 5://nesterov ���� ������� � ������������ � ������ ������
	{
		Set_Networkb2(g);//�������� ��������� ��� �2
		for (int h = 0; h < meadle_size; h++)//����� ����������� ����
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//��������� ����� ����������� ��� ���������� ����� � �������. dw2 ����� ������������ ��� b, dw2*input ��� ����� � ����������� ��
			//������� ������
			dE_b2[h] = gamma * dE_b2[h] + n * dw2;//���������������� ��������� ��� ������
			d_b2[h] += dE_b2[h];//���������� ��� b
		}
		Set_Networkb3(g);//������� ��������� ��� �3
		for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//�������� ��� ���������� b , dw=dw2*act(h)-��� �����
			dE_b3[m] = gamma * dE_b3[m] + n * dw2;//���������������� ��������� ��� B �������� ����
			d_b3[m] += dE_b3[m];
		}
		Set_Networkw1(g);
		for (int h = 0; h < meadle_size; h++)//����� ����������� ����
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];
			for (int i = 0; i < in_size; i++)//���������� �� ���� ����� �������� ���� ��������� � ���������� �������� ����������� ����
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] = gamma * dE_g1[h][i] + n * dw;//���������������� ��������� ��� ����������� ����
				d_weight1[h][i] += dE_g1[h][i];
			}
		}
		Set_Networkw2(g);
		for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//�������� ��� ���������� b , dw=dw2*act(h)-��� �����
			for (int h = 0; h < meadle_size; h++)//h-��������� ���� (���� ������), ��� ����� ��������� � ���������� m
			{
				dw = dw2 * act[h];//������� ����������� ��� wmh(� m �� h)
				dE_g2[m][h] = gamma * dE_g2[m][h] + n * dw;
				d_weight2[m][h] += dE_g2[m][h];
			}
		}
		//�������� ��� �����������. ������ ��� ������������� �������
		if (d_count == d_count_max)
		{
			for (int h = 0; h < meadle_size; h++)//����� ����������� ����
			{
				b2[h] -= d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
				for (int i = 0; i < in_size; i++)
				{
					weight1[h][i] -= d_weight1[h][i] / (1.0 * d_count_max);//��������� ����.
					d_weight1[h][i] = 0;
				}
			}
			for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
			{
				b3[m] -= d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
				for (int h = 0; h < meadle_size; h++)//h-��������� ���� (���� ������), ��� ����� ��������� � ���������� m
				{
					weight2[m][h] -= d_weight2[m][h] / (1.0 * d_count_max);//��������� ����.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 6://adadelta
	{
		for (int h = 0; h < meadle_size; h++)//����� ����������� ����
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//��������� ����� ����������� ��� ���������� ����� � �������. dw2 ����� ������������ ��� b, dw2*input ��� ����� � ����������� �� ������� ������
			dE_b2[h] = gamma1 * dE_b2[h] + (1 - gamma1) * dw2 * dw2;//������� ���
			dw = 1.0 * sqrt(dM_b2[h] + eps) * dw2 / sqrt(dE_b2[h] + eps);//������ ����� �������� ����, ���������� ��� ���������� ���������� ����(����� �� ��� ����������� ����
			dM_b2[h] = gamma * dM_b2[h] + (1 - gamma) * dw * dw;
			//������� ��� ��� ������ ����
			d_b2[h] += dw;//���������� ��� b
			if (d_count == d_count_max)
			{
				b2[h] -= d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//��� ����������� h ��������� ��������� bh, ������ ��� ���� �� ������� ���� ��������� � h*/

			for (int i = 0; i < in_size; i++)//���������� �� ���� ����� �������� ���� ��������� � ���������� �������� ����������� ����
			{
				dw = dw2 * int_input[g][i];
				/*if(time == 1&&g==1)
				{
					dE_g1[h][i] = dw*dw;
					dw2 = -1.0 * sqrt(dM_g1[h][i] + eps) * dw / sqrt(dE_g1[h][i] + eps);
					 dM_g1[h][i] = dw2 * dw2;
				}
				else
				{*/
				dE_g1[h][i] = gamma1 * dE_g1[h][i] + (1 - gamma1) * dw * dw;//���������������� ��������� ��� ����������� ����
				dw2 = sqrt(dM_g1[h][i] + eps) * dw / sqrt(dE_g1[h][i] + eps);//������ ����� �������� ����, ���������� ��� ���������� ���������� ����(����� �� ��� ����������� ����
				dM_g1[h][i] = gamma * dM_g1[h][i] + (1 - gamma)* dw2 * dw2;
				//}//������� ��� ��� ������ ����*/
				d_weight1[h][i] += dw2;//���������� ��� b
				if (d_count == d_count_max)
				{
					weight1[h][i] -= d_weight1[h][i] / (1.0 * d_count_max);//��������� ����.
					//weight1[h][i] =dM_g1[h][i];
					d_weight1[h][i] = 0;
				}
			}
		}
		//���������� �� �������� ���� (���� � ���� � ������ � ���)
		for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//�������� ��� ���������� b , dw=dw2*act(h)-��� �����
			dE_b3[m] = gamma1 * dE_b3[m] + (1 - gamma1) * dw2 * dw2;//���������������� ��������� ��� B �������� ����
			dw = sqrt(dM_b3[m] + eps) * dw2 / sqrt(dE_b3[m] + eps);//������ ����� �������� ����, ���������� ��� ���������� ���������� ����(����� �� ��� ����������� ����
			dM_b3[m] = gamma * dM_b3[m] + (1 - gamma) * dw * dw;
			//}//������� ��� ��� ������ ����*/
			d_b3[m] += dw;// sqrt(dE_b3[m] + eps);//���������� ����� ���������
			if (d_count == d_count_max)
			{
				b3[m] -= d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}

			for (int h = 0; h < meadle_size; h++)//h-��������� ���� (���� ������), ��� ����� ��������� � ���������� m
			{
				dw = dw2 * act[h];//������� ����������� ��� wmh(� m �� h)

				dE_g2[m][h] = gamma1 * dE_g2[m][h] + (1 - gamma1) * dw * dw;//���������������� ��������� ��� ����������� ����
				dw2 = sqrt(dM_g2[m][h] + eps) * dw / sqrt(dE_g2[m][h] + eps);//������ ����� �������� ����, ���������� ��� ���������� ���������� ����(����� �� ��� ����������� ����
				dM_g2[m][h] = gamma * dM_g2[m][h] + (1 - gamma) * dw2 * dw2;
				//}//������� ��� ��� ������ ����*/
				d_weight2[m][h] += dw2;//���������� ��� b
				//if (m==0) ww[h] = dw2;
				if (d_count == d_count_max)
				{
					weight2[m][h] -= d_weight2[m][h] / (1.0 * d_count_max);//��������� ����.

					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 7:
	{/*��������������� ����� �������� ������� ������������. ��� �������� ������������ ��������� ����� ��������. � ������ ������ ����������� ��������� ��� ���������
	 ��� ������� �� ����� ��������. �� ���� ���� ����� �������� ����� ���, �� ��������� ���������� ��� �������, ��� ������� � ��� ��������
	 ������� � ������������ 1)������� (��� ��� � ���� ����� ������ ����� ����������) 2)��� ��� ��������� ������� 
	 3) �������� (��� ������������ ����������� �����������) [ �������� �� ����� ����� ��� ��� ����� �� ����� ������ ����������]
	 dE ����� ��������� ��� �������, dM for momentum, dE_rms for rms*/
		for (int h = 0; h < meadle_size; h++)//����� ����������� ����
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];
			/*��������� ����� ����������� ��� ���������� ����� � �������. dw2 ����� ������������ ��� b, dw2*input ��� ����� � ����������� �� ������� ������*/
			//��������� ��� �������
			dE_b2[h] = dE_b2[h] + dw2 * dw2;//���������� ���������� �������� ���������
			//��������� ��� ���������
			dM_b2[h] = gamma * dM_b2[h] + n * dw2;//���������������� ��������� ��� ������
			//rms
			dE_rms_b2[h] = gamma * dE_rms_b2[h] + (1 - gamma) * dw2 * dw2;//���������������� ��������� ��� ������
			if (time < 50)
				d_b2[h] += dw2 / (sqrt(dE_b2[h]) + eps);//adagrad
			if (time >= 50 && time < 100)
				d_b2[h] += dw2 / sqrt(dE_rms_b2[h] + eps);//rms
			if (time >= 100)
				d_b2[h] += dM_b2[h];//momentum

			if (d_count == d_count_max)//����������
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//��� ����������� h ��������� ��������� bh, ������ ��� ���� �� ������� ���� ��������� � h*/
			for (int i = 0; i < in_size; i++)//���������� �� ���� ����� �������� ���� ��������� � ���������� �������� ����������� ����
			{
				dw = dw2 * int_input[g][i];//��� ����������� ��� ����� ������� ����
				//��������� ��� �������
				dE_g1[h][i] += dw * dw;//���������������� ��������� ��� ����������� ����
				//��������� ��� ���������
				dM_g1[h][i] = gamma * dM_g1[h][i] + n * dw;//���������������� ��������� ��� ����������� ����
				//��������� ��� 
				dE_rms_g1[h][i] = gamma * dE_rms_g1[h][i] + (1 - gamma) * dw * dw;//���������������� ��������� ��� ����������� ����
				//���������� ����� ��� ����������� ����*/
				if (time < 50)
					d_weight1[h][i] += dw / (sqrt(dE_g1[h][i]) + eps);//�������
				if (time >= 50 && time < 100)
					d_weight1[h][i] += dw / sqrt(dE_rms_g1[h][i] + eps);//���
				if (time >= 100)
					d_weight1[h][i] += dM_g1[h][i];//��������

				if (d_count == d_count_max)//����������
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//��������� ����.
					d_weight1[h][i] = 0;
				}
			}
		}
		//���������� �� �������� ���� (���� � ���� � ������ � ���)
		for (int m = 0; m < out_size; m++)//�� ���������� ������� ��������
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//�������� ��� ���������� b , dw=dw2*act(h)-��� �����
			//��������� ��� �������
			dE_b3[m] += dw2 * dw2;//���������������� ��������� ��� B �������� ����
			//��������� ��� ���������
			dM_b3[m] = gamma * dM_b3[m] + n * dw2;//���������������� ��������� ��� B �������� ����
			//��������� rms
			dE_rms_b3[m] = gamma * dE_rms_b3[m] + (1 - gamma) * dw2 * dw2;//���������������� ��������� ��� B �������� ����

			if (time < 50)
				d_b3[m] += dw2 / (sqrt(dE_b3[m]) + eps);//�������
			if (time >= 50 && time < 100)
				d_b3[m] += dw2 / sqrt(dE_rms_b3[m] + eps);//���
			if (time >= 100)
				d_b3[m] += dM_b3[m];//��������
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-��������� ���� (���� ������), ��� ����� ��������� � ���������� m
			{
				dw = dw2 * act[h];//������� ����������� ��� wmh(� m �� h)
			    //��������� ��� �������
				dE_g2[m][h] += dw * dw;
				//��������� ��� ���������
				dM_g2[m][h] = gamma * dM_g2[m][h] + n * dw;
				//��������� rms 
				dE_rms_g2[m][h] = gamma * dE_rms_g2[m][h] + (1 - gamma) * dw * dw;
				
				if (time < 50)
					d_weight2[m][h] += dw / (sqrt(dE_g2[m][h]) + eps);
				if (time >= 50 && time < 100)
					d_weight2[m][h] += dw / sqrt(dE_rms_g2[m][h] + eps);//���
				if (time >= 100)
					d_weight2[m][h] += dM_g2[m][h];//�������
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//��������� ����.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	default:
		break;
	}
	if(d_count==d_count_max) d_count=0;
}

//-----���������� ������ ��� ������ �������� ���� ������
double Error(int g) 
{
    double s = 0;
    for (int i = 0; i < out_size; i++)
		s += (out[i] - answer[g][i]) * (out[i] - answer[g][i]);	
    return (s / 2.0);
}

//-----���������� ����� � ����
void save_weight(int x) 
{
	FILE *fp;
	fp = fopen ( "wei.txt","w");
	switch (x)
	{
	case 0: fp = fopen("weight0.csv", "w"); break;
	case 1: fp = fopen("weight1.csv", "w"); break;
	case 2: fp = fopen("weight2.csv", "w"); break;
	case 3: fp = fopen("weight3.csv", "w"); break;
	case 4: fp = fopen("weight4.csv", "w"); break;
	case 5: fp = fopen("weight5.csv", "w"); break;
	case 6: fp = fopen("weight6.csv", "w"); break;
	case 7: fp = fopen("weight7.csv", "w"); break;
	default: break;
	}
	for (int i = 0; i < meadle_size; i++)
	{
		fprintf(fp, "%.3f; ", b2[i]);
		for (int j = 0; j < in_size; j++)
			fprintf (fp, "%.3f; ", weight1[i][j]);
	}
	for (int i = 0; i < out_size; i++)
	{
		fprintf(fp, "%.3f; ", b3[i]);
		for (int j = 0; j < meadle_size; j++)
			fprintf (fp, "%.3f; ", weight2[i][j]);
	}
	fclose(fp);
}

//-----�������� ����� �� �����
void load_weight() 
{
	FILE *fp;
	float zzz = 0;
	fp = fopen ( "weight.csv","r");
	for (int i = 0; i < meadle_size; i++)
	{
		fscanf (fp, "%f;", &zzz);
		b2[i] =zzz;
		for (int j = 0; j < in_size; j++)
		{
			fscanf (fp, "%f;", &zzz);//weight1[i][j][l]);
			weight1[i][j] = zzz;
		}
	}
	for (int i = 0; i < out_size; i++)
	{
		fscanf (fp, "%f;", &zzz);
		b3[i] =zzz;
		for (int j = 0; j < meadle_size; j++)
		{
			fscanf (fp, "%f; ", &zzz);
			weight2[i][j] = zzz;
		}
	}
	fclose(fp);
}

//-----����������� �������� (������ ���������) ����� ��� ����������� ���������� ������ ��� ���������
void save_min()
{
	for (int h = 0; h < meadle_size; h++)
	{
		b2_min[h] = b2[h];
		for (int i = 0; i < in_size; i++)
			weight1_min[h][i]= weight1[h][i];
		for ( int m = 0; m < out_size; m ++)
			weight2_min[m][h] = weight2[m][h];
	}
	for (int m = 0; m < out_size; m++)
		b3_min[m] = b3[m];
}

//-----��������� ����  ��� ����������� ����������� ������
void load_min()
{
	for (int h = 0; h < meadle_size; h++)
	{
		b2[h] = b2_min[h];
		for (int i = 0; i < in_size; i++)
			weight1[h][i] = weight1_min[h][i];
		for ( int m = 0; m < out_size; m ++)
			weight2[m][h] = weight2_min[m][h];
	}
	for (int m = 0; m < out_size; m++)
		b3[m] = b3_min[m];
}

//-----��������� ��������� ������������� ����� ��� �� ����� ��� ����� �������� ���� �� �� �����
void save_start()
{
	for (int h = 0; h < meadle_size; h++)
	{
		b2_start[h] = b2[h];
		for (int i = 0; i < in_size; i++)
			w_start[h][i] = weight1[h][i];
		for (int m = 0; m < out_size; m++)
			w2_start[m][h] = weight2[m][h];
	}
	for (int m = 0; m < out_size; m++)
		b3_start[m] = b3[m];
}

//-----��������� ��������� ������������������ ��� ��� �� �� ��� ����� ��������
void load_start()
{
	for (int h = 0; h < meadle_size; h++)
	{
		b2[h] = b2_start[h];
		for (int i = 0; i < in_size; i++)
			weight1[h][i] = w_start[h][i];
		for (int m = 0; m < out_size; m++)
			weight2[m][h] = w2_start[m][h];
	}
	for (int m = 0; m < out_size; m++)
		b3[m] = b3_start[m];
}

//-----������� � ����� ������� �������� ���� ��� �������
int Rez()  // ����� ���������� ������ ���� ������ �� ����������� ��������� ������� out
{
    double max = -1;
    int max_index = -1;
    for (int i = 0; i < out_size; i++)
        if (out[i] > max)
        {
            max = out[i];
            max_index = i;
        }
    return max_index;
}

int main()
{
	setlocale(LC_ALL,"");
    Ini();//������������� �������� w � ���
	/*memset(w_start, 0, sizeof(w_start));
	memset(w2_start, 0, sizeof(w2_start));
	memset(b2_start, 0, sizeof(b2_start));
	memset(b3_start, 0, sizeof(b3_start));
	save_start();//*/

    Read_mnist();
	FILE *gr_err, *f1, *z1;
	int res = 0, res2 = 0, ans =0, type_teach=2;
	//type: 0- 1 -rmsprop 2 - adam 3- adagrad 4-momentu 5- ��������(?) 6- ���������
    while (ans != 4) 
    {
        system("cls");
        cout << "1 - �������� ����;" << endl;//��������
        cout << "2 - ������ ��������� ���������;" << endl;//�������� �������
        cout << "3 - �������� ����������� �����;" << endl;
        cout << "4 - ����� (��������� ������)." << endl << endl;
        cin >> ans;
        bool stop = true;//����
        double error_arr[4] = {9999,9999,9999,9999}, error_of_step = 0, err_of_res=0;
        int number_step = 1, z = 0; // number_step - ����� ����� ��������
        switch (ans)
        {
        case 1:
			gr_err = fopen("graf_of_error.csv", "w");
			/*for (type_teach = 0; type_teach < 7; type_teach++)
			{
				
				double error_arr[4] = { 9999,9999,9999,9999 }, error_of_step = 0, err_of_res = 0;
				number_step = 1; z = 0; stop = true;
				
				switch (type_teach)
				{
				case 0: gr_err = fopen("graf_of_error0.csv", "w"); break;
				case 1: gr_err = fopen("graf_of_error1.csv", "w");  break;
				case 2: gr_err = fopen("graf_of_error2.csv", "w");  break;
				case 3: gr_err = fopen("graf_of_error3.csv", "w");  break;
				case 4: gr_err = fopen("graf_of_error4.csv", "w");  break;
				case 5: gr_err = fopen("graf_of_error5.csv", "w");  break;
				case 6: gr_err = fopen("graf_of_error6.csv", "w");  break;//Ini(); load_start();
				default: break;
				}//*/
				while (stop && number_step < steps)  // ����� ��������. ���� - ����������� ���������� ����
				{
					error_of_step = 0; err_of_res = 0;
					cout << endl << endl << "!!!  " << number_step << " (" << type_teach << ") " << "  !!!" << endl << endl;//����� �����
					int curr = 0;
					for (; curr < valid_cur; curr++)  // �������� �� ���� � �� ������������� �����. ������������� ���������� � ��������� �� ����������
					{
						Set_Network(curr);//������ ���������� � ��������� ����������� �����
						Teach(curr, type_teach, number_step);//����������� ��������
						if (curr % 1000 == 0)
							cout << curr / 1000 << " ";
					}
					curr += valid_size;
					for (; curr < base_size - test_size; curr++)  //������������� ���������� � ��������� �� ����������
					{
						Set_Network(curr);//������ ���������� � ��������� ����������� �����
						Teach(curr, type_teach, number_step);//����������� ��������
						if (curr % 1000 == 0)
							cout << curr / 1000 << " ";
					}
					cout << endl << endl;

					//���������
					for (int j = valid_cur; j < valid_size + valid_cur; j++) // ����� �� �� ��������� (������� ������),������������ 1000��������� � ������ ������ ������ �����
					{
						Set_Network(j);//������ � ������� ������
						error_of_step += Error(j);//����� ������ �� ������� ������.
						res = Rez();//����� �� �����, �����
						if (int_label[j] == res) res2 = 0;//��������� ��� ��� � ���������� ������
						else res2 = 1;//������
						err_of_res += res2;
					}
					error_arr[0] = error_arr[1];
					error_arr[1] = error_arr[2];
					error_arr[2] = error_arr[3];
					error_arr[3] = error_of_step / valid_size;//��������� ��� ������� 
					err_of_res /= valid_size;
					cout << endl << endl << error_arr[3] << " ";//������ �� ����� � � ����
					cout << err_of_res;
					fprintf(gr_err, "%.5f;%.5f\n", error_arr[3], err_of_res);
					if (err_of_res < min_err)
					{
						min_err = err_of_res;
						save_min();
					}
					if ((error_arr[0] < error_arr[1]) && (error_arr[1] < error_arr[2]) && (error_arr[2] < error_arr[3]) && (number_step >= 150)) // ���� ������ ��������� ����� ������������� - ��������� ��������
						stop = false;
					number_step++;
					valid_cur -= valid_size;
					if (valid_cur < 0) valid_cur = base_size - valid_size - test_size;
				}
				load_min();
				//cout << endl << "�������� ���������, ���� ��������� �������������." << endl;
				save_weight(type_teach);
				fclose(gr_err);// fclose(z1);
			//}
			system("pause");
			break;
        case 2://������, �������� �� �������� �������
			f1 = fopen ("analiz.csv", "w");
			for (int i = base_size-test_size; i < base_size; i++)//�������� ���������� ����. �� 45 �� 60��� ��������
			{
				Set_Network(i);
				res = Rez();//����� �� �����, �����
				if (int_label[i] == res) res2=0;//��������� ��� ��� � ���������� ������
				else res2 = 1;//������
				cout << "�����: " << res << endl;//����� ������ ����
				fprintf (f1, "%d ; %d ; %d ;\n", int_label[i], res, res2);//���������� ������ � ����
			}
			fclose(f1);
            system("pause");
            break;
        case 3:
            load_weight();
            break;
        default:
            break;
        }
    }
    return 0;
}