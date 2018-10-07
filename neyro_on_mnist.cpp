#include "fstream"
#include "iostream"
#include "math.h"
#include <conio.h>
#include <ctype.h>
#include <random>
using namespace std;
/*база MNIST, 60тыс изображений, 50-для обучения, 10 - тестовое множество. 1объект=1картинке 28*28пикселей и метка класса(что изображено на картинке) 
1)будем проходит по 48тыс, а 2тыс оставлять как проверка обучаемости. Каждый раз будем вырезать ее в разных местах
2)после каждого объекта будем считать градиенты, но обновлять их только при наклоплении 10(например)градиентов и вычитать среднее
по заданию исследуются методы обучения RMS, adam, adagrad
3)движемся обратным распространением ошибок, вычисление градиента, активационная функция везде сигмоидальная
*/
const int steps = 150;// steps - максимальное количество эпох обучения.
const int in_size = 784, meadle_size = 15, out_size = 10;//размеры сетки: вход, внутренний, выходной (картинку вытянем из мтарицы в строку длины 784)
const int valid_size = 2000, test_size = 10000, base_size = 60000;
/* объем проверочной выборки. размер тестовой(конечной, отдельной) выборки,
размер всего тренировочного множества, которое придется загрузить польностью, что бы обращаться к разным объектам при валидации*/
const double n = 0.001,// скорость обучения,
	gamma1 = 0.99, gamma = 0.999, eps = pow (10.0, -8),//  параметр экспоненц забывания, глаживающий параметр (RMS)
	beta1 = 0.9, beta2 = 0.999; //параметры экспоненциального забывания для оценки M D (Adam)
int valid_cur = base_size-valid_size-test_size, int_label[base_size], d_count=0, d_count_max=1; 
/*маркер начала части валидации. если валидация в центре, то обработка данных идет так: отстади_маркера до маркера валидации, 
+размер валидации, если полученное число меньше общего размера базы, то еще обучаемся после этого числа.
int_label - список меток классов. счетчики для спуска и максимальное анкопление по градиенту
счетчик для пакетного обновления, time - счетчик количества обновлений для епременной (адам)*/
unsigned char input[base_size][28][28], label[base_size]; 
double int_input[base_size][in_size], weight1[meadle_size][in_size], weight2[out_size][meadle_size], b2[meadle_size], b3[out_size], w_start[meadle_size][in_size], w2_start[out_size][meadle_size],
/*вся база признаков, веса между входным слоем и среднем, веса из среднего слоя во внешний,
смещения на среднем слое, смещения на внешнем слое(используются при активации, так же обновляются, изначально равно нулю)*/
sum1[meadle_size], sum2[out_size], act[meadle_size], out[out_size], answer[base_size][out_size],  b2_start[meadle_size], b3_start[out_size],
	/*sum1 - массив сумм для скрытого слоя, sum2 - массив сумм для последнего слоя, act - массив значений сигмоиды (скрытый слой)
	out - выходной вектор, answer - массив ответов (заранее известных? 0 и 1)*/
  d_weight1[meadle_size][in_size], d_weight2[out_size][meadle_size], d_b3[out_size], d_b2[meadle_size], 
/*накопления градиента для весов и сдвигов, счетчик накоплений-d_count*/
dE_g1[meadle_size][in_size], dE_g2[out_size][meadle_size], dE_b3[out_size], dE_b2[meadle_size],
/*массивы экспоненциального забывания для соответствующих переменных*/
weight1_min[meadle_size][in_size], weight2_min[out_size][meadle_size], b2_min[meadle_size], b3_min[out_size], min_err=100,
 /*веса и сдвиги при минимальном показателе ошибки*/
 dM_g1[meadle_size][in_size], dM_g2[out_size][meadle_size], dM_b3[out_size], dM_b2[meadle_size];
/*dM- переменные экспоненциального забывания для производной первой степени*/

double dE_rms_g1[meadle_size][in_size], dE_rms_g2[out_size][meadle_size], dE_rms_b3[out_size], dE_rms_b2[meadle_size];//параметры используются дополнительно в типе обучения 7
ifstream file_labels("train-labels.idx1-ubyte", ios::binary);
ifstream file("train-images.idx3-ubyte", ios::binary);

//-----Инициализация весов и сдвигов (b)----
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
	for (int i = 0; i < meadle_size; i++)//инициализация весов из первого слоя во второй и сдвига нейронов на внутреннем слое
	{
		b2[i] = 0.01 *  rand() / RAND_MAX -0.1;
		for (int j = 0; j < in_size; j++)
			weight1[i][j] = 0.1 *  rand() / RAND_MAX ;
	}
	for (int i = 0; i < out_size; i++)//инициализация весов из внутреннего слоя в третий, выходной и сдвига нейронов на внешнем слое
	{
		b3[i] =0.01 *  rand() / RAND_MAX -0.1;
		for (int j = 0; j < meadle_size; j++)
	        weight2[i][j] = 0.1 * rand() / RAND_MAX ;
	}
}

//-----Приведение к нормальному виду чисел из mnist
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;
 
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
 
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

//Считывание базы данных
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
		n_cols = 28; n_rows = 28;//хитрость тк сам он не считывает их
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

//-----Прогон одного элемента базы данных через имеющуюся нейронную сеть
void Set_Network(int g) 
{
    memset (sum1, 0, sizeof(sum1));//суммы весов в скрытом и последне слоях
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//меньше чем число нейронов внутреннего слоя
    {
        for (int i = 0; i < in_size; i++)
            sum1[t] += weight1[t][i] * int_input[g][i];//накапливается сумма от входного пикселя и веса перехода в определенный нейрон внутреннего слоя
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t])));//сигмоидная функция активация нейрона. равна = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// накопление суммы на внешнем слое(полученную активацию сразу рассеиваем на нейроны следующего слоя
			sum2[i] += weight2[i][t] * act[t];//таким образом там сумма накапливается одновременно во всех нейронах
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i])));//активация нейронов последнего слоя
	//для вошедшего элемента получили набор ответов о пренадлежности к классу
}

//дополнения для алгоритма нестерова, когда в высиления подставляется другое значение (с вычитом известного вектора).
//вычисления проводятся 4 раза, для двух Б и для двух весов
void Set_Networkb2(int g) 
{
    memset (sum1, 0, sizeof(sum1));//суммы весов в скрытом и последне слоях
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//меньше чем число нейронов внутреннего слоя
    {
        for (int i = 0; i < in_size; i++)
            sum1[t] += weight1[t][i] * int_input[g][i];//накапливается сумма от входного пикселя и веса перехода в определенный нейрон внутреннего слоя
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t]-gamma*dE_b2[t])));//сигмоидная функция активация нейрона. равна = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// накопление суммы на внешнем слое(полученную активацию сразу рассеиваем на нейроны следующего слоя
			sum2[i] += weight2[i][t] * act[t];//таким образом там сумма накапливается одновременно во всех нейронах
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i])));//активация нейронов последнего слоя
	//для вошедшего элемента получили набор ответов о пренадлежности к классу
}
void Set_Networkb3(int g) 
{
    memset (sum1, 0, sizeof(sum1));//суммы весов в скрытом и последне слоях
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//меньше чем число нейронов внутреннего слоя
    {
        for (int i = 0; i < in_size; i++)
            sum1[t] += weight1[t][i] * int_input[g][i];//накапливается сумма от входного пикселя и веса перехода в определенный нейрон внутреннего слоя
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t])));//сигмоидная функция активация нейрона. равна = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// накопление суммы на внешнем слое(полученную активацию сразу рассеиваем на нейроны следующего слоя
			sum2[i] += weight2[i][t] * act[t];//таким образом там сумма накапливается одновременно во всех нейронах
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i]-gamma*dE_b3[i])));//активация нейронов последнего слоя
	//для вошедшего элемента получили набор ответов о пренадлежности к классу
}
void Set_Networkw1(int g) 
{
    memset (sum1, 0, sizeof(sum1));//суммы весов в скрытом и последне слоях
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//меньше чем число нейронов внутреннего слоя
    {
        for (int i = 0; i < in_size; i++)
			sum1[t] += (weight1[t][i]-gamma*dE_g1[t][i]) * int_input[g][i];//накапливается сумма от входного пикселя и веса перехода в определенный нейрон внутреннего слоя
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t])));//сигмоидная функция активация нейрона. равна = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// накопление суммы на внешнем слое(полученную активацию сразу рассеиваем на нейроны следующего слоя
			sum2[i] += weight2[i][t] * act[t];//таким образом там сумма накапливается одновременно во всех нейронах
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i])));//активация нейронов последнего слоя
	//для вошедшего элемента получили набор ответов о пренадлежности к классу
}
void Set_Networkw2(int g) 
{
    memset (sum1, 0, sizeof(sum1));//суммы весов в скрытом и последне слоях
    memset (sum2, 0, sizeof(sum2));
    for (int t = 0; t < meadle_size; t++)//меньше чем число нейронов внутреннего слоя
    {
        for (int i = 0; i < in_size; i++)
            sum1[t] += weight1[t][i] * int_input[g][i];//накапливается сумма от входного пикселя и веса перехода в определенный нейрон внутреннего слоя
		act[t] = 1 / (1 + exp(-1.0 * (sum1[t] + b2[t])));//сигмоидная функция активация нейрона. равна = 1/(1+exp(-z))
        for (int i = 0; i < out_size; i++)// накопление суммы на внешнем слое(полученную активацию сразу рассеиваем на нейроны следующего слоя
			sum2[i] += (weight2[i][t] - gamma*dE_g2[i][t]) * act[t];//таким образом там сумма накапливается одновременно во всех нейронах
	}
    for (int i = 0; i < out_size; i++)
		out[i] = 1 / (1 + exp(-1.0 * (sum2[i] + b3[i])));//активация нейронов последнего слоя
	//для вошедшего элемента получили набор ответов о пренадлежности к классу
}

//-----Алгоритм обучения на основе градиентного спуска
void Teach(int g, int type, int time)//type: 0 - обычное обучение 1 - рмс 2-адам 3-адаград
{//сейчас реализовано RMS и накопление пачками
	double dw = 0, dw2 = 0;
	double m_estim=0, v_estim=0;//переменные для алгоритма адам, оценка M D
	d_count++;

	switch (type)
	{
	case 0://обновления без ускоряющих методов
	{
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от
			//индекса инпута
			d_b2[h] += dw2;//накопления для b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//для конкретного h закончили обновлять bh, теперь все веса из первого слоя связанные с h*/
			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
			{
				dw = dw2 * int_input[g][i];
				d_weight1[h][i] += dw;//накопление суммы для конкретного веса
				if (d_count == d_count_max)
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//изменения веса.
					d_weight1[h][i] = 0;
				}
			}
		}
		//обновления по внешнему слою (веса в него и сдвиги в нем)
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			d_b3[m] += dw2;//накопление суммы градиента
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				d_weight2[m][h] += dw;
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 1://RMS
	{	
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от
			//индекса инпута
			dE_b2[h] = gamma * dE_b2[h] + (1 - gamma) * dw2 * dw2;//экспоненциальное забывание для сдвига
			d_b2[h] += dw2 / sqrt(dE_b2[h] + eps);//накопления для b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//для конкретного h закончили обновлять bh, теперь все веса из первого слоя связанные с h*/
			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] = gamma * dE_g1[h][i] + (1 - gamma) * dw * dw;//экспоненциальное забывание для конкретного веса
				d_weight1[h][i] += dw / sqrt(dE_g1[h][i] + eps);//накопление суммы для конкретного веса
				if (d_count == d_count_max)
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//изменения веса.
					d_weight1[h][i] = 0;
				}
			}
		}
		//обновления по внешнему слою (веса в него и сдвиги в нем)
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			dE_b3[m] = gamma * dE_b3[m] + (1 - gamma) * dw2 * dw2;//экспоненциальное забывание для B внешнего слоя
			d_b3[m] += dw2 / sqrt(dE_b3[m] + eps);//накопление суммы градиента
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				dE_g2[m][h] = gamma * dE_g2[m][h] + (1 - gamma) * dw * dw;
				d_weight2[m][h] += dw / sqrt(dE_g2[m][h] + eps);
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
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
		//в алгоритме используется экспоненциальное забывание для квадрата производной (возьмеме E_b, E_g и прочие) и забывание по производной (1я степень)(возьмем доп. переменные)
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от индекса инпута
			//построение оценки
			dE_b2[h] = beta1 * dE_b2[h] + (1 - beta1) * dw2 * dw2;//экспоненциальное забывание для сдвига
			dM_b2[h] = beta2 * dM_b2[h] + (1 - beta2) * dw2;//забывание для производной первой степени
			if(time <=10)	m_estim = dM_b2[h] / x1;
			else m_estim= dM_b2[h];
			if (time <=100) v_estim = dE_b2[h] / x2;
			else v_estim= dE_b2[h];
			d_b2[h] += m_estim / sqrt(v_estim + eps);//накопления для b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//для конкретного h закончили обновлять bh, теперь все веса из первого слоя связанные с h
			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] = beta1 * dE_g1[h][i] + (1 - beta1) * dw * dw;//экспоненциальное забывание для конкретного веса
				dM_g1[h][i] = beta2 * dM_g1[h][i] + (1 - beta2) * dw;
				if (time <10) m_estim = dM_g1[h][i] / x1;
				else m_estim= dM_g1[h][i];
				if (time <100) v_estim = dE_g1[h][i] / x2;
				else v_estim= dE_g1[h][i];
				d_weight1[h][i] += m_estim / sqrt(v_estim + eps);//накопление суммы для конкретного веса
				if (d_count == d_count_max)
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//изменения веса.
					d_weight1[h][i] = 0;
				}
			}
		}
		//обновления по внешнему слою (веса в него и сдвиги в нем)
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			dE_b3[m] = beta1 * dE_b3[m] + (1 - beta1) * dw2 * dw2;//экспоненциальное забывание для сдвига
			dM_b3[m] = beta2 * dM_b3[m] + (1 - beta2) * dw2;//забывание для производной первой степени
			if (time <10) m_estim = dM_b3[m] / x1;
			else m_estim= dM_b3[m];
			if (time <100) v_estim = dE_b3[m] / x2;
			else v_estim= dE_b3[m];
			
			d_b3[m] += m_estim / sqrt(v_estim + eps);//накопления для b
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)

				dE_g2[m][h] = beta1 * dE_g2[m][h] + (1 - beta1) * dw * dw;//экспоненциальное забывание для конкретного веса
				dM_g2[m][h] = beta2 * dM_g2[m][h] + (1 - beta2) * dw;
				if (time <10) m_estim = dM_g2[m][h] / x1;
				else m_estim= dM_g2[m][h];
				if (time <100) v_estim = dE_g2[m][h] / x2;
				else v_estim= dE_g2[m][h];
				
				d_weight2[m][h] += m_estim / sqrt(v_estim + eps);//накопление суммы для конкретного веса
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 3://adagrad
	{
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от
			//индекса инпута
			dE_b2[h] = dE_b2[h] + dw2 * dw2;//постоянное накопление квадрата градиента
			d_b2[h] += dw2 / (sqrt(dE_b2[h]) + eps);//накопления для b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//для конкретного h закончили обновлять bh, теперь все веса из первого слоя связанные с h*/
			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] += dw * dw;//экспоненциальное забывание для конкретного веса
				d_weight1[h][i] += dw / (sqrt(dE_g1[h][i]) + eps);//накопление суммы для конкретного веса
				if (d_count == d_count_max)
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//изменения веса.
					d_weight1[h][i] = 0;
				}
			}
		}
		//обновления по внешнему слою (веса в него и сдвиги в нем)
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			dE_b3[m] += dw2 * dw2;//экспоненциальное забывание для B внешнего слоя
			d_b3[m] += dw2 / (sqrt(dE_b3[m]) + eps);//накопление суммы градиента
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				dE_g2[m][h] += dw * dw;
				d_weight2[m][h] += dw / (sqrt(dE_g2[m][h]) + eps);
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 4://momentum
	{
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от
			//индекса инпута
			dE_b2[h] = gamma * dE_b2[h] + n * dw2;//экспоненциальное забывание для сдвига
			d_b2[h] += dE_b2[h];//накопления для b
			if (d_count == d_count_max)
			{
				b2[h] -= d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//для конкретного h закончили обновлять bh, теперь все веса из первого слоя связанные с h
			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] = gamma * dE_g1[h][i] + n * dw;//экспоненциальное забывание для конкретного веса
				d_weight1[h][i] += dE_g1[h][i];//накопление суммы для конкретного веса
				if (d_count == d_count_max)
				{
					weight1[h][i] -= d_weight1[h][i] / (1.0 * d_count_max);//изменения веса.
					d_weight1[h][i] = 0;
				}
			}
		}
		//обновления по внешнему слою (веса в него и сдвиги в нем)
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			dE_b3[m] = gamma * dE_b3[m] + n * dw2;//экспоненциальное забывание для B внешнего слоя
			d_b3[m] += dE_b3[m];//накопление суммы градиента
			if (d_count == d_count_max)
			{
				b3[m] -= d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				dE_g2[m][h] = gamma * dE_g2[m][h] + n * dw;
				d_weight2[m][h] += dE_g2[m][h];
				if (d_count == d_count_max)
				{
					weight2[m][h] -= d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 5://nesterov есть вопросы в применимости в данном случае
	{
		Set_Networkb2(g);//накопили изменения для б2
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от
			//индекса инпута
			dE_b2[h] = gamma * dE_b2[h] + n * dw2;//экспоненциальное забывание для сдвига
			d_b2[h] += dE_b2[h];//накопления для b
		}
		Set_Networkb3(g);//накопим изменения для б3
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			dE_b3[m] = gamma * dE_b3[m] + n * dw2;//экспоненциальное забывание для B внешнего слоя
			d_b3[m] += dE_b3[m];
		}
		Set_Networkw1(g);
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];
			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] = gamma * dE_g1[h][i] + n * dw;//экспоненциальное забывание для конкретного веса
				d_weight1[h][i] += dE_g1[h][i];
			}
		}
		Set_Networkw2(g);
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			for (int h = 0; h < meadle_size; h++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				dE_g2[m][h] = gamma * dE_g2[m][h] + n * dw;
				d_weight2[m][h] += dE_g2[m][h];
			}
		}
		//накопили все необходимое. теперь при необходимости обновим
		if (d_count == d_count_max)
		{
			for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
			{
				b2[h] -= d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
				for (int i = 0; i < in_size; i++)
				{
					weight1[h][i] -= d_weight1[h][i] / (1.0 * d_count_max);//изменения веса.
					d_weight1[h][i] = 0;
				}
			}
			for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
			{
				b3[m] -= d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
				for (int h = 0; h < meadle_size; h++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
				{
					weight2[m][h] -= d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 6://adadelta
	{
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от индекса инпута
			dE_b2[h] = gamma1 * dE_b2[h] + (1 - gamma1) * dw2 * dw2;//элемент рмс
			dw = 1.0 * sqrt(dM_b2[h] + eps) * dw2 / sqrt(dE_b2[h] + eps);//дельта тетта текущего шага, используем для вычислений следующего шага(здесь дм еще предыдущего шага
			dM_b2[h] = gamma * dM_b2[h] + (1 - gamma) * dw * dw;
			//элемент рмс для дельта тета
			d_b2[h] += dw;//накопления для b
			if (d_count == d_count_max)
			{
				b2[h] -= d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//для конкретного h закончили обновлять bh, теперь все веса из первого слоя связанные с h*/

			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
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
				dE_g1[h][i] = gamma1 * dE_g1[h][i] + (1 - gamma1) * dw * dw;//экспоненциальное забывание для конкретного веса
				dw2 = sqrt(dM_g1[h][i] + eps) * dw / sqrt(dE_g1[h][i] + eps);//дельта тетта текущего шага, используем для вычислений следующего шага(здесь дм еще предыдущего шага
				dM_g1[h][i] = gamma * dM_g1[h][i] + (1 - gamma)* dw2 * dw2;
				//}//элемент рмс для дельта тета*/
				d_weight1[h][i] += dw2;//накопления для b
				if (d_count == d_count_max)
				{
					weight1[h][i] -= d_weight1[h][i] / (1.0 * d_count_max);//изменения веса.
					//weight1[h][i] =dM_g1[h][i];
					d_weight1[h][i] = 0;
				}
			}
		}
		//обновления по внешнему слою (веса в него и сдвиги в нем)
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			dE_b3[m] = gamma1 * dE_b3[m] + (1 - gamma1) * dw2 * dw2;//экспоненциальное забывание для B внешнего слоя
			dw = sqrt(dM_b3[m] + eps) * dw2 / sqrt(dE_b3[m] + eps);//дельта тетта текущего шага, используем для вычислений следующего шага(здесь дм еще предыдущего шага
			dM_b3[m] = gamma * dM_b3[m] + (1 - gamma) * dw * dw;
			//}//элемент рмс для дельта тета*/
			d_b3[m] += dw;// sqrt(dE_b3[m] + eps);//накопление суммы градиента
			if (d_count == d_count_max)
			{
				b3[m] -= d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}

			for (int h = 0; h < meadle_size; h++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)

				dE_g2[m][h] = gamma1 * dE_g2[m][h] + (1 - gamma1) * dw * dw;//экспоненциальное забывание для конкретного веса
				dw2 = sqrt(dM_g2[m][h] + eps) * dw / sqrt(dE_g2[m][h] + eps);//дельта тетта текущего шага, используем для вычислений следующего шага(здесь дм еще предыдущего шага
				dM_g2[m][h] = gamma * dM_g2[m][h] + (1 - gamma) * dw2 * dw2;
				//}//элемент рмс для дельта тета*/
				d_weight2[m][h] += dw2;//накопления для b
				//if (m==0) ww[h] = dw2;
				if (d_count == d_count_max)
				{
					weight2[m][h] -= d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.

					d_weight2[m][h] = 0;
				}
			}
		}
		break;
	}
	case 7:
	{/*комбинированный метод обучения который предлагается. для обучения использовать несколько типов обучения. с самого начала параллельно вычислять все параметры
	 для каждого из типов обучения. то есть если типов обучения будет три, то вычислять переменные для первого, для второго и для третьего
	 возьмем в рассмотрение 1)адаград (так как у него может быстро сумма накопиться) 2)рмс как улучшение адаград 
	 3) моментум (как определитель дальнейшего направления) [ нестеров не стала брать так как ведет за собой больше вычислений]
	 dE будут параметры для адаград, dM for momentum, dE_rms for rms*/
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];
			/*посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от индекса инпута*/
			//параметры для адаград
			dE_b2[h] = dE_b2[h] + dw2 * dw2;//постоянное накопление квадрата градиента
			//параметры для моментума
			dM_b2[h] = gamma * dM_b2[h] + n * dw2;//экспоненциальное забывание для сдвига
			//rms
			dE_rms_b2[h] = gamma * dE_rms_b2[h] + (1 - gamma) * dw2 * dw2;//экспоненциальное забывание для сдвига
			if (time < 50)
				d_b2[h] += dw2 / (sqrt(dE_b2[h]) + eps);//adagrad
			if (time >= 50 && time < 100)
				d_b2[h] += dw2 / sqrt(dE_rms_b2[h] + eps);//rms
			if (time >= 100)
				d_b2[h] += dM_b2[h];//momentum

			if (d_count == d_count_max)//обновление
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//для конкретного h закончили обновлять bh, теперь все веса из первого слоя связанные с h*/
			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
			{
				dw = dw2 * int_input[g][i];//это производная для весов первого слоя
				//параметры для адаград
				dE_g1[h][i] += dw * dw;//экспоненциальное забывание для конкретного веса
				//параметры для моментума
				dM_g1[h][i] = gamma * dM_g1[h][i] + n * dw;//экспоненциальное забывание для конкретного веса
				//параметры для 
				dE_rms_g1[h][i] = gamma * dE_rms_g1[h][i] + (1 - gamma) * dw * dw;//экспоненциальное забывание для конкретного веса
				//накопление суммы для конкретного веса*/
				if (time < 50)
					d_weight1[h][i] += dw / (sqrt(dE_g1[h][i]) + eps);//адаград
				if (time >= 50 && time < 100)
					d_weight1[h][i] += dw / sqrt(dE_rms_g1[h][i] + eps);//рмс
				if (time >= 100)
					d_weight1[h][i] += dM_g1[h][i];//моментум

				if (d_count == d_count_max)//обновление
				{
					weight1[h][i] -= n * d_weight1[h][i] / (1.0 * d_count_max);//изменения веса.
					d_weight1[h][i] = 0;
				}
			}
		}
		//обновления по внешнему слою (веса в него и сдвиги в нем)
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			//параметры для адаград
			dE_b3[m] += dw2 * dw2;//экспоненциальное забывание для B внешнего слоя
			//параметры для моментума
			dM_b3[m] = gamma * dM_b3[m] + n * dw2;//экспоненциальное забывание для B внешнего слоя
			//параметры rms
			dE_rms_b3[m] = gamma * dE_rms_b3[m] + (1 - gamma) * dw2 * dw2;//экспоненциальное забывание для B внешнего слоя

			if (time < 50)
				d_b3[m] += dw2 / (sqrt(dE_b3[m]) + eps);//адаград
			if (time >= 50 && time < 100)
				d_b3[m] += dw2 / sqrt(dE_rms_b3[m] + eps);//рмс
			if (time >= 100)
				d_b3[m] += dM_b3[m];//моментум
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
			    //параметры для адаград
				dE_g2[m][h] += dw * dw;
				//параметры для моментума
				dM_g2[m][h] = gamma * dM_g2[m][h] + n * dw;
				//параметры rms 
				dE_rms_g2[m][h] = gamma * dE_rms_g2[m][h] + (1 - gamma) * dw * dw;
				
				if (time < 50)
					d_weight2[m][h] += dw / (sqrt(dE_g2[m][h]) + eps);
				if (time >= 50 && time < 100)
					d_weight2[m][h] += dw / sqrt(dE_rms_g2[m][h] + eps);//рмс
				if (time >= 100)
					d_weight2[m][h] += dM_g2[m][h];//мометум
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
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

//-----Вычисление ошибки для одного элемента базы данных
double Error(int g) 
{
    double s = 0;
    for (int i = 0; i < out_size; i++)
		s += (out[i] - answer[g][i]) * (out[i] - answer[g][i]);	
    return (s / 2.0);
}

//-----СОхранение весов в файл
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

//-----Загрузка весов из файла
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

//-----Записывание отдельно (внутри программы) весов при минимальном показателе ошибки при валидации
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

//-----Загрузили веса  при минимальных показателях ошибки
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

//-----Сохранить стартовую инициализацию весов что бы потом при смене обучения была та же самая
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

//-----Загрузили стартовый инициализированный вес что бы он был везде одинаков
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

//-----Говорит о числе которое получила сеть при проходе
int Rez()  // вывод результата работы сети исходя из полученного выходного вектора out
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
    Ini();//инициальзация вектором w и сум
	/*memset(w_start, 0, sizeof(w_start));
	memset(w2_start, 0, sizeof(w2_start));
	memset(b2_start, 0, sizeof(b2_start));
	memset(b3_start, 0, sizeof(b3_start));
	save_start();//*/

    Read_mnist();
	FILE *gr_err, *f1, *z1;
	int res = 0, res2 = 0, ans =0, type_teach=2;
	//type: 0- 1 -rmsprop 2 - adam 3- adagrad 4-momentu 5- нестеров(?) 6- ададельта
    while (ans != 4) 
    {
        system("cls");
        cout << "1 - Обучение сети;" << endl;//обучение
        cout << "2 - Анализ тестового множества;" << endl;//проверка системы
        cout << "3 - Загрузка сохраненных весов;" << endl;
        cout << "4 - Выход (окончание работы)." << endl << endl;
        cin >> ans;
        bool stop = true;//флаг
        double error_arr[4] = {9999,9999,9999,9999}, error_of_step = 0, err_of_res=0;
        int number_step = 1, z = 0; // number_step - номер эпохи обучения
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
				while (stop && number_step < steps)  // эпохи обучения. степ - максимально количество эпох
				{
					error_of_step = 0; err_of_res = 0;
					cout << endl << endl << "!!!  " << number_step << " (" << type_teach << ") " << "  !!!" << endl << endl;//вывод эпохи
					int curr = 0;
					for (; curr < valid_cur; curr++)  // обучение от нуля и до валидационной части. валидационную пропускаем и обучаемся на оставшемся
					{
						Set_Network(curr);//проход параметров и получение результатов сетки
						Teach(curr, type_teach, number_step);//накапливаем градиент
						if (curr % 1000 == 0)
							cout << curr / 1000 << " ";
					}
					curr += valid_size;
					for (; curr < base_size - test_size; curr++)  //валидационную пропускаем и обучаемся на оставшемся
					{
						Set_Network(curr);//проход параметров и получение результатов сетки
						Teach(curr, type_teach, number_step);//накапливаем градиент
						if (curr % 1000 == 0)
							cout << curr / 1000 << " ";
					}
					cout << endl << endl;

					//валидация
					for (int j = valid_cur; j < valid_size + valid_cur; j++) // часть БД на валидацию (подсчет ошибки),используется 1000элементов в разных местах каждую эпоху
					{
						Set_Network(j);//прогон и подсчет ошибок
						error_of_step += Error(j);//сумма ошибок по решению икселя.
						res = Rez();//ответ на число, номер
						if (int_label[j] == res) res2 = 0;//совпадает или нет с завяленным числом
						else res2 = 1;//ошибка
						err_of_res += res2;
					}
					error_arr[0] = error_arr[1];
					error_arr[1] = error_arr[2];
					error_arr[2] = error_arr[3];
					error_arr[3] = error_of_step / valid_size;//усреднили для данного 
					err_of_res /= valid_size;
					cout << endl << endl << error_arr[3] << " ";//вывели на экран и в файл
					cout << err_of_res;
					fprintf(gr_err, "%.5f;%.5f\n", error_arr[3], err_of_res);
					if (err_of_res < min_err)
					{
						min_err = err_of_res;
						save_min();
					}
					if ((error_arr[0] < error_arr[1]) && (error_arr[1] < error_arr[2]) && (error_arr[2] < error_arr[3]) && (number_step >= 150)) // если ошибка некоторое время увеличивается - окончание обучения
						stop = false;
					number_step++;
					valid_cur -= valid_size;
					if (valid_cur < 0) valid_cur = base_size - valid_size - test_size;
				}
				load_min();
				//cout << endl << "Обучение закончено, веса сохранены автоматически." << endl;
				save_weight(type_teach);
				fclose(gr_err);// fclose(z1);
			//}
			system("pause");
			break;
        case 2://анализ, проверка на тестовой выборке
			f1 = fopen ("analiz.csv", "w");
			for (int i = base_size-test_size; i < base_size; i++)//проверка оставшейся базы. от 45 до 60тыс картинок
			{
				Set_Network(i);
				res = Rez();//ответ на число, номер
				if (int_label[i] == res) res2=0;//совпадает или нет с завяленным числом
				else res2 = 1;//ошибка
				cout << "Ответ: " << res << endl;//вывод ответа сети
				fprintf (f1, "%d ; %d ; %d ;\n", int_label[i], res, res2);//сохранение данных в файл
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