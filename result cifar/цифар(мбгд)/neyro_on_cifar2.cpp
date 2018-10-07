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
const int steps = 150, w_img=32;// steps - максимальное количество эпох обучения.
const int in_size = w_img*w_img*3, meadle_size = 25, out_size = 10;//размеры сетки: вход, внутренний, выходной (картинку вытянем из мтарицы в строку длины 784)
const int valid_size = 5000, test_size = 10000, base_size = 40000;//укороченный вариант
/* объем проверочной выборки. размер тестовой(конечной, отдельной) выборки,
размер всего тренировочного множества, которое придется загрузить польностью, что бы обращаться к разным объектам при валидации*/
const double n = pow(1.0, -2),//0.0001,// скорость обучения,
	gamma1 = 0.99, gamma = 0.999, eps = pow (10.0, -8),//  параметр экспоненц забывания, глаживающий параметр (RMS)
	beta1 = 0.9, beta2 = 0.999; //параметры экспоненциального забывания для оценки M D (Adam)
int valid_cur = base_size-valid_size-test_size, int_label[base_size], d_count = 0, d_count_max =50;// test_size; 
/*маркер начала части валидации. если валидация в центре, то обработка данных идет так: отстади_маркера до маркера валидации, 
+размер валидации, если полученное число меньше общего размера базы, то еще обучаемся после этого числа.
int_label - список меток классов. счетчики для спуска и максимальное анкопление по градиенту
счетчик для пакетного обновления, time - счетчик количества обновлений для епременной (адам)*/
//unsigned char input[base_size][w_img][w_img], label[base_size]; 
double int_input[base_size][in_size], weight1[meadle_size][in_size], weight2[out_size][meadle_size], b2[meadle_size], b3[out_size], w_start[meadle_size][in_size], w2_start[out_size][meadle_size],
/*вся база признаков, веса между входным слоем и среднем, веса из среднего слоя во внешний,
смещения на среднем слое, смещения на внешнем слое(используются при активации, так же обновляются, изначально равно нулю)*/
sum1[meadle_size], sum2[out_size], act[meadle_size], out[out_size], answer[base_size][out_size],  
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

//Считывание базы данных
void Read_cifar(string filename, int curr)
{//название файла, так как там имеется 5 файлов с данными, курр - маркер куда записывать данные, tr_test - маркет Train база(0) или Test(1) база
    ifstream file (filename, ios::binary);
	double Y=0;
    if (file.is_open()){
		for(int i = 0; i < test_size; ++i)
        {
            unsigned char tplabel = 0, temp = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
			int_label[curr+i]=(int)tplabel;
			answer[curr+i][int_label[curr+i]]=1;
			
			for (int j = 0; j < in_size; ++j)// in_size = 32*32*3
			{
				file.read((char*) &temp, sizeof(temp));
				int_input[curr + i][j] = (double)temp / 255.0;
			}
		}
		file.close();
	}
	else
		printf("Error on read cifar file %d\n",2*curr/10000);
}
void Read_base()
{
int curr =0;
	memset (answer, 0, sizeof(answer));//массив ожидаемых предсказаний: 0 или 1. заполним нулями, а в зависимости от лейбла поставим 1
	string filename;
    filename = "cifar-10-batches/data_batch_1.bin";
	Read_cifar(filename, curr); curr += test_size;
	filename = "cifar-10-batches/data_batch_2.bin";
    Read_cifar(filename, curr); curr += test_size;
	filename = "cifar-10-batches/data_batch_3.bin";
    Read_cifar(filename, curr); curr += test_size;
	/*filename = "cifar-10-batches/data_batch_4.bin";
    Read_cifar(filename, curr); curr += test_size;
	filename = "cifar-10-batches/data_batch_5.bin";
    Read_cifar(filename, curr); curr += test_size;*/
	filename = "cifar-10-batches/test_batch.bin";
    Read_cifar(filename, curr);
}
//-----Инициализация весов и сдвигов (b)----
void Ini() 
{
	//инициализация векторов подсчета градиентов и весов
    memset (b2, 0, sizeof(b2));
	memset (b3, 0, sizeof(b3));//сдвиги нейронов 2 и 3 слой
	memset (d_b2, 0, sizeof(d_b2));//величины накопления изменения весов если используется обновление пачками (d_count_max>1)
	memset (d_b3, 0, sizeof(d_b3));
	memset (d_weight1, 0, sizeof(d_weight1));
    memset (d_weight2, 0, sizeof(d_weight2));//аналогичные величины для весов 1-2 и 2-3 слой
	//memset (weight1, 0.1, sizeof(weight1));
	//memset (weight2, 0.1, sizeof(weight2));
	memset (dE_g1, 0.01, sizeof(dE_g1));//массив экспоненциального забывания по градиенту(производной в квадрате) весов для  1->2 слой
    memset (dE_g2, 0.01, sizeof(dE_g2));//массив экспоненциального забывания по градиенту весов для  2->3 слой
	memset (dE_b3, 0.01, sizeof(dE_b3));//массив экспоненциального забывания по градиенту сдвига b на 3м слое
    memset (dE_b2, 0.01, sizeof(dE_b2));//массив экспоненциального забывания по градиенту сдвига b на 2м слое
	
	memset (dM_g1, 0.1, sizeof(dE_g1));//массив экспоненциального забывания по градиенту(производная первой степени) весов для  1->2 слой (адам)
    memset (dM_g2, 0.1, sizeof(dE_g2));//или массив эксп.забывания для величины изменения веса\сдвига в квадрате (ададельта)
	memset (dM_b3, 0.1, sizeof(dE_b3));
    memset (dM_b2, 0.1, sizeof(dE_b2));

	for (int i = 0; i < meadle_size; i++)//инициализация весов из первого слоя во второй и сдвига нейронов на внутреннем слое
	{
		//b2[i] = 0.01 *  rand() / RAND_MAX ;
		for (int j = 0; j < in_size; j++)
			weight1[i][j] = 0.01 *  rand() / RAND_MAX ;
	}
	for (int i = 0; i < out_size; i++)//инициализация весов из внутреннего слоя в третий, выходной и сдвига нейронов на внешнем слое
	{
		//b3[i] =0.01 *  rand() / RAND_MAX;
		for (int j = 0; j < meadle_size; j++)
            weight2[i][j] = 0.01 * rand() / RAND_MAX ;
	}
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
	double dw = 0, dw2 = 0 ;
	double m_estim = 0, v_estim = 0;//переменные для алгоритма адам, оценка M D
	d_count++;
	//int x = base_size- test_size - valid_size;
	switch (type)
	{
	case 0://обновления без ускоряющих методов
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от
			d_b2[h] += dw2 ;//накопления для b
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
			dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			d_b3[m] += dw2 ;//накопление суммы градиента
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h ++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				d_weight2[m][h] += dw;
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h]=0;
				}
			}
		}
		break;
	case 1://RMS
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
			for (int h = 0; h < meadle_size; h ++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				dE_g2[m][h] = gamma * dE_g2[m][h] + (1 - gamma) * dw * dw;
				d_weight2[m][h] += dw / sqrt(dE_g2[m][h] + eps);
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h]=0;
				}
			}
		}
		break;
	case 2://adam
		double x1, x2;
		x1 = 1 - pow (beta1, time);
		x2 = 1 - pow (beta2, time);
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

				d_weight1[h][i] +=  m_estim / sqrt(v_estim + eps);//накопление суммы для конкретного веса
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
			for (int h = 0; h < meadle_size; h ++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)

				dE_g2[m][h] = beta1 * dE_g2[m][h] + (1 - beta1) * dw * dw;//экспоненциальное забывание для конкретного веса
				dM_g2[m][h] = beta2 * dM_g2[m][h] + (1 - beta2) * dw;
				if (time <10) m_estim = dM_g2[m][h] / x1;
				else m_estim= dM_g2[m][h];
				if (time <100) v_estim = dE_g2[m][h] / x2;
				else v_estim= dE_g2[m][h];

				d_weight2[m][h] +=  m_estim / sqrt(v_estim + eps);//накопление суммы для конкретного веса
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h]=0;
				}
			}
		}
		break;
	case 3://adagrad
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от
			//индекса инпута
			dE_b2[h] = dE_b2[h] +  dw2 * dw2;//постоянное накопление квадрата градиента
			d_b2[h] += dw2 / (sqrt(dE_b2[h]) + eps);//накопления для b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//для конкретного h закончили обновлять bh, теперь все веса из первого слоя связанные с h*/
			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] +=  dw * dw;//экспоненциальное забывание для конкретного веса
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
			for (int h = 0; h < meadle_size; h ++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				dE_g2[m][h] += dw * dw;
				d_weight2[m][h] += dw /( sqrt(dE_g2[m][h]) + eps);
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h]=0;
				}
			}
		}
		break;
	case 4://momentum
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
				b2[h] -=  d_b2[h] / (1.0 * d_count_max);
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
			for (int h = 0; h < meadle_size; h ++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				dE_g2[m][h] = gamma * dE_g2[m][h] + n * dw;
				d_weight2[m][h] += dE_g2[m][h];
				if (d_count == d_count_max)
				{
					weight2[m][h] -= d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h]=0;
				}
			}
		}
		break;
	case 5://nesterov есть вопросы в применимости в данном случае
		Set_Networkb2(g);//накопили изменения для б2
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= ((1 - act[h]) * act[h]);//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от
			//индекса инпута
			dE_b2[h] = gamma * dE_b2[h] + n * dw2;//экспоненциальное забывание для сдвига
			d_b2[h] += dE_b2[h];//накопления для b
		}
		Set_Networkb3(g);//накопим изменения для б3
		for (int m = 0; m < out_size; m++)//по количеству внешних нейронов
		{
			dw2 = (out[m] - answer[g][m]) * (1 - out[m]) * out[m];//значение для обновления b , dw=dw2*act(h)-для весов
			dE_b3[m] = gamma * dE_b3[m] + n* dw2;//экспоненциальное забывание для B внешнего слоя
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
			for (int h = 0; h < meadle_size; h ++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
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
				for (int h = 0; h < meadle_size; h ++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
				{
					weight2[m][h] -= d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h]=0;
				}
			}
		}
		break;
	case 6://adadelta
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= ((1 - act[h]) * act[h]);//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от индекса инпута
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
				dE_g1[h][i] = gamma1 * dE_g1[h][i] + (1 - gamma1) * dw * dw;//экспоненциальное забывание для конкретного веса
				dw2 =  sqrt(dM_g1[h][i] + eps) * dw / sqrt(dE_g1[h][i] + eps);//дельта тетта текущего шага, используем для вычислений следующего шага(здесь дм еще предыдущего шага
				dM_g1[h][i] = gamma * dM_g1[h][i] + (1 - gamma)* dw2 * dw2;
				//}//элемент рмс для дельта тета*/
				d_weight1[h][i] += dw2;//накопления для b
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
			dE_b3[m] = gamma1 * dE_b3[m] + (1 - gamma1) * dw2 * dw2;//экспоненциальное забывание для B внешнего слоя
			dw =  sqrt(dM_b3[m] + eps) * dw2 / sqrt(dE_b3[m] + eps);//дельта тетта текущего шага, используем для вычислений следующего шага(здесь дм еще предыдущего шага
			dM_b3[m] = gamma * dM_b3[m] + (1 - gamma) * dw * dw;
			//}//элемент рмс для дельта тета*/
			d_b3[m] += dw ;//накопление суммы градиента
			if (d_count == d_count_max)
			{
				b3[m] -= d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h ++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				dE_g2[m][h] = gamma1 * dE_g2[m][h] + (1 - gamma1) * dw * dw;//экспоненциальное забывание для конкретного веса
				dw2 = sqrt(dM_g2[m][h] + eps) * dw / sqrt(dE_g2[m][h] + eps);//дельта тетта текущего шага, используем для вычислений следующего шага(здесь дм еще предыдущего шага
				dM_g2[m][h] = gamma * dM_g2[m][h] + (1 - gamma) * dw2 * dw2;
				//}//элемент рмс для дельта тета*/
				d_weight2[m][h] += dw2;//накопления для b
				if (d_count == d_count_max)
				{
					weight2[m][h] -= d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					
					d_weight2[m][h]=0;
				}
			}
		}
		break;
	case 7:
		for (int h = 0; h < meadle_size; h++)//номер внутреннего слоя
		{
			x1 = 1 - pow (beta1, time);
			dw2 = 0;
			for (int m = 0; m < out_size; m++)
				dw2 += (out[m] - answer[g][m]) * (1 - out[m]) * out[m] * weight2[m][h];
			dw2 *= (1 - act[h]) * act[h];//посчитали часть производной для обновления весов и сдвигов. dw2 можно использовать для b, dw2*input для весов в зависимости от
			//индекса инпута
			dE_b2[h] = gamma*dE_b2[h] +  (1-gamma)*dw2 * dw2;//постоянное накопление квадрата градиента
			dM_b2[h] = beta2 * dM_b2[h] + (1 - beta2) * dw2;//забывание для производной
			m_estim = dM_b2[h] / x1;
			d_b2[h] += m_estim / (sqrt(dE_b2[h]) + eps);//накопления для b
			if (d_count == d_count_max)
			{
				b2[h] -= n * d_b2[h] / (1.0 * d_count_max);
				d_b2[h] = 0;
			}//для конкретного h закончили обновлять bh, теперь все веса из первого слоя связанные с h*/
			for (int i = 0; i < in_size; i++)//обновление по всем весам внешнего слоя связанных с конкретным нейроном внутреннего слоя
			{
				dw = dw2 * int_input[g][i];
				dE_g1[h][i] = gamma* dE_g1[h][i]+(1-gamma)* dw * dw;//экспоненциальное забывание для конкретного веса
				dM_g1[h][i] = beta2 * dM_g1[h][i] + (1 - beta2) * dw;//забывание для производной
				m_estim = dM_g1[h][i] / x1;
				d_weight1[h][i] += m_estim / (sqrt(dE_g1[h][i]) + eps);//накопление суммы для конкретного веса
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
			dE_b3[m] = gamma*dE_b3[m]+ (1-gamma)*dw2 * dw2;//экспоненциальное забывание для B внешнего слоя
			dM_b3[m] = beta2 * dM_b3[m] + (1 - beta2) * dw2;//забывание для производной
			m_estim = dM_b3[m] / x1;
			d_b3[m] += m_estim / (sqrt(dE_b3[m]) + eps);//накопление суммы градиента
			if (d_count == d_count_max)
			{
				b3[m] -= n * d_b3[m] / (1.0 * d_count_max);
				d_b3[m] = 0;
			}
			for (int h = 0; h < meadle_size; h ++)//h-внутенний слой (веса откуда), для весов связанных с конкретным m
			{
				dw = dw2 * act[h];//частная производная для wmh(в m из h)
				dE_g2[m][h] = gamma*dE_g2[m][h]+(1-gamma)* dw * dw;
				dM_g2[m][h] = beta2 * dM_g2[m][h] + (1 - beta2) * dw;//забывание для производной
				m_estim = dM_g2[m][h] / x1;
				d_weight2[m][h] += m_estim /( sqrt(dE_g2[m][h]) + eps);
				if (d_count == d_count_max)
				{
					weight2[m][h] -= n * d_weight2[m][h] / (1.0 * d_count_max);//изменения веса.
					d_weight2[m][h]=0;
				}
			}
		}
		break;
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

//-----Сохранение весов в файл
void save_weight(int x) 
{
	FILE *fp;
	fp = fopen("weig.txt","w");
	switch (x)
	{
	case 0: fp = fopen("weights0.txt", "w"); break;
	case 1: fp = fopen("weights1.txt", "w"); break;
	case 2: fp = fopen("weights2.txt", "w"); break;
	case 3: fp = fopen("weights3.txt", "w"); break;
	case 4: fp = fopen("weights4.txt", "w"); break;
	case 5: fp = fopen("weights5.txt", "w"); break;
	case 6: fp = fopen("weights6.txt", "w"); break;

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
void load_weight(int x) 
{
	FILE *fp;
	float zzz = 0;
	fp = fopen("weig.txt", "r");
	switch (x)
	{
	case 0: fp = fopen("weights0.txt", "r"); break;
	case 1: fp = fopen("weights1.txt", "r"); break;
	case 2: fp = fopen("weights2.txt", "r"); break;
	case 3: fp = fopen("weights3.txt", "r"); break;
	case 4: fp = fopen("weights4.txt", "r"); break;
	case 5: fp = fopen("weights5.txt", "r"); break;
	case 6: fp = fopen("weights6.txt", "r"); break;

	}
	for (int i = 0; i < meadle_size; i++)
	{
		fscanf(fp, "%f;", &zzz);
		b2[i] =zzz;
		for (int j = 0; j < in_size; j++)
		{
			fscanf (fp, "%f;", &zzz);//weight1[i][j][l]);
			weight1[i][j] = zzz;
		}
	}
	for (int i = 0; i < out_size; i++)
	{
		fscanf(fp, "%f;", &zzz);
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

//-----Запишем один раз стартовые инициализационные веса что бы начинать всегда с одной точки
void save_start()
{
	for (int h = 0; h < meadle_size; h++)
	{
		for (int i = 0; i < in_size; i++)
			w_start[h][i] = weight1[h][i];
		for (int m = 0; m < out_size; m++)
			w2_start[m][h] = weight2[m][h];
	}
}

//-----
void load_start()
{
	for (int h = 0; h < meadle_size; h++)
	{
		for (int i = 0; i < in_size; i++)
			weight1[h][i] = w_start[h][i];
		for (int m = 0; m < out_size; m++)
			weight2[m][h] = w2_start[m][h];
	}
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
    Ini();//инициальзация вектором w и сум//в инициализацию так же ушло чтение базы данных
	memset(w_start,  0, sizeof(w_start));
	memset(w2_start, 0, sizeof(w2_start));
	save_start();

	FILE *gr_err, *analiz_file;
	int res = 0, res2 = 0, ans = 0, type_teach = 0;
	bool flag_read = true;
	bool stop = true;
	//type: 0- обычный град. спуск 1 -rmsprop 2 - adam 3- adagrad 4-momentu 5- нестеров(?) 6- ададельта
    while (ans != 4) 
    {
        system("cls");
        cout << " 1 - Обучение сети;" << endl;//обучение
        cout << " 2 - Анализ тестового множества;" << endl;//проверка системы
        cout << " 3 - Загрузка сохраненных весов;" << endl;
        cout << " 4 - Выход (окончание работы)." << endl;
		cout << " 5 - Смена обучающего метода."<<endl<<endl;
        cin >> ans;
		Ini();
		if (flag_read)
		{	Read_base(); flag_read=false;}//флаг		
        switch (ans)
        {
        case 1://обучение			
			for (type_teach = 0; type_teach < 7; type_teach++)
			{
				Ini(); load_start();//сбросили все насчитанные значения которые были до этого и загрузили первые веса которые сгенерировались
				double error_arr[4] = { 9999,9999,9999,9999 }, error_of_step = 0, err_of_res = 0;//сброс показателей ошибок и инициализация
				int number_step = 1, z = 0;
				res = 0, res2 = 0, ans = 0;
				stop = true;//флаг что еще не остановка
				gr_err = fopen("text.csv", "w");
				switch (type_teach)
				{
				case 0: gr_err = fopen("graf_of_error0.csv", "w"); break;
				case 1: gr_err = fopen("graf_of_error1.csv", "w"); break;
				case 2: gr_err = fopen("graf_of_error2.csv", "w"); break;
				case 3: gr_err = fopen("graf_of_error3.csv", "w"); break;
				case 4: gr_err = fopen("graf_of_error4.csv", "w"); break;
				case 5: gr_err = fopen("graf_of_error5.csv", "w"); break;
				case 6: gr_err = fopen("graf_of_error6.csv", "w"); break;
				}

				while (stop && number_step < steps)  // эпохи обучения. степ - максимально количество эпох
				{
					error_of_step = 0; err_of_res = 0;
					cout << endl << endl << "!!!  " << number_step << "  (" << type_teach << ")" << "  !!!" << endl << endl;//вывод эпохи
					int curr = 0;
					//два цикла с обучением
					for (; curr < valid_cur; curr++)  // обучение от нуля и до валидационной части. валидационную пропускаем и обучаемся на оставшемся
					{
						Set_Network(curr);//проход параметров и получение результатов сетки
						Teach(curr, type_teach, number_step);//накапливаем градиент
						if (curr % 1000 == 0)//просто на экране отмечает количество пройденых тысяч изображений
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
					cout << endl << endl;//

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
					fprintf(gr_err, "%.5f; %.5f\n", error_arr[3], err_of_res);//error_arr[3]);
					if (err_of_res < min_err)//сохранение по процентному угадыванию элемента базы
					{
						min_err = err_of_res;
						save_min();
					}
					if ((error_arr[0] < error_arr[1]) && (error_arr[1] < error_arr[2]) && (error_arr[2] < error_arr[3]) && (number_step >= 100)) 
						stop = false;// если ошибка некоторое время увеличивается - окончание обучения
					number_step++;
					valid_cur -= valid_size;
					if (valid_cur < 0) valid_cur = base_size - valid_size - test_size;
				}
				load_min();
				save_weight(type_teach);
				fclose(gr_err);
			}
			cout << endl << "Обучение закончено, веса сохранены автоматически." << endl;
			system("pause");
			break;
        case 2://анализ, проверка на тестовой выборке
			analiz_file = fopen ("analiz.csv", "w");
			for (int i = base_size-test_size; i < base_size; i++)//проверка оставшейся базы. от 45 до 60тыс картинок
			{
				Set_Network(i);
				res = Rez();//ответ на число, номер
				if (int_label[i] == res) res2 = 0;//совпадает или нет с завяленным числом
				else res2 = 1;//ошибка
				cout << "Ответ: " << res << endl;//вывод ответа сети
				fprintf (analiz_file, "%d ; %d ; %d ;\n", int_label[i], res, res2);//сохранение данных в файл
			}
			fclose(analiz_file);
            system("pause");
            break;
        case 3:
            load_weight(type_teach);
            break;
		case 5:
			cout << "Был номер: " << type_teach << endl;
			 cout << " 0 - Градиентный спуск;" << endl;
			 cout << "1 - RMS;"<<endl;
        cout << " 2 - Adam;" << endl;//проверка системы
        cout << " 3 - Adagrad;" << endl;
        cout << " 4 - Momentum" << endl;
		cout << " 5 - Nesterov"<<endl;
		cout << " 6 - Adadelta"<<endl;
		cin >> type_teach;
		flag_read=false;
		break;
        default:
            break;
        }
    }
    return 0;
}