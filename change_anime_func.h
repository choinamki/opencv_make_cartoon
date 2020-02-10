#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;


/*
--- 3 * 3 �߰��� ���� ---
*/
void median_filter(Mat& input_image, Mat& output_image)
{
	output_image = Mat(input_image.size(), input_image.type()); // ���� ��� �̹���

	if (input_image.channels() == 3) //�Է� ���� �÷��� ���
	{
		int pixelB[9];
		int pixelG[9];
		int pixelR[9];
		for (int y = 1; y < input_image.rows - 1; y++)
		{
			for (int x = 1; x < input_image.cols - 1; x++)              //3*3 ��ǥ ����
			{
				int index = 0; // �ε��� �ʱ�ȭ

				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						pixelB[index] = input_image.at<Vec3b>(y + j - 1, x + k - 1)[0];
						pixelG[index] = input_image.at<Vec3b>(y + j - 1, x + k - 1)[1];
						pixelR[index] = input_image.at<Vec3b>(y + j - 1, x + k - 1)[2];
						index = index + 1;
					}
				}
				// ũŰ�� ����
				sort(pixelB, pixelB + 9);
				sort(pixelG, pixelG + 9);
				sort(pixelR, pixelR + 9);

				// �߰��� �Է�
				output_image.at<Vec3b>(y, x)[0] = pixelB[4];
				output_image.at<Vec3b>(y, x)[1] = pixelG[4];
				output_image.at<Vec3b>(y, x)[2] = pixelR[4];
			}
		}
	}
	else // �Է� ���� ����� ���
	{
		int pixel[9]; // �ȼ� ����

		for (int y = 1; y < input_image.rows - 1; y++)
		{
			for (int x = 1; x < input_image.cols - 1; x++)              //3*3 ��ǥ ����
			{
				int index = 0; // �ε��� �ʱ�ȭ

				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						pixel[index] = input_image.at<uchar>(y + j - 1, x + k - 1);
						index = index + 1;
					}
				}
				sort(pixel, pixel + 9);
				output_image.at<uchar>(y, x) = pixel[4];
			}
		}
	}
}
/*
--- 5 * 5 ����þ� ����
*/
void gausian_filter(Mat& input_image, Mat& output_image)
{
	Mat gausian_image = Mat(input_image.size(), input_image.type());
	//����þ� ����
	float a = 1 / 273.0;
	float b = 4 / 273.0;
	float c = 7 / 273.0;
	float d = 16 / 273.0;
	float e = 26 / 273.0;
	float f = 41 / 273.0;

	float Gousian[5][5] = {
		a, b, c, b, a,
		b, d, e, d, b,
		c, e, f, e, c,
		b, d, e, d, b,
		a, b, c, b, a
	};

	// �÷� �Ǵ� ����� ���� ���ǹ�
	if (input_image.channels() == 3) //�÷��ϰ��
	{
		Mat input_hsv_image;
		cvtColor(input_image, input_hsv_image, COLOR_BGR2HSV);
		input_hsv_image.copyTo(gausian_image);

		//����þ� ���� ������
		for (int y = 2; y < input_hsv_image.rows - 2; y++)                  //5*5 �����̱� ������ +2 -2 �� ����
		{
			for (int x = 2; x < input_hsv_image.cols - 2; x++)
			{
				float sum = 0;

				for (int k = 0; k < 5; k++)                   // ����þ� ���� ����
				{
					for (int l = 0; l < 5; l++)
					{
						sum += input_image.at<Vec3b>(y + k - 2, x + l - 2)[2] * Gousian[k][l];
					}
				}
				gausian_image.at<Vec3b>(y, x)[2] = int(sum);
			}
		}
		cvtColor(gausian_image, output_image, COLOR_HSV2BGR); 		// bgr ��ȭ
	}
	else //����ϰ��
	{
		//����þ� ���� ������
		for (int y = 2; y < input_image.rows - 2; y++)                  //5*5 �����̱� ������ +2 -2 �� ����
		{
			for (int x = 2; x < input_image.cols - 2; x++)
			{
				float sum = 0;
				for (int k = 0; k < 5; k++)                   // ����þ� ���� ����
				{
					for (int l = 0; l < 5; l++)
					{
						sum += input_image.at<uchar>(y + k - 2, x + l - 2) * Gousian[k][l];
					}
				}
				gausian_image.at<uchar>(y, x) = int(sum);
			}
		}
		output_image = gausian_image; // �����
	}
}


/*
�Һ� ���͸� ����� �ܰ��� ���� �Լ�
*/
void sobel_edge(Mat& raw_input_image, Mat& output_image, int threshold)
{
	Mat input_image; // ������ ���� �Է� �̹���
	Mat sobel_x_image; // �Һ����� x �̹���
	Mat sobel_y_image; // �Һ����� y �̹���

	//�Һ� ����
	float sobel_edge_x[3][3] = {
				-1, 0,  1,
				-2, 0,  2,
				-1, 0,  1
	};
	float sobel_edge_y[3][3] = {
				-1, -2,  -1,
				 0,  0,   0,
				 1,  2,   1
	};

	// ������ �÷��� ��� ������� ��ȯ
	if (raw_input_image.channels() == 3)
	{
		cvtColor(raw_input_image, input_image, COLOR_BGR2GRAY);
	}
	else
	{
		raw_input_image.copyTo(input_image);
	}

	sobel_x_image = Mat(input_image.size(), input_image.type());
	sobel_y_image = Mat(input_image.size(), input_image.type());

	// �ܰ��� ����
	for (int y = 1; y < input_image.rows - 1; y++)
	{
		for (int x = 1; x < input_image.cols - 1; x++)
		{
			//
			// --- ���� ���� ---
			//
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					sobel_x_image.at<uchar>(y, x) += input_image.at<uchar>(y + k - 1, x + l - 1) * sobel_edge_y[k][l];
					sobel_y_image.at<uchar>(y, x) += input_image.at<uchar>(y + k - 1, x + l - 1) * sobel_edge_x[k][l];
				}
			}
			//
			// --- �Ӱ谪 ���� ---
			//
			if (threshold != 0)
			{
				if (sobel_x_image.at<uchar>(y, x) > threshold)
				{
					sobel_x_image.at<uchar>(y, x) = 255;
				}
				else
				{
					sobel_x_image.at<uchar>(y, x) = 0;
				}

				if (sobel_y_image.at<uchar>(y, x) > threshold)
				{
					sobel_y_image.at<uchar>(y, x) = 255;
				}
				else
				{
					sobel_y_image.at<uchar>(y, x) = 0;
				}
			}
		}
	}
	//imshow("x", sobel_x_image);
	//imshow("y", sobel_y_image);
	output_image = sobel_x_image + sobel_y_image;
}


/*
k-means �̿��� ���� ���̱�
opencv�� �̿��� ������ ����ó�� õ�α� �� �����߽��ϴ�.
*/
void choose_color(Mat& input_image, Mat& output_image, int cluster_num)
{
	/*
	samples Ŭ������ ���� Mat
	labels Ŭ������ ������ �÷� label ��
	centers Ŭ�������� �߽��� ����� ��� ���
	KMEANS_PP_CENTERS �ʱⰪ �÷���
	TermCriteria::MAX_ITER  �˰����� �ִ� �ݺ��� Ƚ��
	TermCriteria::EPS �־��� ��Ȯ���� �����ϸ� ����
	10000�� �ݺ� 0.0001���� ������ ������ ����
	*/

	Mat samples(input_image.rows * input_image.cols, 3, CV_32F);
	Mat labels;
	int attempts = 5;
	Mat centers;

	output_image = Mat(input_image.size(), input_image.type()); // ���� ������ �̹���

	// �н� ������ ����
	for (int y = 0; y < input_image.rows; y++)
	{
		for (int x = 0; x < input_image.cols; x++)
		{
			for (int z = 0; z < 3; z++)
			{
				samples.at<float>(y + x * input_image.rows, z) = input_image.at<Vec3b>(y, x)[z];
			}
		}
	}

	// k-means ����
	kmeans(samples, cluster_num, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001), 3, KMEANS_PP_CENTERS, centers);

	for (int y = 0; y < input_image.rows; y++)
	{
		for (int x = 0; x < input_image.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input_image.rows, 0);
			output_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			output_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			output_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}
}