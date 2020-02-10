#include "opencv2/opencv.hpp"
#include <iostream>
#include "change_anime_func.h"

using namespace cv;
using namespace std;

int main()
{
	string RAW_IMAGE_LINK = "./input_image/input4.jpg"; // �̹��� ��ũ

	Mat input_raw_image; // ó�� �Է¹޴� raw �̹���

	Mat color_choose_image; // raw �̹������� kmean�� ���� �÷��� �� �̹���
	Mat gausian_color_image; // �÷��̹��� ����þ� ����

	Mat color_choose_sobel_image; // �Һ� ���� �ܰ��� �̹���
	Mat sobel_median_image; // �Һ� ���Ϳ��� �̵������ ������ �̹���

	Mat add_edge_image; // ������ �÷��� �ܰ����� ��ģ �̹���
	Mat result_image; //  ���� ��� �̹���
	



	// --- �̹��� �Է� ---
	input_raw_image = imread(RAW_IMAGE_LINK, IMREAD_COLOR);

	if (input_raw_image.empty()) // �̹����� ���� ���
	{
		cout << "no image";
		return -1;
	}
	imshow("raw_image", input_raw_image);

	// --- �̹��� ó�� �κ� ---
	choose_color(input_raw_image, color_choose_image, 22);
	imshow("clustered color", color_choose_image);
	
	gausian_filter(color_choose_image, gausian_color_image);
	imshow("color_gusian", color_choose_image);

	sobel_edge(color_choose_image, color_choose_sobel_image, 90);
	imshow("sobel_image", color_choose_sobel_image);

	median_filter(color_choose_sobel_image, sobel_median_image);
	imshow("median_image", sobel_median_image);

	// --- �׵θ� �߰� �κ� ---
	gausian_color_image.copyTo(add_edge_image); //�𼭸��� �߰��� ���� ����

	for (int y = 0; y < sobel_median_image.rows; y++)
	{
		for (int x = 0; x < sobel_median_image.cols; x++)
		{
			if (sobel_median_image.at<uchar>(y, x) == 255)
			{
				for (int z = 0; z < 3; z++)
				{
					add_edge_image.at<Vec3b>(y, x)[z] = 0;
				}
			}
		}
	}
	imshow("add_edge_image", add_edge_image);

	// --- ������ ���� ���� ���� ����� ---
	float weights[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
	Mat sharp_mask = Mat(3, 3, CV_32F, weights);
	filter2D(add_edge_image, result_image, -1, sharp_mask, Point(-1, -1), 0, BORDER_DEFAULT);
	imshow("result_image", result_image);

	while (1)
	{
		int key = waitKey(50);

		if (key == '1')
		{
			return 0;
		}
	}
}