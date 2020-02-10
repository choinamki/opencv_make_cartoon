#include "opencv2/opencv.hpp"
#include <iostream>
#include "change_anime_func.h"

using namespace cv;
using namespace std;

int main()
{
	string RAW_IMAGE_LINK = "./input_image/input4.jpg"; // 이미지 링크

	Mat input_raw_image; // 처음 입력받는 raw 이미지

	Mat color_choose_image; // raw 이미지에서 kmean를 통해 컬러를 고른 이미지
	Mat gausian_color_image; // 컬러이미지 가우시안 적용

	Mat color_choose_sobel_image; // 소벨 필터 외곽선 이미지
	Mat sobel_median_image; // 소벨 필터에서 미디언필터 적용한 이미지

	Mat add_edge_image; // 선택한 컬러와 외곽선을 합친 이미지
	Mat result_image; //  최종 결과 이미지
	



	// --- 이미지 입력 ---
	input_raw_image = imread(RAW_IMAGE_LINK, IMREAD_COLOR);

	if (input_raw_image.empty()) // 이미지가 없을 경우
	{
		cout << "no image";
		return -1;
	}
	imshow("raw_image", input_raw_image);

	// --- 이미지 처리 부분 ---
	choose_color(input_raw_image, color_choose_image, 22);
	imshow("clustered color", color_choose_image);
	
	gausian_filter(color_choose_image, gausian_color_image);
	imshow("color_gusian", color_choose_image);

	sobel_edge(color_choose_image, color_choose_sobel_image, 90);
	imshow("sobel_image", color_choose_sobel_image);

	median_filter(color_choose_sobel_image, sobel_median_image);
	imshow("median_image", sobel_median_image);

	// --- 테두리 추가 부분 ---
	gausian_color_image.copyTo(add_edge_image); //모서리를 추가할 사진 복사

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

	// --- 샤프닝 필터 적용 최종 결과물 ---
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