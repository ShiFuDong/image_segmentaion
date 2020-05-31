/*
流程处理思路：
先通过Kmeans训练，目的为了得到背景像素的一个标记（个人觉得没有这个必要，性价比太低了）
利用这个标记跟证件照的原图进行一波处理，得到背景跟前景的一个（0跟255）的分开，架构这个作为mask
对这个mask进行一波骚操作，目的是为了后面的羽化边缘，得到更加精确的效果，
羽化边缘有几种思路：先讲下这种的，即
利用3X3的内核去掉一个边界像素，再利用高斯模糊得到梯度边缘，得到这个之后，再最后的一步操作中，
会根据梯度的高低赋予不同的边界像素，即由权重得到边界值
最后的操作是根据mask的值赋予不同的值
如果是0则为背景像素
如果是255则为证件照的前景像素
其他的值就根据梯度的大小，权重进行重新赋值
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat mat_to_samples(Mat &image);
int main(int argc, char** argv) {
	Mat src = imread("./huan.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", src);

	// 数据组装
	Mat points = mat_to_samples(src);

	// 使用KMeans
	int numCluster = 4;
	Mat labels;
	Mat centers;
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(points, numCluster, labels, criteria, 3, KMEANS_PP_CENTERS, centers);

	// 去除背景，遮罩生成
	Mat mask=Mat::zeros(src.size(), CV_8UC1);
	int index = src.rows*2 + 2;
	int cindex = labels.at<int>(index, 0);
	int height = src.rows;
	int width = src.cols;
	Mat dst;
	src.copyTo(dst);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row*width + col;
			int label = labels.at<int>(index, 0);
			if (label == cindex) { // 背景
				dst.at<Vec3b>(row,col)[0] =0;
				dst.at<Vec3b>(row,col)[1] =0;
				dst.at<Vec3b>(row,col)[2] =0;
				mask.at<uchar>(row, col) = 0; 
			} else {
				mask.at<uchar>(row, col) = 255; //前景
			}
		}
	}
	imshow("mask", mask);
	imshow("K-Means result", dst);

	// 腐蚀 + 高斯模糊
	// 腐蚀是因为K-means之后还有部分的边缘背景没有取出
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	erode(mask, mask, k);//腐蚀操作后的图像白色边界少了一个边界像素，是全部的轮廓边界
	imshow("erode-mask", mask);
	GaussianBlur(mask, mask, Size(3, 3), 0, 0);	//3X3的内核去掉的是广义上的一个边界像素，5X5去掉的是两个边界像素
	imshow("Blur Mask", mask);

	// 通道混合
	RNG rng(12345);
	Vec3b color;
	color[0] = 255;//rng.uniform(0, 255);
	color[1] = 255;// rng.uniform(0, 255);
	color[2] = 255;// rng.uniform(0, 255);
	Mat result(src.size(), src.type());

	double w = 0.0;
	int b = 0, g = 0, r = 0;
	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int m = mask.at<uchar>(row, col);
			if (m == 255) {
				
				result.at<Vec3b>(row, col) = src.at<Vec3b>(row, col); //  前景
			}
			else if (m == 0) {
				result.at<Vec3b>(row, col) = color; // 背景
			} 
			else {
				w = m / 255.0;
				b1 = src.at<Vec3b>(row, col)[0];
				g1 = src.at<Vec3b>(row, col)[1];
				r1 = src.at<Vec3b>(row, col)[2];

				b2 = color[0];
				g2 = color[1];
				r2 = color[2];

				b = b1*w + b2*(1.0 - w);
				g = g1*w + g2*(1.0 - w);
				r = r1*w + r2*(1.0 - w);

				result.at<Vec3b>(row, col)[0] = b;
				result.at<Vec3b>(row, col)[1] = g;
				result.at<Vec3b>(row, col)[2] = r;
			}
		}
	}
	imshow("图像背景替换结果", result);

	waitKey(0);
	return 0;
}

Mat mat_to_samples(Mat &image) {
	int w = image.cols;
	int h = image.rows;
	int samplecount = w*h;
	int dims = image.channels();
	Mat points(samplecount, dims, CV_32F, Scalar(10));

	int index = 0;
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			index = row*w + col;
			Vec3b bgr = image.at<Vec3b>(row, col);
			points.at<float>(index, 0) = static_cast<int>(bgr[0]);
			points.at<float>(index, 1) = static_cast<int>(bgr[1]);
			points.at<float>(index, 2) = static_cast<int>(bgr[2]);
		}
	}
	return points;
}