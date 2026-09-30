#include <iostream>
#include <vector>
#include "matrix.h"
#include "datatype.h"


void makeData(Matrix<f32>& X, Matrix<f32>& y)
{
	const f32 xs[] = { 2, 4, 5, 7, 8, 10, 12, 14, 16, 18 };
	const f32 ys[] = { 5, 9, 11, 15, 17, 20, 24, 27, 31, 35 };
	const std::size_t n = 10;

	if (X.rows != n || X.cols != 2 || y.rows != n || y.cols != 1)
	{
		throw std::invalid_argument("makeData: X must be 10x2 and y must be 10x1");
	}

	for (std::size_t i = 0; i < n; i++)
	{
		X.at(i, 0) = xs[i];
		X.at(i, 1) = 1.0f;      // bias column
		y.at(i, 0) = ys[i];
	}
}

void LinearRegression()
{
	const std::size_t n = 10;
	Matrix<f32> X(n, 2), y(n, 1);
	makeData(X, y);
	//initial weights will be 0,0
	Matrix<f32> w(2, 1);
	const f32 lr = 0.005f; //learning rate
	const int epochs = 500;
	const f32 factor = 2 / static_cast<f32>(n);

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		Matrix<f32> pred = mat_mul(X, w);
		Matrix<f32> e = subtract(pred, y);
		Matrix<f32> grad = scale(mat_mul(transpose(X), e), factor);
		w = subtract(w, scale(grad, lr));
		if (epoch < 20 || epoch % 1000 == 0)
		{
			std::cout << "Epoch: " << epoch << " mse: " << mse(pred, y) << std::endl;
		}

	}
	std::cout << "Final:- w :\n";
	w.print();
}

void matrixLesson();