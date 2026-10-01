#include <iostream>
#include <vector>
#include "matrix.h"
#include "datatype.h"
#include <cmath>

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
struct Scaler
{
	f32 mean;
	f32 sd;
};
Scaler normalize_columns(Matrix<f32>& X, std::size_t col)
{
	const std::size_t n = X.rows;
	const f32 count = static_cast<f32> (n);
	f32 sum = 0;
	for (std::size_t i = 0; i < n; i++)
	{
		sum += X.at(i, col);
	}
	f32 mean = sum / count;
	f32 var = 0;
	for (std::size_t i = 0; i < n; i++)
	{
		f32 d = X.at(i, col) - mean;
		var += d * d;
	}
	f32 sd = std::sqrt(var / count);
	if (sd == 0) throw std::invalid_argument("Column has zero variance");
	for (std::size_t i = 0; i < n; i++)
	{
		X.at(i, col) = (X.at(i, col) - mean) / sd;
	}
	return { mean, sd };
}
// given value of X -> predict y
f32 predict(const Matrix<f32>& w, f32 x)
{
	Matrix<f32> input(1, 2); //one for x and other for w
	input.at(0, 0) = x; 
	input.at(0, 1) = 1.0f;
	/*
	suppose I give x = 3 
	[3 1] x [w0] 
			[w1]
	Basically we are updating only the column 2 of input matrix
	*/
	Matrix<f32> out = mat_mul(input, w);
	return out.at(0, 0);
}
f32 predictScaled(const Matrix<f32>& w, f32 x, const Scaler& s)
{
	return predict(w, (x - s.mean) / s.sd);
}
void LinearRegression()
{
	const std::size_t n = 10;
	Matrix<f32> X(n, 2), y(n, 1);
	makeData(X, y);
	Scaler s = normalize_columns(X, 0);
	//initial weights will be 0,0
	Matrix<f32> w(2, 1);
	const f32 lr = 0.005f; //learning rate
	const int epochs = 10000;
	const f32 factor = 2 / static_cast<f32>(n);
	// formula for gradient descent = 1/2(Xw - y)^2
	//after you apply chain rule (Xw - y) X ...wrt w
	//
 	for (int epoch = 0; epoch < epochs; epoch++)
	{
		Matrix<f32> pred = mat_mul(X, w); // Xw
		Matrix<f32> e = subtract(pred, y); //e = Xw - y 
		Matrix<f32> grad = scale(mat_mul(transpose(X), e), factor); // (e . X)^2 so we do XT.e to get sum, and just scale it
		w = subtract(w, scale(grad, lr)); // w = w - alpha(grad)
		if (epoch % 1000 == 0)
		{
			std::cout << "Epoch: " << epoch << " mse: " << mse(pred, y) << std::endl;
		}

	}
	std::cout << "Final:- w :\n";
	w.print();
	std::cout << "Prediction for X = 20  --> " << predictScaled(w, 20.0f, s) << std::endl;
}


// we need to normalize the data right


void matrixLesson();