import numpy as np
import Jacobi
import Zadel
import Relaxation
import FastestDesc

def main():
	n = 40
	h = 1/n
	eps = pow(h, 3)
	eps = 10**-4

	alpha = 2
	beta = 3
	gamma = 2

	q = lambda x: 1 + x
	p = lambda x: 1 + pow(x, gamma)
	u = lambda x: pow(x, alpha)*pow((1-x), beta)
	#  u(x) = x^2*(1-x)^3
	#  u'(x) = -3*x^2*(1-x)^2 + 2*x*(1-x)^3
	#  d_u = lambda x: -beta*pow(x, alpha)*pow((1-x), beta-1) + alpha*pow(x, alpha-1)*pow((1-x), beta)


	ai = np.array([p(i*h) for i in range(n)])

	ui = np.array([u(i*h) for i in range(n)])
	gi = np.array([q(i*h) for i in range(n)])

	# ((1+x^gamma)*(-3*x^2*(1-x)^2 + 2*x*(1-x)^3))'
	# (-3*x^2*(1-x)^2 + 2*x*(1-x)^3)+x^gamma*((-3*x^2*(1-x)^2 + 2*x*(1-x)^3))'



	left_f = lambda x: (
							- beta*alpha*pow(x, alpha-1)*pow((1-x), beta-1) +
							beta*pow(x, alpha)*(beta-1)*pow(1-x, beta-2) +
							alpha*(alpha-1)*pow(x, alpha-2)*pow(1-x, beta) -
							alpha*pow(x, alpha-1)*beta*pow(1-x, beta-1) -
							beta*(alpha+gamma)*pow(x, alpha+gamma-1)*pow((1-x), beta-1) +
							beta*pow(x, alpha+gamma)*(beta-1)*pow(1-x, beta-2) +
							alpha*(alpha+gamma-1)*pow(x, alpha+gamma-2)*pow(1-x, beta) -
							alpha*pow(x, gamma+alpha-1)*beta*pow(1-x, beta-1)
						)
	left_fi = np.array([left_f(i*h) for i in range(n)])

	fi = ui*gi - left_fi


	A = np.zeros((n, n))

	A[0, 0] = ai[0] + ai[1] + h ** 2 * gi[0];
	A[0, 1] = -ai[1];

	for i in range(1, n-1):
		A[i, i - 1] = -ai[i];
		A[i, i] = ai[i] + ai[i + 1] + h ** 2 * gi[i];
		A[i, i + 1] = -ai[i + 1];

	A[n-1, n-2] = -ai[n-1];
	A[n-1, n-1] = ai[n-1] + h ** 2 * gi[n-1] / 2;

	print(f'A: ')
	print(A)
	print()

	b = np.array([fi[i] * h**2 for i in range(n)]);
	b[n-1] = h**2 * fi[n-1]/2 - h * p(0)
	# b[n] = h ** 2 * fi[n] / 2 - h * u2 * p(1);

	print(f'b: {b}')
	print()

	y, iter, err = Jacobi.jacobi(A, b, eps)

	print(f'Решение методом Якоби, полученное за {iter} итераций:')
	print(y)
	print(f'Ошибка: {err[len(err) - 1]}', '\n')

	y, iter, err = Zadel.zadel(A, b, eps)

	print(f'Решение методом Зейделя, полученное за {iter} итераций:')
	print(y)
	print(f'Ошибка: {err[len(err) - 1]}', '\n')

	y, iter, err = Relaxation.relaxation(A, b, eps)

	print(f'Решение методом Релаксации, полученное за {iter} итераций:')
	print(y)
	print(f'Ошибка: {err[len(err) - 1]}', '\n')

	y, iter, err = FastestDesc.fastest_desc(A, b, eps)
	print(f'Решение методом Наискорейшего спуска, полученное за {iter} итераций:')
	print(y)
	print(f'Ошибка: {err[len(err) - 1]}', '\n')


if __name__ == "__main__":
	main()


