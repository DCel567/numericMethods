import numpy as np
from sympy import diff
import pandas as pd
from time import time
#from tabulate import tabulate
import matplotlib.pyplot as plt


n = 40
h = 1/n
eps = h**3
m3 = 4
m4 = 5
gamma = 2
def u(x):
    return (x+2)*(2-x)
def p(x):
    return 1+x**gamma
def g(x):
    return x+1
def f(x):
    return -diff(p(x)*diff(u(x)))+g(x)*u(x)
b=[]
b.append((1/2)*(h**2)*f(0)-h*p(0)*m3)
for i in range(1,n-1):
    b.append(f(i*h)*(h**2))
b.append((1/2)*(h**2)*f(n)-h*p(n)*m4)
As=[]
for i in range(1,n+1):
    As.append((p(i*h-h)+p(i*h))/2)
A=np.zeros((n,n))
A[0][0]=As[0]+(1/2)*(h**2)*g(0)-h*p(0)
A[0][1]=-As[0]
for i in range(1,len(A)-1):
  A[i][i] = As[i] + As[i+1] + (h**2)*g(1)
  A[i][i+1]=-As[i+1]
  A[i+1][i] = -As[i]
  stop=1
A[n-1][n-1]=As[n-1]+(1/2)*(h**2)*g(n)-h*p(n)

print('A: ')
print(A)

#Метод прогонки
'''
def solution(a, b):


    n = len(a)
    x = [0 for k in range(0, n)]  # обнуление вектора решений
    print('Размерность матрицы: ', n, 'x', n)

    # Прямой ход
    v = [0 for k in range(0, n)]
    u = [0 for k in range(0, n)]
    # для первой 0-й строки
    v[0] = a[0][1] / (-a[0][0])
    u[0] = (- b[0]) / (-a[0][0])
    for i in range(1, n - 1):  # заполняем за исключением 1-й и (n-1)-й строк матрицы
        v[i] = a[i][i + 1] / (-a[i][i] - a[i][i - 1] * v[i - 1])
        u[i] = (a[i][i - 1] * u[i - 1] - b[i]) / (-a[i][i] - a[i][i - 1] * v[i - 1])
    # для последней (n-1)-й строки
    v[n - 1] = 0
    u[n - 1] = (a[n - 1][n - 2] * u[n - 2] - b[n - 1]) / (-a[n - 1][n - 1] - a[n - 1][n - 2] * v[n - 2])



    # Обратный ход
    x[n - 1] = u[n - 1]
    for i in range(n - 1, 0, -1):
        x[i - 1] = v[i - 1] * x[i] + u[i - 1]

    return x
'''
#Метод Зейделя

x0=np.zeros(n)
# MAIN - блок программмы
def Zeidel(A,b,x0,eps):
 Adiag=A-np.diag(np.diag(A))
 x1=np.zeros_like(x0)
 n=0
 epsilons=[]
 vect=[]
 while max(abs(A.dot(x1)-b))>eps:

  #print(pd.DataFrame(x1).T)

  epsilons.append(max(abs(A.dot(x1)-b)))

  x0=x1

  for i in range(len(x0)):
    summa1 = 0
    summa2 = 0
    for j in range(i):
     summa1 += (-x1[j])*(A[i][j]/A[i][i])
    for j in range(i+1, len(x0)):
     summa2 += (-x0[j])*(A[i][j]/A[i][i])
    x1[i] = summa1+summa2+(b[i]/A[i][i])
    vect.append(list(x0))

    n+=1
  return x1,n,epsilons,vect


solved, iterations, error1, vect = Zeidel(A,b,x0,eps=eps)
print("Кол-во итераций: " + str(iterations))
print("Ответ алгоритма: " + str(solved))
print("Ошибки: " + str(error1))
# plt.plot(error,label='График погрешности для метода Зейделя')
# plt.xlabel('Кол-во итераций')
# plt.ylabel('Величина невязки')
# plt.legend()
# #plt.show()
# pd.set_option('display.max_columns', None)
# pd.DataFrame(vect).to_excel('zeidel.xlsx')
#print(tabulate(pd.DataFrame(vect)))

#print("Решение: " + str(np.linalg.solve(A,b)))


#Метод релаксации

x0=np.zeros(n)
def Relax(A,x0,eps):
 x1=np.zeros_like(x0)
 n=0
 w = 1.3#[w for w in np.arange(0,2,0.01)]
 epsilons=[]
 vect = []

 while max(abs(A.dot(x1)-b))>eps:
  epsilons.append(max(abs(A.dot(x1) - b)))
  x0=x1
  for i in range(len(x0)):
    summa1=0
    summa2=0
    for j in range(i):
     summa1+=(-x1[j])*(A[i][j]/A[i][i])
    for j in range(i+1,len(x0)):
     summa2+=(-x0[j])*(A[i][j]/A[i][i])
    x1[i]=(1-w)*x0[i]+w*(summa1+summa2+(b[i]/A[i][i]))
  #w += 0.1
    n+=1
    vect.append(list(x0))
  return x1,n,w,epsilons,vect


solved, iterations, param, error2, vect = Relax(A, x0, eps)
# par=[par for par in np.arange(0,1.6,0.1)]
# spisok=[610,290,184,130,98,77,62,50,41,34,28,23,19,25,38,77]
# plt.plot(par,spisok,label='Зависимость числа итераций от значения параметра')
# plt.legend()
# plt.show()
print("Ответ алгоритма: " + str(solved))
print("Количество итераций: " + str(iterations))
#print("Оптимальное значение параметра: " + str(param))
print("Ошибки: " + str(error2))
# pd.DataFrame(vect).to_excel('relax.xlsx')

# plt.plot(error)
# plt.plot(error,label='График погрешности для метода релаксации')
# plt.xlabel('Кол-во итераций')
# plt.ylabel('Величина невязки')
# plt.legend()
#plt.show()




#Метод наискорейшего спуска
x0=np.zeros(n)
def spusk(A,b,x0,eps):
    x1=x0
    n=0
    errors=[]
    vect = []
    for i in range(50):#while max(abs(A.dot(x1) - b)) > eps:
        errors.append(max(abs(A.dot(x1) - b)))
        #print(max(abs(A.dot(x1) - b)))
        x0 = x1
        r=A.dot(x0)-b
        tau=((A.dot(r)).dot(r))/((A.dot(r)).dot((A.dot(r))))
        #tau = (r.dot(r)) / ((A.dot(r)).dot(r))
        x1 = x0 - tau*r

        n += 1
        vect.append(list(x0))

    return x1, n,errors,vect
x1,iterations,error3,vect=spusk(A,b,x0,eps)

print("Ответ алгоритма: " + str(x1))
print("Количество итераций: " + str(iterations))
print("Ошибки: " + str(error3))
# vct=pd.DataFrame(vect)

#vct=vct.apply(lambda x:print(x))
#vct.to_excel('spusk.xlsx')
#
plt.plot(error1,label='График погрешности для метода Зейделя')
plt.plot(error2,label='График погрешности для метода релаксации')
plt.plot(error3,label = 'Невязка метода наискор. спуска')
plt.xlabel('Итерации')
plt.ylabel('Невязка')
plt.legend()
plt.show()



# start=time()
# y=f(1)
# end=time()
# print(end-start)