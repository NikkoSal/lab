import numpy as np

# Поиск шага методом золотого сечения

def line_search(f, x, grad, a=0.0, b=1.0, tol=1e-6):
    phi = (1 + np.sqrt(5)) / 2

    def phi_func(alpha):
        return f(x - alpha * grad)

    c = b - (b - a) / phi
    d = a + (b - a) / phi

    while abs(b - a) > tol:
        if phi_func(c) < phi_func(d):
            b = d
        else:
            a = c
        c = b - (b - a) / phi
        d = a + (b - a) / phi

    return (a + b) / 2


# Метод наискорейшего градиентного спуска

def steepest_descent(f, grad_f, x0, eps=1e-6, max_iter=1000):
    x = x0.astype(float)
    history = [x.copy()]

    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < eps:
            print(f"Сходимость достигнута за {k} итераций")
            break

        alpha = line_search(f, x, grad)
        x = x - alpha * grad
        history.append(x.copy())

    return x, history


# квадратичная функция
# f(x) = x1^2 + x2^2

def f1(x):
    return x[0]**2 + x[1]**2

def grad_f1(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([5.0, -3.0])
xmin, history = steepest_descent(f1, grad_f1, x0)

print("\nТест 1")
print("Минимум:", xmin)
print("Значение функции:", f1(xmin))


# эллипсоид
# f(x) = 10x1^2 + x2^2

def f2(x):
    return 10*x[0]**2 + x[1]**2

def grad_f2(x):
    return np.array([20*x[0], 2*x[1]])

x0 = np.array([3.0, 3.0])
xmin, history = steepest_descent(f2, grad_f2, x0)

print("\nТест 2")
print("Минимум:", xmin)
print("Значение функции:", f2(xmin))


#функция Розенброка

def f3(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_f3(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

x0 = np.array([-1.2, 1.0])
xmin, history = steepest_descent(f3, grad_f3, x0, max_iter=5000)

print("\nТест 3 (Розенброк)")
print("Минимум:", xmin)
print("Значение функции:", f3(xmin))
