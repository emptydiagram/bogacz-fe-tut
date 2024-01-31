import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate as integrate

def ex1():
    v_p = 3
    Sigma_p = 1
    Sigma_u = 1
    g = lambda v: v**2
    probv = scipy.stats.norm(v_p, Sigma_p).pdf
    probuv = lambda v: scipy.stats.norm(g(v), Sigma_u).pdf

    u = 2
    probu = integrate.quad(lambda v: probv(v) * probuv(v)(u), -np.inf, np.inf)[0]
    
    probvu = lambda v: probv(v) * probuv(v)(u) / probu

    vs = np.linspace(0, 5, 100)
    plt.figure(figsize=(6,4))
    plt.plot(vs, probvu(vs))
    plt.xlabel('v')
    plt.ylabel('probability')
    plt.legend()
    plt.show()

def ex2():
    v_p = 3
    Sigma_p = 1
    Sigma_u = 1
    g = lambda v: v**2

    u = 2
    lr = 0.01

    phi = v_p
    num_secs = 5
    ts = [0]
    phis = [phi]
    for i in range(num_secs * 100):
        phi += lr *( (v_p - phi)/Sigma_p + (u - g(phi)) * (2 * phi)/Sigma_u)
        phis.append(phi)
        ts.append(ts[-1] + lr)

    plt.figure(figsize=(6,4))
    plt.plot(ts, phis)
    plt.xlabel('t')
    plt.ylabel('phi')
    plt.ylim(0, 4)
    plt.legend()
    plt.show()
    
def ex3():
    v_p = 3
    Sigma_p = 1
    Sigma_u = 1
    g = lambda v: v**2
    g_prime = lambda v: 2 * v

    u = 2
    e_u = e_p = 0
    phi = v_p
    dt = 0.01
    num_secs=  5
    steps_per_sec = 100

    ts = [0]
    eus = [e_u]
    eps = [e_p]
    phis = [phi]

    for i in range(num_secs * steps_per_sec):
        print(u - g(phi))
        print(phi - v_p)
        e_u += dt * (u - g(phi) - Sigma_u * e_u)
        e_p += dt * (phi - v_p - Sigma_p * e_p)
        phi += dt * (e_u * g_prime(phi) - e_p)

        ts.append(ts[-1] + dt)
        eus.append(e_u)
        eps.append(e_p)
        phis.append(phi)

    plt.figure(figsize=(6,4))
    plt.plot(ts, phis, label='phi')
    plt.plot(ts, eus, label='e_u')
    plt.plot(ts, eps, label='e_p')

    plt.xlabel('t')
    plt.ylim(-2, 3.5)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # ex1()
    # ex2()
    ex3()

