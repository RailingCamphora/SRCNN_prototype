import numpy as np
import matplotlib.pyplot as plt

# grid \Omega = [0,1]
n = 100
h = 1/(n-1)
x = np.linspace(0,1,n)

# parameters
sigma = 1e-1

# make data
u = np.heaviside(x - 0.2,0)
f_delta = u + sigma*np.random.randn(n)






###iteration preparation
# FD differentiation matrix
D = (np.diag(np.ones(n-1),1) - np.diag(np.ones(n),0))/h
a=1/np.linalg.norm(D)**2
D1=np.transpose(D)
def barrer(v,l=90):
    v_abs=np.abs(v)
    v[v_abs>l]=v[v_abs>l]/v_abs[v_abs>l]*l
    return v

def iter_denosing(f,v=np.ones(100)):
    for k in range(10):
        v=barrer(v-a*((np.dot(v,D1)-f)@D))
    return v@D1


# plot
plt.subplot(121)
plt.plot(x,u,x,f_delta)
plt.xlabel(r'$x$')


plt.subplot(122)
u1=iter_denosing(f_delta,u@np.linalg.inv(D1))
plt.plot(x,u1)
plt.show()
