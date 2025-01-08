# Teoria_systemow

#deklaracja wartości s oraz podanej Transmitacji
ω = sympy.Symbol('ω', real = True)

s = 0 + sympy.I*ω

H = ((N + 1)*s + 1)/(2*s**2 + s + N + 5)


# Charakteryka amplitudaowa i fazowa
re =  sympy.re(H)
im = sympy.im(H)

modul =  sympy.sqrt(re**2 + im**2)
faza = sympy.atan(im/re)

#wykres charakterstki amplitudowaej i fazowej

w = np.logspace(-2, 2, 10000)
Mod = sympy.lambdify(ω, modul, 'numpy')(w)
P = sympy.lambdify(ω, faza, 'numpy')(w)

fig,axs=plt.subplots(1,2)
axs[0].loglog(w,Mod)
axs[0].set_title("charakterystyka amplitudowa")
axs[0].set_xlabel("omega")
axs[0].set_ylabel("|H(jω)|")
axs[0].grid()

axs[1].semilogx(w,P)
axs[1].set_title("charakterystyka fazowa")
axs[1].set_xlabel("omega")
axs[1].set_ylabel("phi(w)/pi")
axs[1].grid()

#Odpowiedz impulsowa

A = np.poly1d([N + 1, 1])
B = np.poly1d([2, 1, N + 5])

def calculateImpulseResponseFromNumDen(A, B, TimeDomain):
    step = scipy.signal.impulse(scipy.signal.tf2ss(B, A), T=TimeDomain)[1]
    return step

time = np.linspace(0, 30, 2000)
impulse = calculateImpulseResponseFromNumDen(B, A, time)

figI, axI = plt.subplots(1, 1)
axI.plot(time, impulse)
axI.set_title("odpowiedź impulsowa")
axI.grid()
figI.tight_layout()  


B2 = np.poly1d([1, 5, 7, 2*N, 1, 0, 2])
A1 = np.poly1d([1, (0.5*N + 1), 3])

poles = np.roots(A1)

fig, ax = plt.subplots(1, 1)
ax.plot(poles.real, poles.imag,'rx') 
ax.set_title("bieguny transmitancji")
ax.set_xlabel("Re") 
ax.set_ylabel("Im")
ax.set_xlim([-12, 12])
ax.set_ylim([-12, 12])
ax.grid()
fig.tight_layout()


t = sympy.Symbol('t', real = 'True', nonzero = 'True')

NumberOfSamples = 1001

T = 1
#T = T/2

fN = sympy.Piecewise((4*t/T + 2, t < -T/4), (-4*t/T, (t>-T/4)&(t < T/4)), (4*t/T - 2, t > T/4))

fun = sympy.lambdify(t, fN, 'numpy')
lin = np.linspace(-T/2, T/2, NumberOfSamples)
x = fun(lin)
plt.plot(lin, x)
plt.grid()

...

t = sympy.Symbol('t', real = 'True')
k = sympy.Symbol('k', real = 'True', nonzero = True, integer=True)

foo = sympy.Piecewise((-N, (t>-1) & (t<-1/2)), (1, (t>-1/2) & (t<1/2)), (-N, (t>1/2) & (t<1)))

def wyznacz_wspolczynniki_szeregu(funkcja, początek_zakresu, koniec_zakresu):
    T = abs(początek_zakresu) + abs(koniec_zakresu)
    kernel =  sympy.exp((-sympy.I*2*k*sympy.pi*t)/T)
    F0 = 1 / T * sympy.integrate(funkcja, początek_zakresu, koniec_zakresu)
    Fk = 1 / T * sympy.integrate(funkcja * kernel, (t, początek_zakresu, koniec_zakresu))
    return F0, Fk

F0, Fk = wyznacz_wspolczynniki_szeregu(foo, -1, T)
print('F0:')
display(F0)
print('Fk:')
display(Fk)

def oblicz_wartosci_wspolczynnikow(F0, Fk, T):
    coeffF0=np.array(F0,dtype=np.cdouble)
    coeffFk=np.cdouble(sympy.lambdify(k,Fk,'numpy')(np.arange()))
    Fk_vector = np.append(coeffF0, coeffFk)
    return Fk_vector

Fk_vector = oblicz_wartosci_wspolczynnikow(F0, Fk, T)

fig, ax = plt.subplots(1, 2)
## Wykres amplitud
ax[0].stem(...)
ax[0].set_title(...)
ax[0].set_xlabel(...)
ax[0].set_ylabel(...)
## wykres faz
ax[1].stem(...)
ax[1].set_title(...)
ax[1].set_xlabel(...)
ax[1].set_ylabel(...)
fig.show()

## Rekonstrukcja (dla wybitnych):
