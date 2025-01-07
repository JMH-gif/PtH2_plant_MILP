from gekko import GEKKO

m = GEKKO()
x1 = m.Var(value=2,lb=0,ub=2)
x2 = m.Var(value=3,lb=0,ub=2)
x3 = m.Var(value=4,lb=0,ub=2)
m.Equation(x1 + x2 + x3 == 5)
m.Equation(x2 + x3 <= 4)
m.Minimize(0.5*x1 + 0.8*x2 + 0.7*x3)
m.solve(disp=False)
print('Optimal cost: ' + str(m.options.objfcnval))
print('x1: ' + str(x1.value[0]))
print('x2: ' + str(x2.value[0]))
print('x3: ' + str(x3.value[0]))

