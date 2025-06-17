# This code computes quantum imaginary time evolution (QITE) for Eq.7 of Kavaki and Lewis,arXiv:2503.01119 [hep-lat] (2025).
# It computes imaginary time evolution of SU(2) lattice gauge theory on a periodic 12-plaquette lattice, including self-mitigation.
# The code is available at https://github.com/randylewis/SelfMitigatedQITE
# If you find this code useful, please cite the paper as well as the code.
# Note: The code uses IBM's open-source software called qiskit, available from https://www.ibm.com/quantum/qiskit
#
# randy.lewis@yorku.ca

# Define user choices.
myhardware = 1                                   # Choose the qubit hardware (1 for brisbane, 2 for kyoto, 3 for osaka, 4 for error-free simulator.)
myqubits = [28,29,30,31,32,36,51,50,49,48,47,35] # Choose the specific qubits to be used.
mycoupling = 1.0                                 # Choose the value for x = 2/g^4 from Eq.7 of Kavaki and Lewis,arXiv:2503.01119.
mytimestep = 0.1                                 # Choose the imaginary time step, Delta tau.
myruns = 50                                      # Choose the number of times you want to run this physics circuit. Each run will have a new CNOT randomization.
myshots = 1000                                   # Choose the number of shots for each run.
myconfidencelevel = 0.95                         # Choose 0.95, for example, for error bars that represent a 95% confidence level.
myrcond = 0.5                                    # Choose the reciprocal condition number to be used in the Moore-Penrose inverse.
print("Input:",myhardware,myqubits,mycoupling,mytimestep,myruns,myshots,myconfidencelevel,myrcond)
Nplaq = len(myqubits)

# Import tools from qiskit, and load my IBM Q account.
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, result
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider import IBMProvider
provider = IBMProvider()

# Import standard python tools.
import datetime
from numpy import pi, sqrt, zeros, dot, mean
from numpy.linalg import pinv
from scipy.stats import norm, bootstrap
from scipy.optimize import minimize
from random import random, randrange

# Identify the hardware that will be used.
if myhardware == 1:
    chosenhardware = "ibm_brisbane"
elif myhardware == 2:
    chosenhardware = "ibm_kyoto"
elif myhardware == 3:
    chosenhardware = "ibm_osaka"
else:
    chosenhardware = "ibmq_qasm_simulator"
backend = provider.get_backend(chosenhardware,instance=f"ibm-q/open/main")

# List results for expectation values from previous time steps: <X0>,<X1>,<X2>,...<Z0>,<Z1>,<Z2>,...,<X1X0>,...,etc.
Xdata = []
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Xdata.append( [0] )
Zdata = []
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
Zdata.append( [1] )
XXdata = []
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XXdata.append( [0] )
XZdata = []
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
XZdata.append( [0] )
YYdata = []
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
YYdata.append( [0] )
ZXdata = []
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZXdata.append( [0] )
ZZdata = []
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZZdata.append( [1] )
ZXZdata = []
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )
ZXZdata.append( [0] )

# List results from the mitigation runs of previous time steps: <Z0>, <Z1>, <Z2>, ..., <Z1 Z0>, <Z2 Z1>, <Z3 Z2>, ...
Zgate = []
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
Zgate.append( [1] )
ZZgate = []
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZgate.append( [1] )
ZZZgate = []
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )
ZZZgate.append( [1] )

# Define the vector b at every time step.
b = []
for tau in range(len(Xdata[0])):
    btau = []
    for j in range(Nplaq):
        jm = (Nplaq+j-1)%Nplaq
        btau.append(3/2*Xdata[j][tau]+3/4*ZXdata[j][tau]+3/4*XZdata[jm][tau]+mycoupling*(-9/4*Zdata[j][tau]+3/4*XXdata[jm][tau]+3/4*XXdata[j][tau]))
        btau.append(-3/2*YYdata[jm][tau]+3/2*XXdata[jm][tau]+mycoupling*(-9/4*ZXdata[jm][tau]+3/4*Xdata[j][tau]))
        btau.append(3/2*XXdata[j][tau]-3/2*YYdata[j][tau]+mycoupling*(-9/4*XZdata[j][tau]+3/4*Xdata[j][tau]))
        btau.append(3/2*XZdata[jm][tau]+3/4*Xdata[j][tau]+mycoupling*(9/4*YYdata[jm][tau]-9/4*ZZdata[jm][tau]-3/4*Zdata[j][tau]-1/4*ZZdata[j][tau]))
        btau.append(3/2*ZXdata[j][tau]+3/4*Xdata[j][tau]+mycoupling*(-9/4*ZZdata[j][tau]+9/4*YYdata[j][tau]-3/4*Zdata[j][tau]-1/4*ZZdata[jm][tau]))
    b.append(btau)

# Define the matrix S + S^T and calculate its inverse at every time step.
SSTinverse = []
for tau in range(len(Xdata[0])):
    SplusST = zeros((60,60))
    for j in range(Nplaq):
        jm = (Nplaq+j-1)%Nplaq
        jp = (j+1)%Nplaq
        SplusST[5*j,5*jm] = 2*YYdata[jm][tau]
        SplusST[5*j+1,5*jm+2] = 2*ZZdata[jm][tau]
        SplusST[5*j+1,5*jm+4] = -2*XZdata[jm][tau]
        SplusST[5*j+3,5*jm+2] = -2*ZXdata[jm][tau]
        SplusST[5*j+3,5*jm+4] = 2*XXdata[jm][tau]
        SplusST[5*j,5*j] = 2
        SplusST[5*j,5*j+1] = 2*Xdata[jm][tau]
        SplusST[5*j,5*j+2] = 2*Xdata[j][tau]
        SplusST[5*j,5*j+3] = 2*Zdata[jm][tau]
        SplusST[5*j,5*j+4] = 2*Zdata[j][tau]
        SplusST[5*j+1,5*j] = 2*Xdata[jm][tau]
        SplusST[5*j+1,5*j+1] = 2
        SplusST[5*j+2,5*j] = 2*Xdata[j][tau]
        SplusST[5*j+2,5*j+2] = 2
        SplusST[5*j+3,5*j] = 2*Zdata[jm][tau]
        SplusST[5*j+3,5*j+3] = 2
        SplusST[5*j+4,5*j] = 2*Zdata[j][tau]
        SplusST[5*j+4,5*j+4] = 2
        SplusST[5*j,5*jp] = 2*YYdata[j][tau]
        SplusST[5*j+2,5*jp+2] = 2*ZZdata[j][tau]
        SplusST[5*j+2,5*jp+4] = -2*ZXdata[j][tau]
        SplusST[5*j+4,5*jp+2] = -2*XZdata[j][tau]
        SplusST[5*j+4,5*jp+4] = 2*XXdata[j][tau]
    temp = pinv(SplusST,rcond=myrcond,hermitian=True).tolist()
    SSTinverse.append(temp)

# Calculate the coefficients in the operator A of the time evolution factor exp(-i A DeltaTau/2) at every time step.
thetaY = []
thetaYX = []
thetaXY = []
thetaYZ = []
thetaZY = []
for tau in range(len(Xdata[0])):
    a = -dot(SSTinverse[tau],b[tau])
    oneY = []
    oneYX = []
    oneXY = []
    oneYZ = []
    oneZY = []
    for j in range(Nplaq):
        oneY.append(mytimestep*a[5*j])
        oneYX.append(mytimestep*a[5*j+1])
        oneXY.append(mytimestep*a[5*j+2])
        oneYZ.append(mytimestep*a[5*j+3])
        oneZY.append(mytimestep*a[5*j+4])
    thetaY.append(oneY)
    thetaYX.append(oneYX)
    thetaXY.append(oneXY)
    thetaYZ.append(oneYZ)
    thetaZY.append(oneZY)

# This function uses the Pauli gates that precede a randomized CNOT to define the Pauli gates that follow the randomized CNOT.
def RSfromPQ(P,Q):
    if P==0 and Q==0:
        R = 0
        S = 0
    elif P==0 and Q==1:
        R = 0
        S = 1
    elif P==0 and Q==2:
        R = 3
        S = 2
    elif P==0 and Q==3:
        R = 3
        S = 3
    elif P==1 and Q==0:
        R = 1
        S = 1
    elif P==1 and Q==1:
        R = 1
        S = 0
    elif P==1 and Q==2:
        R = 2
        S = 3
    elif P==1 and Q==3:
        R = 2
        S = 2
    elif P==2 and Q==0:
        R = 2
        S = 1
    elif P==2 and Q==1:
        R = 2
        S = 0
    elif P==2 and Q==2:
        R = 1
        S = 3
    elif P==2 and Q==3:
        R = 1
        S = 2
    elif P==3 and Q==0:
        R = 3
        S = 0
    elif P==3 and Q==1:
        R = 3
        S = 1
    elif P==3 and Q==2:
        R = 0
        S = 2
    elif P==3 and Q==3:
        R = 0
        S = 3
    return R, S

# This function prepares the randomized Pauli gates for the first CNOTs.
def firstCNOTs():
    global thisRS, nextPQ
    for j in range(Nplaq):
        thisRS[j] = 0
        nextPQ[j] = randrange(4)

# This function applies a CNOT gate, records the Pauli gates that follow it, and defines Pauli gates to precede the next CNOT.
def applyCNOT(ctrl,targ):
    global thisRS, nextPQ
    P = nextPQ[ctrl]
    Q = nextPQ[targ]
    circ.cx(qreg[ctrl],qreg[targ])
    R,S = RSfromPQ(P,Q)
    thisRS[ctrl] = R
    thisRS[targ] = S
    nextPQ[ctrl] = randrange(4)
    nextPQ[targ] = randrange(4)

# This function applies a Pauli gate, then the identity, then another Pauli gate.
def applyI(qubit):
    global thisRS, nextPQ
    if thisRS[qubit]==0 and nextPQ[qubit]==1:
        circ.x(qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==2:
        circ.y(qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==3:
        circ.z(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==0:
        circ.x(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==2:
        circ.z(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==3:
        circ.y(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==0:
        circ.y(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==1:
        circ.z(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==3:
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==0:
        circ.z(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==1:
        circ.y(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==2:
        circ.x(qreg[qubit])

# This function applies a Pauli gate, then RY(theta), then another Pauli gate.
def applyRY(qubit,theta):
    global thisRS, nextPQ
    if thisRS[qubit]==0 and nextPQ[qubit]==0:
        circ.ry(theta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==1:
        circ.ry(theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==2:
        circ.ry(pi+theta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==3:
        circ.ry(pi+theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==0:
        circ.ry(-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==1:
        circ.ry(-theta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==2:
        circ.ry(pi-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==3:
        circ.ry(pi-theta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==0:
        circ.ry(pi+theta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==1:
        circ.ry(pi+theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==2:
        circ.ry(theta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==3:
        circ.ry(theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==0:
        circ.ry(pi-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==1:
        circ.ry(pi-theta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==2:
        circ.ry(-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==3:
        circ.ry(-theta,qreg[qubit])

# This function inserts the beginning of the very first Trotter step.
def Aportion(tau):
    for j in range(0,Nplaq,2):
        applyRY(j,thetaXY[tau][j])
        applyRY(j+1,thetaYZ[tau][j+1])

# This function inserts the bulk of the first half of the Trotter step, forward (direction=1) or backward (direction=-1) in time.
def Bportion(tau,direction):
    for j in range(0,Nplaq,2):
        applyCNOT(j,j+1)
    for j in range(Nplaq):
        applyI(j)
    for j in range(1,Nplaq,2):
        jp = (j+1)%Nplaq
        applyCNOT(j,jp)
    for j in range(0,Nplaq,2):
        applyRY(j,direction*thetaYZ[tau][j])
        applyRY(j+1,direction*thetaXY[tau][j+1])
    for j in range(1,Nplaq,2):
        jp = (j+1)%Nplaq
        applyCNOT(j,jp)
    for j in range(Nplaq):
        applyRY(j,direction*thetaY[tau][j])
    for j in range(0,Nplaq,2):
        applyCNOT(j+1,j)
    for j in range(0,Nplaq,2):
        applyRY(j,direction*thetaZY[tau][j])
        applyRY(j+1,direction*thetaYX[tau][j+1])
    for j in range(0,Nplaq,2):
        applyCNOT(j+1,j)
    for j in range(Nplaq):
        applyI(j)
    for j in range(1,Nplaq,2):
        jp = (j+1)%Nplaq
        applyCNOT(jp,j)

# This function inserts the center portion of the Trotter step, forward (direction=1), backward (direction=-1) or nowhere (direction=0) in time.
def Cportion(tau,direction):
    for j in range(0,Nplaq,2):
        applyRY(j,2*direction*thetaYX[tau][j])
        applyRY(j+1,2*direction*thetaZY[tau][j+1])
    if direction==0:
        circ.barrier(qreg)

# This function inserts the bulk of the second half of the Trotter step, forward (direction=1) or backward (direction=-1) in time.
def Dportion(tau,direction):
    for j in range(1,Nplaq,2):
        jp = (j+1)%Nplaq
        applyCNOT(jp,j)
    for j in range(Nplaq):
        applyI(j)
    for j in range(0,Nplaq,2):
        applyCNOT(j+1,j)
    for j in range(0,Nplaq,2):
        applyRY(j,direction*thetaZY[tau][j])
        applyRY(j+1,direction*thetaYX[tau][j+1])
    for j in range(0,Nplaq,2):
        applyCNOT(j+1,j)
    for j in range(Nplaq):
        applyRY(j,direction*thetaY[tau][j])
    for j in range(1,Nplaq,2):
        jp = (j+1)%Nplaq
        applyCNOT(j,jp)
    for j in range(0,Nplaq,2):
        applyRY(j,direction*thetaYZ[tau][j])
        applyRY(j+1,direction*thetaXY[tau][j+1])
    for j in range(1,Nplaq,2):
        jp = (j+1)%Nplaq
        applyCNOT(j,jp)
    for j in range(Nplaq):
        applyI(j)
    for j in range(0,Nplaq,2):
        applyCNOT(j,j+1)

# This function inserts the transition between Trotter steps, forward (direction=1), backward (direction=-1) or nowhere (direction=0) in time.
def Eportion(tau1,tau2,direction1,direction2):
    for j in range(0,Nplaq,2):
        thetaj = direction1*thetaXY[tau1][j] + direction2*thetaXY[tau2][j]
        thetajp = direction1*thetaYZ[tau1][j+1] + direction2*thetaYZ[tau2][j+1]
        applyRY(j,thetaj)
        applyRY(j+1,thetajp)

# This function inserts the end of the very last Trotter step.
def Fportion(tau,direction):
    global nextPQ
    for j in range(0,Nplaq,2):
        applyRY(j,direction*thetaXY[tau][j])
        applyRY(j+1,direction*thetaYZ[tau][j+1])
    for j in range(0,Nplaq,2):
        applyCNOT(j,j+1)
        nextPQ[j] = 0
        nextPQ[j+1] = 0
        applyI(j)
        applyI(j+1)

# Build the eight circuits for mitigation of readout errors.
circlist = []
for i in range(8):
    qreg = QuantumRegister(Nplaq)
    creg = ClassicalRegister(Nplaq)
    circ = QuantumCircuit(qreg,creg)
    if i==1:
        for j in range(0,Nplaq,3):
            circ.x(qreg[j])
    elif i==2:
        for j in range(0,Nplaq,3):
            circ.x(qreg[j+1])
    elif i==3:
        for j in range(0,Nplaq,3):
            circ.x(qreg[j])
            circ.x(qreg[j+1])
    elif i==4:
        for j in range(0,Nplaq,3):
            circ.x(qreg[j+2])
    elif i==5:
        for j in range(0,Nplaq,3):
            circ.x(qreg[j])
            circ.x(qreg[j+2])
    elif i==6:
        for j in range(0,Nplaq,3):
            circ.x(qreg[j+1])
            circ.x(qreg[j+2])
    elif i==7:
        for j in range(0,Nplaq,3):
            circ.x(qreg[j])
            circ.x(qreg[j+1])
            circ.x(qreg[j+2])
    circ.measure(qreg,creg)
    circlist.append(circ)

# Create the circuits that evolve through previous time steps, which will lead to the physics expectation values.
CircuitOrder = ["Z1Z0","Z1X0","X1Z0","X1X0","Y1Y0"]
ncircuits = len(CircuitOrder)
thisRS = [999]*Nplaq
nextPQ = [999]*Nplaq
for i in range(ncircuits):
    for k in range(myruns):
        qreg = QuantumRegister(Nplaq)
        creg = ClassicalRegister(Nplaq)
        circ = QuantumCircuit(qreg,creg)
        if len(Xdata[0]) > 0:
            firstCNOTs()
            Aportion(0)
            for tau in range(len(Xdata[0])):
                Bportion(tau,1)
                Cportion(tau,1)
                Dportion(tau,1)
                if tau+1 < len(Xdata[0]):
                    tau2 = tau + 1
                    Eportion(tau,tau2,1,1)
                else:
                    Fportion(tau,1)
        if i==1:
            for j in range(0,Nplaq,2):
                circ.ry(-pi/2,qreg[j])
        elif i==2:
            for j in range(0,Nplaq,2):
                circ.ry(-pi/2,qreg[j+1])
        elif i==3:
            for j in range(Nplaq):
                circ.ry(-pi/2,qreg[j])
        elif i==4:
            for j in range(Nplaq):
                circ.rx(pi/2,qreg[j])
        circ.measure(qreg,creg)
        circlist.append(circ)

# Create mitigation circuits that evolve forward then backward in time, leading to <0|Z|0> and <0|ZZ|0>.
for k in range(myruns):
    qreg = QuantumRegister(Nplaq)
    creg = ClassicalRegister(Nplaq)
    circ = QuantumCircuit(qreg,creg)
    if len(Xdata[0]) > 0:
        firstCNOTs()
        Aportion(0)
        for tau in range(len(Xdata[0])):
            tau1 = min( tau, len(Xdata[0])-1-tau )
            tau2 = min( tau+1, len(Xdata[0])-2-tau )
            if 2*tau < len(Xdata[0]):
                Bdirection = 1
            else:
                Bdirection = -1
            if 2*tau+1 < len(Xdata[0]):
                Cdirection = 1
            elif 2*tau+1 == len(Xdata[0]):
                Cdirection = 0
            else:
                Cdirection = -1
            if 2*tau+1 < len(Xdata[0]):
                Ddirection = 1
            else:
                Ddirection = -1
            if 2*tau+2 < len(Xdata[0]):
                Edirection1 = 1
                Edirection2 = 1
            elif 2*tau+2 == len(Xdata[0]):
                Edirection1 = 1
                Edirection2 = -1
            elif tau+1 < len(Xdata[0]):
                Edirection1 = -1
                Edirection2 = -1
            Bportion(tau1,Bdirection)
            Cportion(tau1,Cdirection)
            Dportion(tau1,Ddirection)
            if tau+1 < len(Xdata[0]):
                Eportion(tau1,tau2,Edirection1,Edirection2)
            else:
                Fportion(tau1,-1)
    circ.measure(qreg,creg)
    circlist.append(circ)

# Run all circuits.
print("Queuing the circuits on the hardware at",datetime.datetime.now())
job = execute(circlist,backend,initial_layout=myqubits,shots=myshots)
print("The job identifier is",job.job_id())
print("The hardware has returned results at",datetime.datetime.now())

# Obtain the results.
circoutput = job.result().get_counts()

# Assign longer dictionary keys to the output from each circuit.
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for i in range(8):
    for ibinary in range(2**Nplaq):
        instring = '{0:012b}'.format(ibinary)
        if instring in circoutput[i]:
            outstring = "readout"
            for iplaq in range(Nplaq):
                jplaq = Nplaq - 1 - iplaq
                outstring += alphabet[jplaq:jplaq+1]+instring[iplaq:iplaq+1]
            circoutput[i][outstring] = circoutput[i].pop(instring)
k = 7
for i in range(ncircuits):
    for j in range(myruns):
        k += 1
        for ibinary in range(2**Nplaq):
            instring = '{0:012b}'.format(ibinary)
            if instring in circoutput[k]:
                outstring = CircuitOrder[i]+"physics"
                for iplaq in range(Nplaq):
                    jplaq = Nplaq - 1 - iplaq
                    outstring += alphabet[jplaq:jplaq+1]+instring[iplaq:iplaq+1]
                circoutput[k][outstring] = circoutput[k].pop(instring)
for j in range(myruns):
    k += 1
    for ibinary in range(2**Nplaq):
            instring = '{0:012b}'.format(ibinary)
            if instring in circoutput[k]:
                outstring = "ZZZgate"
                for iplaq in range(Nplaq):
                    jplaq = Nplaq - 1 - iplaq
                    outstring += alphabet[jplaq:jplaq+1]+instring[iplaq:iplaq+1]
                circoutput[k][outstring] = circoutput[k].pop(instring)

# Collect the counts for the readout error mitigation matrix for each triplet of neighbouring plaquettes.
CalibrationMatrix = []
row = [ [0,1,2,3,4,5,6,7], [0,2,4,6,1,3,5,7], [0,4,1,5,2,6,3,7] ]
for iplaq in range(Nplaq):
    jplaq = (iplaq+1)%Nplaq
    kplaq = (iplaq+2)%Nplaq
    newMatrix = []
    for i in range(8):
        newMatrix.append([0,0,0,0,0,0,0,0])
    for i in range(8):
        for j,(key,value) in enumerate(circoutput[row[iplaq%3][i]].items()):
            if (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"0" in key):
                newMatrix[0][i] += value/myshots
            elif (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"1" in key):
                newMatrix[1][i] += value/myshots
            elif (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"0" in key):
                newMatrix[2][i] += value/myshots
            elif (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"1" in key):
                newMatrix[3][i] += value/myshots
            elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"0" in key):
                newMatrix[4][i] += value/myshots
            elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"1" in key):
                newMatrix[5][i] += value/myshots
            elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"0" in key):
                newMatrix[6][i] += value/myshots
            elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"1" in key):
                newMatrix[7][i] += value/myshots
    CalibrationMatrix.append(newMatrix)
    print("The calibration matrix for qubits",alphabet[kplaq],alphabet[jplaq],alphabet[iplaq],"is")
    for i in CalibrationMatrix[iplaq]:
        print(" ".join(map(str,i)))

# Collect the counts for the physics circuits.
PhysicsMatrix = []
n = 7
for k in range(ncircuits):
    PhysicsVector = []
    for i in range(myruns):
        n += 1
        newVector = []
        for iplaq in range(Nplaq):
            jplaq = (iplaq+1)%Nplaq
            kplaq = (iplaq+2)%Nplaq
            newVector.append([0,0,0,0,0,0,0,0])
            for j,(key,value) in enumerate(circoutput[n].items()):
                if (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"0" in key):
                    newVector[iplaq][0] += value
                elif (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"1" in key):
                    newVector[iplaq][1] += value
                elif (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"0" in key):
                    newVector[iplaq][2] += value
                elif (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"1" in key):
                    newVector[iplaq][3] += value
                elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"0" in key):
                    newVector[iplaq][4] += value
                elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"1" in key):
                    newVector[iplaq][5] += value
                elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"0" in key):
                    newVector[iplaq][6] += value
                elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"1" in key):
                    newVector[iplaq][7] += value
        PhysicsVector.append(newVector)
    PhysicsMatrix.append(PhysicsVector)

# Collect the counts for the gate-error mitigation circuits.
n = 7 + myruns*ncircuits
GateVector = []
for i in range(myruns):
    n += 1
    newVector = []
    for iplaq in range(Nplaq):
        jplaq = (iplaq+1)%Nplaq
        kplaq = (iplaq+2)%Nplaq
        newVector.append([0,0,0,0,0,0,0,0])
        for j,(key,value) in enumerate(circoutput[n].items()):
            if (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"0" in key):
                newVector[iplaq][0] += value
            elif (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"1" in key):
                newVector[iplaq][1] += value
            elif (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"0" in key):
                newVector[iplaq][2] += value
            elif (alphabet[kplaq]+"0" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"1" in key):
                newVector[iplaq][3] += value
            elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"0" in key):
                newVector[iplaq][4] += value
            elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"0" in key) and (alphabet[iplaq]+"1" in key):
                newVector[iplaq][5] += value
            elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"0" in key):
                newVector[iplaq][6] += value
            elif (alphabet[kplaq]+"1" in key) and (alphabet[jplaq]+"1" in key) and (alphabet[iplaq]+"1" in key):
                newVector[iplaq][7] += value
    GateVector.append(newVector)

# This function accounts for readout errors by using sequential least squares programming (SLSQP).
def ReadoutMitigation(Vector):
    counts = []
    for i in range(myruns):
        newcount0 = []
        newcount1XOR0 = []
        newcount2XOR1XOR0 = []
        for iplaq in range(Nplaq):
            def fun(vect):
                return sum((Vector[i][iplaq] - dot(CalibrationMatrix[iplaq],vect))**2)
            BestVector = Vector[i][iplaq]
            constraints = ({'type': 'eq', 'fun': lambda x: myshots-sum(x)})
            bounds = tuple((0, myshots) for x in BestVector)
            leastsquares = minimize(fun, BestVector, method='SLSQP', constraints=constraints, bounds=bounds, tol=1e-6)
            BestVector = leastsquares.x
            newcount0.append(BestVector[1] + BestVector[3] + BestVector[5] + BestVector[7])
            newcount1XOR0.append(BestVector[1] + BestVector[2] + BestVector[5] + BestVector[6])
            newcount2XOR1XOR0.append(BestVector[1] + BestVector[2] + BestVector[4] + BestVector[7])
        counts.append(newcount0+newcount1XOR0+newcount2XOR1XOR0)
    return counts

# Account for readout errors by using sequential least squares programming (SLSQP).
allcounts = []
for i in range(ncircuits):
    counts = ReadoutMitigation(PhysicsMatrix[i])
    allcounts.append(counts)
counts = ReadoutMitigation(GateVector)
allcounts.append(counts)

# This function calculates a central value and error bar, given that "shots1" measurements gave 1 and "shotstotal-shots1" gave 0.
def WilsonScore(shots1):
    z = norm.ppf((1+myconfidencelevel)/2)
    denom = 1 + z**2/myshots
    frac1 = shots1/myshots
    centralvalue = ( frac1 + z**2/(2*myshots) )/denom
    errorbar = z*sqrt( frac1*(1-frac1)/myshots + z**2/(4*myshots**2) )/denom
    return centralvalue,errorbar

# This function calculates an error bar by bootstrapping a set of values.
def BootstrapError(inputmean):
    finalmean = mean(inputmean)
    bootsample = (inputmean,)
    answer = bootstrap(bootsample, mean, n_resamples=10*len(inputmean), confidence_level=myconfidencelevel)
    finalerrorbar = (answer.confidence_interval[1]-answer.confidence_interval[0])/2
    return finalmean,finalerrorbar

# This function calculates an error bar by bootstrapping from the raw counts.
def CountsToBootstrap(countsvector):
    thismean = []
    thiserrorbar = []
    for k in range(myruns):
        onemean, oneerrorbar = WilsonScore(countsvector[k])
        thismean.append(onemean)
        thiserrorbar.append(oneerrorbar)
    hitserrorbar = mean(thiserrorbar)/sqrt(myruns)
    standardmean,booterrorbar = BootstrapError(thismean)
    return standardmean,hitserrorbar,booterrorbar

# Calculate the central value and error bars for < psi | one Pauli | psi > and < psi | two Paulis | psi >.
newtimeindex = len(Xdata[0])
numbers = ["0","1","2","3","4","5","6","7","8","9","t","e"]
for i in range(ncircuits):
    for j in range(2*Nplaq):
        if j%2==0:
            label1 = CircuitOrder[i][0:1]
            label0 = CircuitOrder[i][2:3]
        else:
            label1 = CircuitOrder[i][2:3]
            label0 = CircuitOrder[i][0:1]
        if j<Nplaq:
            label = label0 + numbers[j%Nplaq]
        else:
            label = label1 + numbers[(j+1)%Nplaq] + label0 + numbers[j%Nplaq]
        runcounts = []
        for k in range(myruns):
            runcounts.append(allcounts[i][k][j])
        standardmean,hitserrorbar,randomizedcompilingerrorbar = CountsToBootstrap(runcounts)
        print(newtimeindex,"<psi|",label,"|psi> =",1-2*standardmean,2*hitserrorbar,2*randomizedcompilingerrorbar)
        if j<Nplaq:
            if label1=="X" and label0=="X":
                Xdata[j%Nplaq].append(1-2*standardmean)
            elif label1=="Z" and label0=="Z":
                Zdata[j%Nplaq].append(1-2*standardmean)
        else:
            exec(label1+label0+"data[j%Nplaq].append(1-2*standardmean)")

# Calculate the central value and error bars for < psi | three Paulis | psi >.
for j in range(Nplaq):
    i = 1 + j%2
    label = "Z"+ numbers[(Nplaq+j+1)%Nplaq] + "X" + numbers[j] + "Z" + numbers[(j-1)%Nplaq]
    runcounts = []
    for k in range(myruns):
        runcounts.append(allcounts[i][k][2*Nplaq+j])
    standardmean,hitserrorbar,randomizedcompilingerrorbar = CountsToBootstrap(runcounts)
    print(newtimeindex,"<psi|",label,"|psi> =",1-2*standardmean,2*hitserrorbar,2*randomizedcompilingerrorbar)
    ZXZdata[j].append(1-2*standardmean)

# Calculate the central value and error bars for <0|Z|0> and <0|ZZ|0> and <0|ZZZ|0>.
i = ncircuits
for j in range(3*Nplaq):
    if j<Nplaq:
        label = "Z" + numbers[j]
    elif j<2*Nplaq:
        label = "Z" + numbers[(j+1)%Nplaq] + "Z" + numbers[j%Nplaq]
    else:
        label = "Z" + numbers[(j+2)%Nplaq] + "Z" + numbers[(j+1)%Nplaq] + "Z" + numbers[j%Nplaq]
    runcounts = []
    for k in range(myruns):
        runcounts.append(allcounts[i][k][j])
    standardmean,hitserrorbar,randomizedcompilingerrorbar = CountsToBootstrap(runcounts)
    print(newtimeindex,"gate mitigation: <0|",label,"|0> =",1-2*standardmean,2*hitserrorbar,2*randomizedcompilingerrorbar)
    if j<Nplaq:
        Zgate[j%Nplaq].append(1-2*standardmean)
    elif j<2*Nplaq:
        ZZgate[j%Nplaq].append(1-2*standardmean)
    else:
        ZZZgate[j%Nplaq].append(1-2*standardmean)

# Provide expectation values in a form that can be pasted directly into the code.
print("Xdata = []")
for j in range(Nplaq):
    print("Xdata.append(",Xdata[j],")")
print("Zdata = []")
for j in range(Nplaq):
    print("Zdata.append(",Zdata[j],")")
print("XXdata = []")
for j in range(Nplaq):
    print("XXdata.append(",XXdata[j],")")
print("XZdata = []")
for j in range(Nplaq):
    print("XZdata.append(",XZdata[j],")")
print("YYdata = []")
for j in range(Nplaq):
    print("YYdata.append(",YYdata[j],")")
print("ZXdata = []")
for j in range(Nplaq):
    print("ZXdata.append(",ZXdata[j],")")
print("ZZdata = []")
for j in range(Nplaq):
    print("ZZdata.append(",ZZdata[j],")")
print("ZXZdata = []")
for j in range(Nplaq):
    print("ZXZdata.append(",ZXZdata[j],")")

# Provide expectation values for mitigation in a form that can be pasted directly into the code.
print("Zgate = []")
for j in range(Nplaq):
    print("Zgate.append(",Zgate[j],")")
print("ZZgate = []")
for j in range(Nplaq):
    print("ZZgate.append(",ZZgate[j],")")
print("ZZZgate = []")
for j in range(Nplaq):
    print("ZZZgate.append(",ZZZgate[j],")")

# Calculate the unmitigated energy expectation value, <Psi| H |Psi> / <Psi|Psi>   =   <psi| H |psi>.
hamiltonian = []
for k in range(myruns):
    hamilk = 9/8*Nplaq
    for j in range(Nplaq):
        onemean, oneerrorbar = WilsonScore(allcounts[0][k][j])
        hamilk -= 3/4*(1-2*onemean) # These are the <Z> terms.
    for j in range(Nplaq,2*Nplaq):
        onemean, oneerrorbar = WilsonScore(allcounts[0][k][j])
        hamilk -= 3/8*(1-2*onemean) # These are the <ZZ> terms.
    for j in range(0,Nplaq,2):
        onemean, oneerrorbar = WilsonScore(allcounts[1][k][j])
        hamilk -= 9/8*mycoupling*(1-2*onemean) # These are half of the <X> terms
    for j in range(1,Nplaq,2):
        onemean, oneerrorbar = WilsonScore(allcounts[2][k][j])
        hamilk -= 9/8*mycoupling*(1-2*onemean) # These are half of the <X> terms
    for j in range(Nplaq,2*Nplaq):
        onemean, oneerrorbar = WilsonScore(allcounts[1][k][j])
        hamilk -= 3/8*mycoupling*(1-2*onemean) # These are half of the <ZX> terms and half of the <XZ> terms.
    for j in range(Nplaq,2*Nplaq):
        onemean, oneerrorbar = WilsonScore(allcounts[2][k][j])
        hamilk -= 3/8*mycoupling*(1-2*onemean) # These are half of the <ZX> terms and half of the <XZ> terms.
    for j in range(2*Nplaq+1,3*Nplaq,2):
        onemean, oneerrorbar = WilsonScore(allcounts[1][k][j])
        hamilk -= 1/8*mycoupling*(1-2*onemean) # These are half of the <ZXZ> terms.
    for j in range(2*Nplaq,3*Nplaq,2):
        onemean, oneerrorbar = WilsonScore(allcounts[2][k][j])
        hamilk -= 1/8*mycoupling*(1-2*onemean) # These are half of the <ZXZ> terms.
    hamiltonian.append(hamilk)
energy,energyerrorbar = BootstrapError(hamiltonian)
print(newtimeindex,"<H_raw> =",energy,energyerrorbar)

# Calculate the mitigated energy expectation value, <Psi| H |Psi> / <Psi|Psi>   =   <psi| H |psi>.
Zmit = [0]*Nplaq
ZZmit = [0]*Nplaq
ZZZmit = [0]*Nplaq
for j in range(Nplaq):
    for k in range(myruns):
        onemean, oneerrorbar = WilsonScore(allcounts[ncircuits][k][j])
        Zmit[j] += (1-2*onemean)/myruns
        onemean, oneerrorbar = WilsonScore(allcounts[ncircuits][k][Nplaq+j])
        ZZmit[j] += (1-2*onemean)/myruns
        onemean, oneerrorbar = WilsonScore(allcounts[ncircuits][k][2*Nplaq+j])
        ZZZmit[j] += (1-2*onemean)/myruns
hamiltonian = []
for k in range(myruns):
    hamilk = 9/8*Nplaq
    for j in range(Nplaq):
        onemean, oneerrorbar = WilsonScore(allcounts[0][k][j])
        hamilk -= 3/4*(1-2*onemean)/Zmit[j] # These are the <Z> terms.
    for j in range(Nplaq,2*Nplaq):
        onemean, oneerrorbar = WilsonScore(allcounts[0][k][j])
        hamilk -= 3/8*(1-2*onemean)/ZZmit[j%Nplaq] # These are the <ZZ> terms.
    for j in range(0,Nplaq,2):
        onemean, oneerrorbar = WilsonScore(allcounts[1][k][j])
        hamilk -= 9/8*mycoupling*(1-2*onemean)/Zmit[j] # These are half of the <X> terms
    for j in range(1,Nplaq,2):
        onemean, oneerrorbar = WilsonScore(allcounts[2][k][j])
        hamilk -= 9/8*mycoupling*(1-2*onemean)/Zmit[j] # These are half of the <X> terms
    for j in range(Nplaq,2*Nplaq):
        onemean, oneerrorbar = WilsonScore(allcounts[1][k][j])
        hamilk -= 3/8*mycoupling*(1-2*onemean)/ZZmit[j%Nplaq] # These are half of the <ZX> terms and half of the <XZ> terms.
    for j in range(Nplaq,2*Nplaq):
        onemean, oneerrorbar = WilsonScore(allcounts[2][k][j])
        hamilk -= 3/8*mycoupling*(1-2*onemean)/ZZmit[j%Nplaq] # These are half of the <ZX> terms and half of the <XZ> terms.
    for j in range(2*Nplaq+1,3*Nplaq,2):
        onemean, oneerrorbar = WilsonScore(allcounts[1][k][j])
        hamilk -= 1/8*mycoupling*(1-2*onemean)/ZZmit[j%Nplaq] # These are half of the <ZXZ> terms.
    for j in range(2*Nplaq,3*Nplaq,2):
        onemean, oneerrorbar = WilsonScore(allcounts[2][k][j])
        hamilk -= 1/8*mycoupling*(1-2*onemean)/ZZZmit[j%Nplaq] # These are half of the <ZXZ> terms.
    hamiltonian.append(hamilk)
energy,energyerrorbar = BootstrapError(hamiltonian)
print(newtimeindex,"<H_mitigated> =",energy,energyerrorbar)
