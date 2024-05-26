# This code computes quantum imaginary time evolution (QITE) for Eq.1 of Kavaki and Lewis,arXiv:2401.14570 [hep-lat] (2024).
# It computes imaginary time evolution of SU(2) lattice gauge theory on a 2-plaquette lattice.
# The code is available at https://github.com/randylewis/SelfMitigatedQITE
# If you find this code useful, please cite the paper as well as the code.
# Note: The code uses IBM's open-source software called qiskit, available from https://qiskit.org/
#
# randy.lewis@yorku.ca

# Define user choices.
myprovider = 2           # Choose 0 for the public account or 2 for a research account.
myhardware = 2           # Choose the qubit hardware.
myqubits = [5,6]         # Choose the specific qubits to be used.
mycoupling = 1.0         # Choose the value for x = 2/g^4 from Eq.1 of Kavaki and Lewis,arXiv:2401.14570.
mytimestep = 0.1         # Choose the imaginary time step, Delta tau.
myshots = 10000          # Choose the number of shots for each run.
myconfidencelevel = 0.95 # Choose 0.95, for example, for error bars that represent a 95% confidence level.
myrcond = 0.5            # Choose the reciprocal condition number to be used in the Moore-Penrose inverse.
print("Input:",myprovider,myhardware,myqubits,mycoupling,mytimestep,myshots,myconfidencelevel,myrcond)

# Import tools from qiskit, and load my IBM Q account.
from qiskit import IBMQ, QuantumRegister, ClassicalRegister, QuantumCircuit, execute, result
from qiskit.tools.monitor import job_monitor
provider = IBMQ.load_account()
if (myprovider==2):
    provider = IBMQ.get_provider(hub='ibm-q-research-2')
else:
    provider = IBMQ.get_provider(hub='ibm-q')

# Import standard python tools.
import datetime
from numpy import pi, sqrt, zeros, dot
from numpy.linalg import pinv
from scipy.stats import norm

# Identify the hardware that will be used.
if myhardware==1:
    chosenhardware = "ibm_perth"
elif myhardware==2:
    chosenhardware = "ibm_lagos"
elif myhardware==3:
    chosenhardware = "ibm_oslo"
else:
    chosenhardware = "ibmq_qasm_simulator"
backend = provider.get_backend(chosenhardware)

# Use 9 lists of expectation values from the previous job: <X0>, <X1>, <Z0>, <Z1>, <X1 X0>, <X1 Z0>, <Z1 X0>, <Z1 Z0>, <Y1 Y0>.
X0data = []
X1data = []
Z0data = []
Z1data = []
X1X0data = []
X1Z0data = []
Z1X0data = []
Z1Z0data = []
Y1Y0data = []

# Define the vector b at every time step.
b = []
for tau in range(len(X0data)):
    bY0 = 9/4*X0data[tau] + 3/4*Z1X0data[tau] - 3*mycoupling*Z0data[tau] - mycoupling*Z1Z0data[tau] + mycoupling*X1X0data[tau]
    bX1Y0 = 9/4*X1X0data[tau] - 9/4*Y1Y0data[tau] - 3*mycoupling*X1Z0data[tau] + mycoupling*X0data[tau]
    bZ1Y0 = 9/4*Z1X0data[tau] + 3/4*X0data[tau] - 3*mycoupling*Z1Z0data[tau] - mycoupling*Z0data[tau] + 3*mycoupling*Y1Y0data[tau]
    bY1 = 9/4*X1data[tau] + 3/4*X1Z0data[tau] - 3*mycoupling*Z1data[tau] - mycoupling*Z1Z0data[tau] + mycoupling*X1X0data[tau]
    bY1X0 = 9/4*X1X0data[tau] - 9/4*Y1Y0data[tau] - 3*mycoupling*Z1X0data[tau] + mycoupling*X1data[tau]
    bY1Z0 = 9/4*X1Z0data[tau] + 3/4*X1data[tau] - 3*mycoupling*Z1Z0data[tau] - mycoupling*Z1data[tau] + 3*mycoupling*Y1Y0data[tau]
    b.append([bY0, bX1Y0, bZ1Y0, bY1, bY1X0, bY1Z0])

# Define the matrix S + S^T and calculate its inverse at every time step.
SSTinverse = []
for tau in range(len(X0data)):
    SplusST = zeros((6,6))
    SplusST[0][0] = 2
    SplusST[0][1] = 2*X1data[tau]
    SplusST[0][2] = 2*Z1data[tau]
    SplusST[0][3] = 2*Y1Y0data[tau]
    SplusST[1][0] = 2*X1data[tau]
    SplusST[1][1] = 2
    SplusST[1][4] = 2*Z1Z0data[tau]
    SplusST[1][5] = -2*Z1X0data[tau]
    SplusST[2][0] = 2*Z1data[tau]
    SplusST[2][2] = 2
    SplusST[2][4] = -2*X1Z0data[tau]
    SplusST[2][5] = 2*X1X0data[tau]
    SplusST[3][0] = 2*Y1Y0data[tau]
    SplusST[3][3] = 2
    SplusST[3][4] = 2*X0data[tau]
    SplusST[3][5] = 2*Z0data[tau]
    SplusST[4][1] = 2*Z1Z0data[tau]
    SplusST[4][2] = -2*X1Z0data[tau]
    SplusST[4][3] = 2*X0data[tau]
    SplusST[4][4] = 2
    SplusST[5][1] = -2*Z1X0data[tau]
    SplusST[5][2] = 2*X1X0data[tau]
    SplusST[5][3] = 2*Z0data[tau]
    SplusST[5][5] = 2
    temp = pinv(SplusST,rcond=myrcond,hermitian=True).tolist()
    SSTinverse.append(temp)

# Calculate the coefficients in the operator A of the time evolution factor exp(-i A DeltaTau) at every time step.
thetaY0 = []
thetaX1Y0 = []
thetaZ1Y0 = []
thetaY1 = []
thetaY1X0 = []
thetaY1Z0 = []
for tau in range(len(X0data)):
    a = -dot(SSTinverse[tau],b[tau])
    thetaY0.append(2*mytimestep*a[0])
    thetaX1Y0.append(2*mytimestep*a[1])
    thetaZ1Y0.append(2*mytimestep*a[2])
    thetaY1.append(2*mytimestep*a[3])
    thetaY1X0.append(2*mytimestep*a[4])
    thetaY1Z0.append(2*mytimestep*a[5])

# Create 5 circuits that first evolve through previous time steps and then give <X0>, <X1>, <Z0>, ... <Y1 Y0>.
circlist = []
for i in range(5):
    qreg = QuantumRegister(2)
    creg = ClassicalRegister(2)
    circ = QuantumCircuit(qreg,creg)
    if len(X0data) > 0:
        circ.cx(qreg[1],qreg[0])
        for tau in range(len(X0data)):
            circ.ry(thetaZ1Y0[tau],qreg[0])
            circ.cx(qreg[1],qreg[0])
            circ.ry(thetaY0[tau],qreg[0])
            circ.cx(qreg[0],qreg[1])
            circ.ry(thetaX1Y0[tau],qreg[0])
            circ.ry(thetaY1Z0[tau],qreg[1])
            circ.cx(qreg[0],qreg[1])
            circ.ry(thetaY1[tau],qreg[1])
            circ.cx(qreg[1],qreg[0])
            circ.ry(thetaY1X0[tau],qreg[1])
        circ.cx(qreg[1],qreg[0])
    if i==1:
        circ.ry(-pi/2,qreg[0])
        circ.ry(-pi/2,qreg[1])
    elif i==2:
        circ.rx(pi/2,qreg[0])
        circ.rx(pi/2,qreg[1])
    elif i==3:
        circ.ry(-pi/2,qreg[0])
    elif i==4:
        circ.ry(-pi/2,qreg[1])
    circ.measure(qreg,creg)
    circlist.append(circ)

# Run the circuits.
print("Queuing the circuits on the hardware at",datetime.datetime.now())
job = execute(circlist,backend,initial_layout=myqubits,shots=myshots)
print("The job identifier is",job.job_id())
job_monitor(job)
print("The hardware has returned results at",datetime.datetime.now())
circoutput = job.result().get_counts()
print("The output from all circuits is")
for i in range(5):
    print(circoutput[i])

# This function calculates a central value and error bar, given that "shots1" measurements gave 1 and "shotstotal-shots1" gave 0.
def WilsonScore(shots1):
    z = norm.ppf((1+myconfidencelevel)/2)
    denom = 1 + z**2/myshots
    frac1 = shots1/myshots
    centralvalue = ( frac1 + z**2/(2*myshots) )/denom
    sigma = z*sqrt( frac1*(1-frac1)/myshots + z**2/(4*myshots**2) )/denom
    return centralvalue,sigma

# Calculate the 9 expectation values <X0>, <X1>, <Z0>, <Z1>, <X1 X0>, <X1 Z0>, <Z1 X0>, <Z1 Z0>, <Y1 Y0>.
print("The expectation values are...")
label = []
label.append(["<Z0>    =","<Z1>    =","<Z1 Z0> ="])
label.append(["<X0>    =","<X1>    =","<X1 X0> ="])
label.append(["<Y0>    =","<Y1>    =","<Y1 Y0> ="])
label.append(["<X0>    =","<Z1>    =","<Z1 X0> ="])
label.append(["<Z0>    =","<X1>    =","<X1 Z0> ="])
expval = zeros((5,3))
expvalsig = zeros((5,3))
for i in range(5):
    shots00 = 0
    shots01 = 0
    shots10 = 0
    shots11 = 0
    for j,(key,value) in enumerate(circoutput[i].items()):
        if key=="00":
            shots00 = value
        elif key=="01":
            shots01 = value
        elif key=="10":
            shots10 = value
        elif key=="11":
            shots11 = value
        else:
            print("FATAL ERROR: What should be done with key =",key,"?")
    if shots00+shots01+shots10+shots11 != myshots:
        print("FATAL ERROR: The number of shots has changed to",shots00+shots01+shots10+shots11)
    for j in range(3):
        if j==0:
            shots1 = shots01 + shots11
        elif j==1:
            shots1 = shots10 + shots11
        elif j==2:
            shots1 = shots01 + shots10
        centralvalue,sigma = WilsonScore(shots1)
        expval[i][j] = 1 - 2*centralvalue
        expvalsig[i][j] = 2*sigma
        print(label[i][j],expval[i][j],"+/-",expvalsig[i][j])

# Calculate the energy expectation value, <Psi| H |Psi> / <Psi|Psi>   =   <psi| H |psi>.
hamiltonian = 21/8 - 9/8*(expval[0][0]+expval[0][1]) - 3/8*expval[0][2] - mycoupling/2*(3*expval[1][0]+3*expval[1][1]+expval[3][2]+expval[4][2])
hamiltonsig = sqrt( (9/8*expvalsig[0][0])**2 + (9/8*expvalsig[0][1])**2 + (3/8*expvalsig[0][2])**2 + (mycoupling/2*3*expvalsig[1][0])**2 \
                   + (mycoupling/2*3*expvalsig[1][1])**2 + (mycoupling/2*expvalsig[3][2])**2 + (mycoupling/2*expvalsig[4][2])**2 )
print("Time step =",len(X0data),"gives <Hamiltonian> =",hamiltonian,"+/-",hamiltonsig)

# Provide expectation values in a form that can be pasted directly into the code as input for the next job.
X0data.append( (expval[1][0]+expval[3][0])/2 )
X1data.append( (expval[1][1]+expval[4][1])/2 )
Z0data.append( (expval[0][0]+expval[4][0])/2 )
Z1data.append( (expval[0][1]+expval[3][1])/2 )
X1X0data.append(expval[1][2])
X1Z0data.append(expval[4][2])
Z1X0data.append(expval[3][2])
Z1Z0data.append(expval[0][2])
Y1Y0data.append(expval[2][2])
print("X0data =",X0data)
print("X1data =",X1data)
print("Z0data =",Z0data)
print("Z1data =",Z1data)
print("X1X0data =",X1X0data)
print("X1Z0data =",X1Z0data)
print("Z1X0data =",Z1X0data)
print("Z1Z0data =",Z1Z0data)
print("Y1Y0data =",Y1Y0data)
