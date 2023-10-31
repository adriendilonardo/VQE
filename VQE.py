import numpy as np
import cirq
import sympy
import pennylane as qml
import scipy as sp

# Hamiltonian Input Parsing Functions ---------------------------------------------------------------------

def is_pauli_operator(pennylane_operator):
    """
    Tests if the input is a Pauli matrix operator of the pennylane package, i.e. if the input is an instance of:
        - pennylane.PauliX
        - pennylane.PauliZ
        - pennylane.PauliY
        - pennylane.Identity

    Parameters
        pennylane_operator - obj
            Input object which will be tested
    
    Returns
        result - bool
            Boolean identifying whether the object is an instance of the pennylane Pauli operators or Identity operator
    """
    result = (
        isinstance(pennylane_operator, qml.PauliX)
        or isinstance(pennylane_operator, qml.PauliY)
        or isinstance(pennylane_operator, qml.PauliZ)
        or isinstance(pennylane_operator, qml.Identity)
    )
    return result

def pennylane_ham_to_str_ham(pHam):
    """
    Convert a pennylane Hamiltonian object to a list of (coefficient, operator description) tuple pairs.
    The format of the tuples is:
        - first element - float
            Gives the coefficient for the operator
        - second element - str
            Gives a description of the operator as a tensor product of Pauli operators and Identity operators.
            The str will explicitly describe the operator (including identity operator) acting on all qubits, in order.
            E.g. for a set of 4 qubits, an operator which has Z operators that act on the second and third qubit only 
            will be described as: "IZZI"

    Note: This function will raise an error for Hamiltonians constructed from observables other than the Pauli operators
    and Identity operator.

    Parameters
        pHam - pennylane.ops.qubit.hamiltonian.Hamiltonian
            pennylane hamiltonian object

    Returns
        Ham - list[tuple(float, str)]
    """
    coeffs = [float(coeff) for coeff in pHam.coeffs] # Term coefficients
    num_qubits = len(pHam.wires) # number of qubits

    # Instantiating list of operator descriptions
    operator_strings = []

    # Translating pennylane operators to str objects
    for tensor_product in pHam.ops: #for each operator in the pennylane Hamiltonian
        temp_str = ["I"]*num_qubits   # Instantiate a list of num_qubit "I"s
        if not isinstance(tensor_product, qml.Identity):    # If not identity on all qubits

            if not isinstance(tensor_product, qml.operation.Tensor): # If tensor product is a single operator
                if not is_pauli_operator(tensor_product):   # Input handling for non-Pauli operators
                    raise ValueError("Hamiltonian has a non-Pauli term")
                qubit_index = tensor_product.wires[0]       # Index of qubit being acted on
                temp_str[qubit_index] = tensor_product.label() 

            elif isinstance(tensor_product, qml.operation.Tensor):  # If tensor product has multiple operators
                for operator in tensor_product.obs: # loop through each operator
                    if not is_pauli_operator(operator):     # Input handling for non-Pauli operators
                        raise ValueError("Hamiltonian has a non-Pauli term")
                    qubit_index = operator.wires[0]         # Index of qubit being acted on
                    temp_str[qubit_index] = operator.label()

        operator_strings.append(''.join(temp_str))

    return list(zip(coeffs, operator_strings))

def check_ham_input(Ham):
    """
    Parse an input Hamiltonian to validate the following:
        - Data types and structure 
            - Hamiltonian should be of the form list[tuple(float, str)]
        - Input values
            - The tuple pairs in the Hamiltonian describe its operators and their respective coefficients.
            The strings will explicitly describe the full tensor product (including identity operator) acting on all qubits, in order.
            E.g. for a set of 4 qubits, an operator which has Z operators that act on the second and third qubit only 
            will be described as: "IZZI"
            - Operators that are not tensor products of Pauli and Identity operators will not be accepted
    
    The function does not return any information, rather an error will be raised if the input Hamiltonian is not acceptable.

    Parameters
        Ham - obj
            Object to be validated, ideally has the form list[tuple(float, str)]
        
    Returns
        None
    """
    # Catching inputs which are not lists of lists/tuples
    try:
        zip(*Ham)
    except TypeError:
        raise TypeError("Hamiltonian must have the form: list[tuple(int|float, str)]")

    # Tuples have length > 2
    for array in Ham:
        if len(array) != 2:
            raise ValueError("List should consist of tuple/list pairs, i.e. array-like objects of length 2")

    operator_string_length = None
    for coeff, operator_str in Ham: 
        # type errors - coefficient, operator str pair don't have the types float, str respectively
        if (not (isinstance(coeff, float) or isinstance(coeff, int))) or (not isinstance(operator_str, str)):
            raise TypeError("Hamiltonian must have the form: list[tuple(int|float, str)]")
        
        # value error - length of each operator str is not the same (identity operators must be explicitly included)
        # determining the length of the first operator string
        if not operator_string_length:
            operator_string_length = len(operator_str)

        # if the length of this operator string is not the same as the first
        if operator_string_length != len(operator_str):
            raise ValueError("All Pauli strings must have the same length (identity operators must be explicitly included), see example from docstring")
        
        # non-pauli operators
        removeIXZY = operator_str.lower().replace("i", '').replace("x", '').replace("y", '').replace("z", '')
        if len(removeIXZY) > 0:
            raise ValueError("All Pauli strings must consist exclusively of the characters I, X, Y, and Z")

# Functions which produce parametrised state preparation circuits --------------------------------------------------------

def YZCircuit(num_qubits, depth):
    """
    Produces a parametrised RY-RZ state preparation circuit with 2*n*depth parameters

    The circuit uses a number of repetitions of the following sequence:
        - Parametrised RY gates on every qubit
        - Parametrised Rz gates on every qubit
        - Entangling CNOT gates between every neighbouring pair of qubits

    Note that each repitition introduces additional parameters for the parametrised single qubit gates
    
    Parameters
        num_qubits - int > 0
            The number of qubits in the state preparation sequence
        depth - int > 0
            Specifies the number of repetitions of the parametrised sequence

    Return 
        circuit - cirq.Circuit
            The cirq.Circuit which prepares the state
    """
    num_parameters = 2*num_qubits*depth   # Number of Parameters
    theta = sympy.symbols(f"theta0:{num_parameters}")    # array of sympy symbols, theta0 to theta<num_parameters-1>

    # Instantiate circuit and qubits
    circuit = cirq.Circuit()
    q = cirq.LineQubit.range(num_qubits)

    # Constructing all of the parametrised gates
    # There are depth number of cycles of the gate sequence described above
    # Each cycle has num_qubit parameterised RY gates (one for each qubit)
    # + num_qubit parameterised RZ gates (one for each qubit)
    for cycle in range(depth):
        index_base = 2*num_qubits*cycle # starting index for parameters in this cycle

        # Parametrised RY gates on all qubits, with indices (i.e. theta<index>) starting from index_base
        RY = [cirq.ry(theta[index_base+j]).on(q[j]) for j in range(num_qubits)]
        circuit.append(RY)    

        # Parametrised RZ gates on all qubits, with parameter indices (i.e. theta<index>) starting from index_base + num_qubits
        RZ = [cirq.rz(theta[index_base+num_qubits+j]).on(q[j]) for j in range(num_qubits)]
        circuit.append(RZ)   
        
        # CNOT gates between every neighbouring pair
        circuit.append([cirq.CNOT(q[j], q[j+1]) for j in range(num_qubits-1)])

    return circuit

# Functions for evaluating expectation values ----------------------------------------------------------------

def operator_str_to_hadamard_test(operator_str):
    """
    Produces a circuit performing a Hadamard test for a given input operator string

    Parameters
        operator_str - str
            str describing a tensor product of operators. 
            The string must explicitly describe the full tensor product (including identity operator) acting on all qubits, in order.
            E.g. for a set of 4 qubits, an operator which has Z operators that act on the second and third qubit only 
            will be described as: "IZZI"

            Note: the operator string in this case need not be exclusively be Pauli and Identity operators, but any unitary operators
            which are specified as a single uppercase character in cirq, e.g. I, X, Y, Z, S, T, H
    
    Returns
        circuit - cirq.Circuit
            The cirq.Circuit performing the Hadamard test
    """

    #Instantiate Circuit and qubits
    circuit = cirq.Circuit()
    num_qubits = len(operator_str)
    q = cirq.LineQubit.range(num_qubits)
    a = cirq.NamedQubit("ancilla")

    #Hadamard Test
    circuit.append(cirq.H(a))  # Hadamard on ancilla
    for i, op in enumerate(operator_str):
        if op.upper() != "I":
            U = eval('cirq.'+op.upper())    # Pauli operator specified by character
            cU = cirq.ControlledGate(U).on(a, q[i]) # Controlled Pauli on ith qubit
            circuit.append(cU)
    circuit.append(cirq.H(a))   # Hadamard on ancilla

    circuit.append(cirq.measure(a)) # Measure ancilla

    return circuit

def operator_str_to_cirq_operator(operator_str, qubits):
    """
    Turn an operator string into cirq operators acting on qubits (for the purpose of evaluating expectation values)

    Parameters
        operator_str - str
            str describing a tensor product of pauli operators. 
            The string must explicitly describe the full tensor product (including identity operator) acting on all qubits, in order,
            and must consist exclusively of Pauli and Identity operators.
            E.g. for a set of 4 qubits, an operator which has Z operators that act on the second and third qubit only 
            will be described as: "IZZI"
        qubits - 1D Array-like[cirq.NamedQubit | cirq.LineQubit | cirq.GridQubit]
            array-like container of qubits

    Returns
        pauli_str - cirq.ops.pauli_string.PauliString
            Pauli string object acting on the qubit objects provided

    """
    # Testing if the number of qubits matches the number of operators in the operator string
    if len(qubits) != len(operator_str):
        raise ValueError("The number of operators in the operator_str should be equal to the number of qubits")
    
    # Testing that the operator string consists only of ("I", "X", "Y", "Z")
    removeIXZY = operator_str.lower().replace("i", '').replace("x", '').replace("y", '').replace("z", '')
    if len(removeIXZY) > 0:
        raise ValueError("All Pauli strings must consist exclusively of the characters I, X, Y, and Z")

    operators = []
    for i, op in enumerate(operator_str):
        cirq_operator = eval('cirq.'+op.upper())
        operators.append(cirq_operator.on(qubits[i]))
    pauli_str = operators[0]
    for op in operators[1:]:
        pauli_str *= op
    return pauli_str

# Output Pretty print function ---------------------------------------------------------------

def pretty_print_results(energy, state):
    """
    Print the ground state energy and state vector in a clear format

    Parameters
        energy - float
            ground state energy
        state - 1D array-like[complex or similar]
            complex vector describing the ground state in the qubit computational basis (i.e. |00..0>, |00..1>, ..., |11..1>)

    Returns
        None
    """
    dim = len(state)
    num_qubits =int(np.log2(dim))
    if abs(int(np.log2(dim)) - np.log2(dim)) > 1e-9:
        raise ValueError("Vector must be of length 2**n where n is some integer")
    print(f"Ground State Energy: {energy:.6e}")
    print(f"Ground State:")
    for i, a in enumerate(state): 
        print(f"{a.real:+.2e} {a.imag:+.2e}j |{i:0>{num_qubits}b}>   |  {np.abs(a):.2e} exp(i*pi*{np.angle(a)/np.pi:.3f}) |{i:0>{num_qubits}b}>")

# High-level VQE functions for users ------------------------------------------------------------------------------

def VQE_HT(Ham, depth=2, initial_params=None, precision=1e-3, optimisation_method="Nelder-Mead", pretty_print=True):
    """
    Run a VQE using a circuit that is executable on a quantum device (although this version currently only simulates the circuit).
    Expectation value estimates are obtained using repeated probabilistic Hadamard tests (this is very slow).
    It is recommended to use VQE_Sim for the purposes of simulating the VQE algorithm, since it relies on direct expectation value
    calculations on the state vector, which is much more efficient than repeated State-Preparation + Hadamard Tests.

    Parameters
        Ham - list[tuple(float, str)] | pennylane.ops.qubit.hamiltonian.Hamiltonian
            Description of the Hamiltonian consisting of tuple pairs of operators and their respective coefficients. The operators must 
            exclusively be tensor products of Pauli and Identity operators and described as str using the following format:

            The string must explicitly describe the full tensor product (including identity operator) acting on all qubits, in order.
            E.g. for a set of 4 qubits, an operator which has Z operators that act on the second and third qubit only 
            will be described as: "IZZI"

        depth - int
            depth of the state preparation circuit (also affects the number of independent parameters)

        initial params - array-like[float | int] of length 2*num_qubits*depth
            initial values for the parameters in the parametrised state preparation circuit.
            The parameters are indexed as follows:
                - j*2*num_qubits to j*2*num_qubits + num_qubits - 1     = RY gates on all qubits starting from qubit 0 in the (j+1)th cycle of gates (j starts at 0)
                - j*2*num_qubits + num_qubits to (j+1)*2*num_qubits - 1 = RZ gates on all qubits starting from qubit 0 in the (j+1)th cycle of gates (j starts at 0)
            
            (See YZCircuit docstring for more detail)

        precision - float (between 0 and 1)
            precision to which the expectation value will be estimated

        optimisation_method - str or callable
            Optimisation method employed by scipy.optimize.minimize. Available options can be found on the following page under "method":
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        pretty_print - boolean
            Boolean indicating whether the ground state and ground energy should be printed out (using the pretty_print_results function)
    
    Returns
        energy - float
            Ground state energy (in the units used to describe the Hamiltonian)
        state - np.ndarray(dtype=complex)
            State vector of ground state
    """

    # If Ham is pennylane.ops.qubit.hamiltonian.Hamiltonian, convert into list[tuple(float, str)]
    if isinstance(Ham, qml.ops.qubit.hamiltonian.Hamiltonian):
        Ham = pennylane_ham_to_str_ham(Ham)
    
    # Validate input is correct
    check_ham_input(Ham)

    num_qubits = len(Ham[0][1])
    num_parameters = 2*num_qubits*depth
    initial_params = 2*np.pi*np.random.rand(num_parameters) if initial_params is None else initial_params

    # Objects required for the cost function
    coeffs, operators = zip(*Ham)   # Extract coefficients and operators
    hadamard_tests = [operator_str_to_hadamard_test(op) for op in operators] # Produce Hadamard tests for the expectation value of each operator
    state_prep_circuit = YZCircuit(num_qubits, depth)   # Parametrised state preparation circuit
    sim = cirq.Simulator()

    # Build the cost function
    def costFunc(theta):
        # Populate the state preparation circuit with parameter values
        fixed_state_prep_circuit = cirq.resolve_parameters(state_prep_circuit, {f"theta{i}":val for i, val in enumerate(theta)})
        
        # Determine expectation values of each operator, weighted by the associated coefficient
        expectation_values = []
        for coeff, h_test in zip(coeffs, hadamard_tests):
            result = sim.run(fixed_state_prep_circuit + h_test, repetitions=int(1/precision))
            histogram = result.histogram(key='ancilla')
            expectation_value = (histogram[1]-histogram[0])/(histogram[1]+histogram[0]) # Re<state|U|state> = counts(measure 1) - counts(measure 0) / total counts
            expectation_values.append(coeff*expectation_value)
        
        return sum(expectation_values)
    
    res = sp.optimize.minimize(
        costFunc, 
        np.random.rand(num_parameters), # Randomised initial parameter values
        method=optimisation_method, 
        bounds=[(0, 2*np.pi)]*num_parameters  # Parameters bounded between 0 and 2*pi
    )
    final_state_prep_circuit = cirq.resolve_parameters(
        state_prep_circuit, 
        {f"theta{i}":val for i, val in enumerate(res.x)}
    )
    state = cirq.Simulator().simulate(final_state_prep_circuit).state_vector()
    energy = res.fun

    if pretty_print:
        pretty_print_results(energy, state)

    return energy, state

def VQE_Sim(Ham, depth=2, initial_params=None, optimisation_method="Nelder-Mead", pretty_print=True):
    """
    Run a VQE using expectation value calculations on a state vector rather than measurements (algorithm cannot be executed on a quantum device)

    Simulate a VQE which is not executable on a quantum device. This is because expectation values are calculated directly from the state vector
    after state preparation, which cannot be obtained directly through measurement. This is much more efficient than repeated projective measurements,
    however, which takes significantly longer.

    Note, the current version of VQE_HT, which is designed such that it uses a circuit that is executable on a quantum device, is still only simulating
    a circuit. Future versions may allow execution on a physical device.

    Parameters
        Ham - list[tuple(float, str)] | pennylane.ops.qubit.hamiltonian.Hamiltonian
            Description of the Hamiltonian consisting of tuple pairs of operators and their respective coefficients. The operators must 
            exclusively be tensor products of Pauli and Identity operators and described as str using the following format:

            The string must explicitly describe the full tensor product (including identity operator) acting on all qubits, in order.
            E.g. for a set of 4 qubits, an operator which has Z operators that act on the second and third qubit only 
            will be described as: "IZZI"

        initial params - array-like[float | int] of length 2*num_qubits*depth
            initial values for the parameters in the parametrised state preparation circuit.
            The parameters are indexed as follows:
                - j*2*num_qubits to j*2*num_qubits + num_qubits - 1     = RY gates on all qubits starting from qubit 0 in the (j+1)th cycle of gates (j starts at 0)
                - j*2*num_qubits + num_qubits to (j+1)*2*num_qubits - 1 = RZ gates on all qubits starting from qubit 0 in the (j+1)th cycle of gates (j starts at 0)
                
            (See YZCircuit docstring for more detail)
        
        depth - int
            depth of the state preparation circuit (also affects the number of independent parameters)

        optimisation_method - str or callable
            Optimisation method employed by scipy.optimize.minimize. Available options can be found on the following page under "method":
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        pretty_print - boolean
            Boolean indicating whether the ground state and ground energy should be printed out (using the pretty_print_results function)
    
    Returns
        energy - float
            Ground state energy (in the units used to describe the Hamiltonian)
        state - np.ndarray(dtype=complex)
            State vector of ground state

    """

    # If Ham is pennylane.ops.qubit.hamiltonian.Hamiltonian, convert into list[tuple(float, str)]
    if isinstance(Ham, qml.ops.qubit.hamiltonian.Hamiltonian):
        Ham = pennylane_ham_to_str_ham(Ham)

    # Validate input is correct
    check_ham_input(Ham)

    num_qubits = len(Ham[0][1])
    num_parameters = 2*num_qubits*depth
    initial_params = 2*np.pi*np.random.rand(num_parameters) if initial_params is None else initial_params

    # Objects required for the cost function
    coeffs, operators = zip(*Ham)   # Extract the coefficients
    state_prep_circuit = YZCircuit(num_qubits, depth)
    
    qubits = list(state_prep_circuit.all_qubits())
    cirq_operators = [operator_str_to_cirq_operator(op, qubits) for op in operators]
    
    sim = cirq.Simulator()

    # Building the cost function
    def costFunc(theta):
        fixed_state_prep_circuit = cirq.resolve_parameters(
            state_prep_circuit, 
            {f"theta{i}":val for i, val in enumerate(theta)}
        )
        expList = sim.simulate_expectation_values(fixed_state_prep_circuit, cirq_operators)
        return sum([coeff*expectation_value.real for coeff, expectation_value in zip(coeffs, expList)])
    
    res = sp.optimize.minimize(
        costFunc, 
        initial_params, # Randomised initial parameter values
        method=optimisation_method, 
        bounds=[(0, 2*np.pi)]*num_parameters # Parameters bounded between 0 and 2*pi
    )
    final_state_prep_circuit = cirq.resolve_parameters(
        state_prep_circuit,
        {f"theta{i}":val for i, val in enumerate(res.x)}
    )
    state = sim.simulate(final_state_prep_circuit).state_vector()
    energy = res.fun

    if pretty_print:
        pretty_print_results(energy, state)

    return energy, state