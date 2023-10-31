import numpy as np
import cirq
import pennylane as qml
from VQE import *
from unittest.mock import patch

def test_is_pauli_operator():
    """
    Test whether is_pauli_operator produces the correct outcome for a range of input objects
    """

    #Objects which should return False
    False_objs = [
        'test', 
        [1, 2, 3], 
        qml.operation.Tensor(qml.PauliX(1), qml.PauliY(2)), 
        cirq.X
    ]

    #Objects which should return True
    True_objs = [
        qml.Identity(1), 
        qml.PauliX(500), 
        qml.PauliY(2), 
        qml.PauliZ(412)
    ]

    for obj in False_objs:
        assert not is_pauli_operator(obj)

    for obj in True_objs:
        assert is_pauli_operator(obj)

def test_success_pennylane_ham_to_str_ham():
    """
    Test whether pennylane_ham_to_str_ham produces the correct output given some test inputs
    """

    # Test Hamiltonian condtructed in pennylane
    test_Ham = qml.Hamiltonian(
        coeffs=[1, 2, 3, 4, 5, 6, 7], 
        observables=(
            qml.Identity(0), 
            qml.PauliX(0),
            qml.PauliY(1),
            qml.PauliZ(2),
            qml.operation.Tensor(qml.PauliZ(0), qml.PauliZ(1)),
            qml.operation.Tensor(qml.PauliZ(1), qml.PauliZ(2)),
            qml.operation.Tensor(qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)),
        )
    )

    output = pennylane_ham_to_str_ham(test_Ham)

    expected_output = [
        (1, "III"),
        (2, "XII"),
        (3, "IYI"),
        (4, "IIZ"),
        (5, "ZZI"),
        (6, "IZZ"),
        (7, "XXX")
    ]

    for out, exp_out in zip(output, expected_output):
        assert abs(out[0]-exp_out[0]) < 1e-8    # test that the values of the coefficients are correct
        assert out[1] == exp_out[1]             # test that the operator strings are correct

def test_error_pennylane_ham_to_str_ham():
    """
    Test if pennylane_ham_to_str_ham raises the correct errors for invalid inputs.

    This includes:
        - inputs which have non-Pauli observables, as part of both:
            - a tensor product
            - a standalone observable
    """

    # Hamiltonian with Standalone non-Pauli operator
    test_Ham_standalone = qml.Hamiltonian(
        coeffs=[1, 2, 3], 
        observables=(
            qml.Identity(0), 
            qml.PauliX(0),
            qml.Hadamard(1),    # Hadamard should raise error
        )
    ) 

    # Hamiltonian with non-Pauli operator in a tensor product
    test_Ham_tensor = qml.Hamiltonian(
        coeffs=[1, 2, 3], 
        observables=(
            qml.operation.Tensor(qml.PauliZ(0), qml.PauliZ(1)),
            qml.operation.Tensor(qml.PauliX(0), qml.NumberOperator(1), qml.PauliX(2)), # Number operator should raise error
            qml.operation.Tensor(qml.PauliZ(1), qml.PauliZ(2)),   
        )
    )

    # If the ValueError is raised, continue
    # If a different error or no error is raised, raise an error
    for ham in [test_Ham_standalone, test_Ham_tensor]:
        try:
            pennylane_ham_to_str_ham(ham)
        except ValueError as e:
            if str(e) != "Hamiltonian has a non-Pauli term":    # Checking if the right error was raised
                raise ValueError("Incorrect error raised")
        else:
            raise ValueError("Error should have been raised")

def test_success_check_ham_input():
    """
    Test if check_ham_input correctly identifies a valid hamiltonian input without raising an error
    """

    test_Ham = [(1, "XXY"), (2.5, "ZII"), (1/4, "IIX")] # Valid Hamiltonian Input
    check_ham_input(test_Ham)

def test_error_check_ham_input():
    """
    Test if check_ham_input correctly catches all of the invalid inputs it is designed to catch:
        - Inputs which are not array-likes of array-likes (e.g. lists of tuples)
        - Inner array-likes have len != 2 (e.g. tuples have length 1 or 3)
        - Inner array-likes don't contain float/int then str
        - All strings (describing tensor products) are not all the same length
        - Strings do not consist exclusively of Pauli and Identity operators
    """

    # A list of invalid Hamiltonian descriptions, matching the order of the errors listed in the docstring
    test_Hams = [
        [1, "XXIX"],
        [(1, "XXI"), (1, 2, 4)],
        [(1, "XXI"), (1.5, qml.PauliX(1))],
        [(1, "XXI"), (1.5, "XXZ"), (1.6, "X"), (5, "ZZI")],
        [(1, "XXI"), (1.5, "XXZ"), (1.6, "XHZ"), (5, "ZZI")]
    ]

    # errors types associated with each invalid Hamiltonian
    errors = [
        TypeError,
        ValueError,
        TypeError,
        ValueError,
        ValueError
    ]

    # error strings associated with each invalid Hamiltonian
    error_strings = [
        "Hamiltonian must have the form: list[tuple(int|float, str)]",
        "List should consist of tuple/list pairs, i.e. array-like objects of length 2",
        "Hamiltonian must have the form: list[tuple(int|float, str)]",
        "All Pauli strings must have the same length (identity operators must be explicitly included), see example from docstring",
        "All Pauli strings must consist exclusively of the characters I, X, Y, and Z"
    ]

    # for each invalid Hamiltonian, and its associated error type + error string
    for ham, error, error_string in zip(test_Hams, errors, error_strings):
        # If the correct error is raised, continue
        # If a different error or no error is raised, raise an error
        try:
            check_ham_input(ham)
        except error as e:
            if str(e) != error_string:
                raise ValueError("Incorrect error raised")
        else:
            raise ValueError("Error should have been raised")

def test_YZCircuit_output():
    """
    Test that YZCircuit produces the correct parametrized circuit for some given arguments
    """
    circuit = YZCircuit(num_qubits=5, depth=2)

    # expected text diagram output for a state preparation circuit with the above parameters (num_qubits=5, depth=2)
    # This was output from a circuit which was constructed carefully and validated
    diagram = (
      '0: ───Ry(theta0)───Rz(theta5)───@───Ry(theta10)───Rz(theta15)─────────────────@───────────────────────────────────\n'
    + '                                │                                             │\n'
    + '1: ───Ry(theta1)───Rz(theta6)───X───@─────────────Ry(theta11)───Rz(theta16)───X─────────────@─────────────────────\n'
    + '                                    │                                                       │\n'
    + '2: ───Ry(theta2)───Rz(theta7)───────X─────────────@─────────────Ry(theta12)───Rz(theta17)───X─────────────@───────\n'
    + '                                                  │                                                       │\n'
    + '3: ───Ry(theta3)───Rz(theta8)─────────────────────X─────────────@─────────────Ry(theta13)───Rz(theta18)───X───@───\n'
    + '                                                                │                                             │\n'
    + '4: ───Ry(theta4)───Rz(theta9)───────────────────────────────────X─────────────Ry(theta14)───Rz(theta19)───────X───'
    )

    assert circuit.to_text_diagram() == diagram

    circuit = YZCircuit(num_qubits=4, depth=3)

    # expected text diagram output for a state preparation circuit with the above parameters (num_qubits=4, depth=3)
    # This was output from a circuit which was constructed carefully and validated
    diagram = (
      '0: ───Ry(theta0)───Rz(theta4)───@───Ry(theta8)───Rz(theta12)─────────────────@─────────────Ry(theta16)───Rz(theta20)─────────────────@─────────────────────\n'
    + '                                │                                            │                                                       │\n'
    + '1: ───Ry(theta1)───Rz(theta5)───X───@────────────Ry(theta9)────Rz(theta13)───X─────────────@─────────────Ry(theta17)───Rz(theta21)───X─────────────@───────\n'
    + '                                    │                                                      │                                                       │\n'
    + '2: ───Ry(theta2)───Rz(theta6)───────X────────────@─────────────Ry(theta10)───Rz(theta14)───X─────────────@─────────────Ry(theta18)───Rz(theta22)───X───@───\n'
    + '                                                 │                                                       │                                             │\n'
    + '3: ───Ry(theta3)───Rz(theta7)────────────────────X─────────────Ry(theta11)───Rz(theta15)─────────────────X─────────────Ry(theta19)───Rz(theta23)───────X───'
    )

    assert circuit.to_text_diagram() == diagram

def test_YZCircuit_is_parametrized():
    """
    Verify that the circuit created by YZCircuit has:
        - 2*num_qubits*depth parameters
        - with labels theta0:theta<2*num_qubits*depth - 1>
    """
    circuit = YZCircuit(num_qubits=4, depth=3)
    parameter_names = circuit._parameter_names_()
    expected_parameter_names = {f'theta{i}' for i in range(2*4*3)}

    assert circuit._is_parameterized_() # Check if the circuit is parametrized
    assert parameter_names == expected_parameter_names  # Check that the parameter names are correct

def test_YZCircuit_with_assigned_parameters():
    """
    Test that the overall unitary of a YZCircuit with assigned parameter values is correct
    """
    # Defining gates which are used in the state preparation circuit
    RY = lambda angle: np.array(
        [[np.cos(angle/2), -np.sin(angle/2)],
         [np.sin(angle/2), np.cos(angle/2)]]
    )
    RZ = lambda angle: np.array(
        [[np.exp(-1j*angle/2), 0],
         [0, np.exp(1j*angle/2)]]
    )
    I = np.array(
        [[1, 0], 
         [0, 1]]
    )
    CNOT = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1], 
         [0, 0, 1, 0]]
    )

    num_qubits = 2
    depth = 2
    theta = [i for i in range(2*2*2)]
    
    circuit = YZCircuit(num_qubits, depth)._resolve_parameters_({f"theta{i}":val for i, val in enumerate(theta)}, recursive=False)
    unitary = circuit.unitary()

    # Constructing the expected unitary
    expected_unitary = (
        CNOT
        @ np.kron(RZ(theta[6]), RZ(theta[7]))
        @ np.kron(RY(theta[4]), RY(theta[5]))
        @ CNOT
        @ np.kron(RZ(theta[2]), RZ(theta[3]))
        @ np.kron(RY(theta[0]), RY(theta[1]))
    )

    assert np.allclose(unitary, expected_unitary)

def test_operator_str_to_hadamard_test():
    """
    Test Pauli_str_to_hadamard_test produces the correct Hadamard test circuits for different input Pauli strings
    """

    test_str = "IZII"
    circuit = operator_str_to_hadamard_test(test_str)

    # expected text diagram output for a Hadamard test of the above tensor product
    # This was output from a circuit which was constructed carefully and validated
    diagram = (
          '1: ─────────────Z───────────\n'
        + '                │\n'
        + 'ancilla: ───H───@───H───M───'
    )

    assert circuit.to_text_diagram() == diagram

    test_str = "XYZ"
    circuit = operator_str_to_hadamard_test(test_str)

    # expected text diagram output for a Hadamard test of the above tensor product
    # This was output from a circuit which was constructed carefully and validated
    diagram = (
          '0: ─────────────X───────────────────\n'
        + '                │\n'
        + '1: ─────────────┼───Y───────────────\n'
        + '                │   │\n'
        + '2: ─────────────┼───┼───Z───────────\n'
        + '                │   │   │\n'
        + 'ancilla: ───H───@───@───@───H───M───'
    )

    assert circuit.to_text_diagram() == diagram

    test_str = "IIIII"
    circuit = operator_str_to_hadamard_test(test_str)

    # expected text diagram output for a Hadamard test of the above tensor product
    # This was output from a circuit which was constructed carefully and validated
    diagram = 'ancilla: ───H───H───M───'

    assert circuit.to_text_diagram() == diagram

def test_success_operator_str_to_cirq_operator():
    """
    Test that operator_str_to_cirq_operator successfully produces the correct Pauli string output for some given inputs
    """
    qubits = cirq.LineQubit.range(5)
    test_inputs = [
        ("IIXII", qubits),
        ("XYIZ", qubits[:4]),
        ("II", qubits[:2]),
    ]

    expected_outputs = [
        cirq.X.on(qubits[2]),
        cirq.X.on(qubits[0])*cirq.Y.on(qubits[1])*cirq.Z.on(qubits[3]),
        cirq.I.on(qubits[1])
    ]

    for inp, out in zip(test_inputs, expected_outputs):
        assert operator_str_to_cirq_operator(*inp) == out

def test_error_operator_str_to_cirq_operator():
    """
    Test that operator_str_to_cirq_operator correctly catches invalid inputs:
        - The number of operators in the operator_str is not equal to the number of qubits
        - All Pauli strings do not consist exclusively of the characters I, X, Y, and Z
    """

    test_inputs = [
        ("XXIX", cirq.LineQubit.range(5)),  # number of operators < number of qubits
        ("IZ", [cirq.NamedQubit('test')]),  # number of operators > number of qubits
        ("HIZ", cirq.LineQubit.range(3))    # some operators are not I, X, Y, or Z
    ]

    # associated error strings
    error_strings = [
        "The number of operators in the operator_str should be equal to the number of qubits",
        "The number of operators in the operator_str should be equal to the number of qubits",
        "All Pauli strings must consist exclusively of the characters I, X, Y, and Z"
    ]

    for inp, error_string in zip(test_inputs, error_strings):
        # If the correct error is raised, continue
        # If a different error or no error is raised, raise an error
        try:
            operator_str_to_cirq_operator(*inp)
        except ValueError as e:
            if str(e) != error_string:
                raise ValueError("Incorrect error raised")
        else:
            raise ValueError("Error should have been raised")

@patch('builtins.print')
def test_pretty_print_results(mock_print):
    """
    Verify that the output of pretty_print_results is correct for two given inputs
    """

    energy = 1.23456789
    state = [
        1+2j,
        3+4j,
        5+6j,
        7+8j
    ]
    pretty_print_results(energy, state)
    expected_output = [
        "Ground State Energy: 1.234568e+00",
        "Ground State:",
        "+1.00e+00 +2.00e+00j |00>   |  2.24e+00 exp(i*pi*0.352) |00>",
        "+3.00e+00 +4.00e+00j |01>   |  5.00e+00 exp(i*pi*0.295) |01>",
        "+5.00e+00 +6.00e+00j |10>   |  7.81e+00 exp(i*pi*0.279) |10>",
        "+7.00e+00 +8.00e+00j |11>   |  1.06e+01 exp(i*pi*0.271) |11>",
    ]
    for out, expected_out in zip(mock_print.call_args_list, expected_output):
        assert out.args[0] == expected_out

    mock_print.call_args_list = []

    energy = 1.23456789
    state = [
        1*np.exp(1j*np.pi*0.25),
        2*np.exp(1j*np.pi*0.5),
        3*np.exp(1j*np.pi*0.75),
        4*np.exp(1j*np.pi*1),
        5*np.exp(1j*np.pi*1.25),
        6*np.exp(1j*np.pi*1.5),
        7*np.exp(1j*np.pi*2),
        8*np.exp(1j*np.pi*2.5),
    ]
    pretty_print_results(energy, state)
    expected_output = [
        "Ground State Energy: 1.234568e+00",
        "Ground State:",
        "+7.07e-01 +7.07e-01j |000>   |  1.00e+00 exp(i*pi*0.250) |000>",
        "+1.22e-16 +2.00e+00j |001>   |  2.00e+00 exp(i*pi*0.500) |001>",
        "-2.12e+00 +2.12e+00j |010>   |  3.00e+00 exp(i*pi*0.750) |010>",
        "-4.00e+00 +4.90e-16j |011>   |  4.00e+00 exp(i*pi*1.000) |011>",
        "-3.54e+00 -3.54e+00j |100>   |  5.00e+00 exp(i*pi*-0.750) |100>",
        "-1.10e-15 -6.00e+00j |101>   |  6.00e+00 exp(i*pi*-0.500) |101>",
        "+7.00e+00 -1.71e-15j |110>   |  7.00e+00 exp(i*pi*-0.000) |110>",
        "+2.45e-15 +8.00e+00j |111>   |  8.00e+00 exp(i*pi*0.500) |111>"
    ]
    for out, expected_out in zip(mock_print.call_args_list, expected_output):
        assert out.args[0] == expected_out

def test_error_pretty_print_results():
    """
    Test that pretty_print_results correctly catches the following errors:
        - State vector is not of length 2**n for some integer n
    """
    energy = 1.23456789
    state = [
        1+2j,
        3+4j,
        5+6j,
        7+8j,
        9+0j,
    ]
    
    # If the correct error is raised, continue
    # If a different error or no error is raised, raise an error
    try:
        pretty_print_results(energy, state)
    except ValueError as e:
        if str(e) != "Vector must be of length 2**n where n is some integer":
            raise ValueError("Incorrect error raised")
    else:
        raise ValueError("Error should have been raised")

# Helper functions/objects for the next two tests
operator_dict = {
    'i': np.array([[1, 0], [0, 1]]),
    'x': np.array([[0, 1], [1, 0]]),
    'y': np.array([[0, -1j], [1j, 0]]),
    'z': np.array([[1, 0], [0, -1]]),
}

def operator_str_to_matrix(operator_string):
    """
    Convert an operator string (e.g. "IIXZIY") into a matrix
    """
    op = operator_dict[operator_string[0].lower()]
    for op_i in operator_string[1:]: 
        op = np.kron(op, operator_dict[op_i.lower()])
    return op

### COMMENTING OUT THIS TEST SINCE IT TAKES A WHILE TO EXECUTE

# def test_VQE_HT():
#     """
#     Test that, given a simple problem a large set of randomised starting parameters, VQE_HT successfully converges the solution within some small error for at least one of the runs.

#     OLD TEST IDEA:  Test that, given some problem with the solution as the initial parameter values, VQE_HT successfully converges the solution within some small error
#     since it is already starting at the global minimum.

#         - It took too long trying to find analytical solutions to the parameters in the state preparation sequence, even for some simple 2 qubit problems
#     """

#     test_Hams = [
#         [
#             (1, "ZI"), 
#             (1, "IZ"),
#             (0.5, "XX"),
#         ],
#         [
#             (1, "ZII"), 
#             (1, "IZI"),
#             (1, "IIZ"),
#             (0.5, "XYX")
#         ],
#         [
#             (1, "XXI"),
#             (1, "IYY"),
#             (1, "ZIZ"),
#             (1, "IZZ")
#         ]
#     ]
    
#     for ham in test_Hams:
#         HMat = sum([g*operator_str_to_matrix(opStr) for g, opStr in ham])
#         expected_eig = sorted(np.linalg.eigvals(HMat))[0]
#         depth = 1
#         # For some Hamiltonians, the ground state exists outside of the space of states reachable by the state preparation circuit
#         # In the case that the correct solution is not converged upon quickly, we increase the depth of the state preparation circuit 
#         # (an consequently introduce more parameters) to increase the space of reachable states
#         for i in range(100):
#             if i > 25:
#                 depth = 2
#             if i > 50:
#                 depth = 3
#             if i > 75:
#                 depth = 4
#             if abs(expected_eig - VQE_HT(ham, depth)[0]) < 1e-4:
#                 break
#         else:
#             raise ValueError("VQE_HT unable to converge on correct solution")

def test_VQE_Sim():
    """
    Test that, given a simple problem a large set of randomised starting parameters, VQE_Sim successfully converges the solution within some small error for at least one of the runs.

    OLD TEST IDEA:  Test that, given some problem with the solution as the initial parameter values, VQE_HT successfully converges the solution within some small error
    since it is already starting at the global minimum.

        - It took too long trying to find analytical solutions to the parameters in the state preparation sequence, even for some simple 2 qubit problems
    """

    test_Hams = [
        [
            (1, "ZI"), 
            (1, "IZ"),
            (0.5, "XX"),
        ],
        [
            (1, "ZII"), 
            (1, "IZI"),
            (1, "IIZ"),
            (0.5, "XYX")
        ],
        [
            (1, "XXI"),
            (1, "IYY"),
            (1, "ZIZ"),
            (1, "IZZ")
        ]
    ]
    
    for ham in test_Hams:
        HMat = sum([g*operator_str_to_matrix(opStr) for g, opStr in ham])
        expected_eig = sorted(np.linalg.eigvals(HMat))[0]   # correct eigval
        depth = 1
        # For some Hamiltonians, the ground state exists outside of the space of states reachable by the state preparation circuit
        # In the case that the correct solution is not converged upon quickly, we increase the depth of the state preparation circuit 
        # (an consequently introduce more parameters) to increase the space of reachable states
        for i in range(100):
            if i > 25:
                depth = 2
            if i > 50:
                depth = 3
            if i > 75:
                depth = 4
            if abs(expected_eig - VQE_Sim(ham, depth)[0]) < 1e-4:
                break
        else:
            raise ValueError("VQE_Sim unable to converge on correct solution")