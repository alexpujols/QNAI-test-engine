##!/usr/bin/env python
'''
                      ::::::
                    :+:  :+:
                   +:+   +:+
                  +#++:++#++:::::::
                 +#+     +#+     :+:
                #+#      #+#     +:+
               ###       ###+:++#""
                         +#+
                         #+#
                         ###
'''
__author__ = "Alex Pujols"
__copyright__ = "Alex Pujols"
__credits__ = ["Alex Pujols"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Alex Pujols"
__email__ = "A.Pujols@o365.ncu.edu; alexpujols@ieee.org"
__status__ = "Prototype"

'''
Title	      :	{Quantum Neuromorphic Artificial Intelligence Test Engine}
Date		  :	{05-18-2025}
Description   :	{A Python/Cirq test engine that simulates multiple quantum and classical neural networks for testing/simulation purposes}
Options	      :	{TBD}
Notes	      :	{Available at Github at https://github.com/alexpujols/QNAI-test-engine}
'''

## Import modules declarations
import cirq

## Class declarations

# Class for TBD

# Class for TBD

# Class for TBD


## Function declarations

# Function to test for valid input and convert to int for further processing
def input_int_validate():
    while True:
        try:
            validate = int(input(": "))
            break
        except:
            print ("Incorrect value! Please make a new selection")
    return validate
# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD


### Main code begins ###

# Set global vaiables

# Begin main code execution
while True:
    print ("\n")
    print ("Hi, which quantum simulation scenario would you like to run?")
    print ("1 - Pattern Matching")
    print ("2 - Problem Solving")
    print ("3 - Creative Thinking")
    print ("4 - SyDGE")
    print ("0 - EXIT")

    # Take user input and validate
    selection = input_int_validate()

    # Take action based on user selection
    if (selection == 1):
        print ("\nYou selected Pattern Matching!\n")
    
    # Pick a qubit.
    qubit = cirq.GridQubit(0, 0)
    
    # Create a circuit that applies a square root of NOT gate, then measures the qubit.
    circuit = cirq.Circuit(cirq.X(qubit) ** 0.5, cirq.measure(qubit, key='m'))
    print("Circuit:")
    print(circuit)
   
    # Simulate the circuit several times.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=20)
    print("Results:")
    print(result)
    
    if (selection == 2):
        print ("\nYou selected Problem Solving!\n")
    if (selection == 3):
        print ("\nYou selected Creative Thinking!\n")
    if (selection == 4):
        print ("\nYou selected Sythetic Data Generation Engine (SyDGE)\n")
    if (selection == 0):
        print ("\n You have chosen to leave the program.  Goodbye! \n")
        break
## Main code ends ###
