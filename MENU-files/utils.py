import os

# Function to test for valid input and convert to int for further processing
def input_int_validate():
    while True:
        try:
            validate = int(input(": "))
            break
        except ValueError: # Catch specific ValueError for non-integer input
            print("\nIncorrect value! Please make a new selection\n")
    return validate
# Function to check and warn about large pattern sizes
def pattern_size_check():
    while True:
        # Prompt user for pattern size 
        size = int(input("Enter pattern size (e.g., '5' for 5x5): "))
        # Warn the user if too computationaly intensive
        if size > 5:
            qubits = size * size
            print(f"\n   - WARNING: You have selected a {size}x{size} pattern ({qubits} qubits).")
            print("   - Simulations above ~25 qubits are extremely memory-intensive and may fail on standard computers.\n")
        
            proceed = input("Are you sure you want to continue? (y/n): ")
            if proceed.lower() != 'y':
                continue # This will restart the main menu loop
        return size
# Function to clear the console screen
def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')