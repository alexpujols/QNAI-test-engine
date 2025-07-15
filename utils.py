# Function to test for valid input and convert to int for further processing
def input_int_validate():
    while True:
        try:
            validate = int(input(": "))
            break
        except ValueError: # Catch specific ValueError for non-integer input
            print("\nIncorrect value! Please make a new selection\n")
    return validate