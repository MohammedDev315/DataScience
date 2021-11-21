
# Create variables with descriptive names to store the conversion constants.
from_C_to_F = 0
from_F_to_C = 0

# There is a Celsius to Fahrenheit scaler, 9/5, and an offset, 32.
OFFESET = 32
F_SCALER = 5/9
C_SCALER = 9/5

# Create variables Celsius and one in Fahrenheit, and values 20 and 60. Hint: simultaneous assignment
celsius, fahrenheit = 20, 60

# Use compound assignment operators and the conversion constants to convert 20°C to Fahrenheit.
from_C_to_F = (celsius * F_SCALER) + OFFESET

# Use compound assignment operators and the conversion constants to convert 60°F to Celsius.
from_F_to_C = (fahrenheit - OFFESET) * C_SCALER

print(f"Convert 20C to F = {from_C_to_F}  -- convert 60F to C = {from_F_to_C} ")


