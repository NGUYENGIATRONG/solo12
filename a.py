import numpy as np

def check_valid(x, y, l1=0.16, l2=0.16):
    discriminant = 16 * l2**2 * y**2 - 4 * (-l1**2 + l2**2 - 2 * l2 * x + x**2 + y**2) * (-l1**2 + l2**2 + 2 * l2 * x + x**2 + y**2)
    return discriminant >= 0

# Kiểm tra ví dụ
x = 0.1
y = 0.1
print("Valid:", check_valid(x, y))
