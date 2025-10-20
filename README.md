# IEEE 754 Floating-Point Arithmetic Demo

## Description

This project demonstrates IEEE 754 floating-point arithmetic concepts through 7 comprehensive exercises, including binary conversion, arithmetic operations, special values handling, rounding modes, overflow/underflow scenarios, precision visualization, and cross-language comparison.

## Requirements

- Python 3.7+
- numpy
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IEEE_754_Demo
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

Or use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy matplotlib
```

## How to Run

```bash
python ieee754_demo.py
```

## Output

The program will:
- Display results for all 7 exercises in the terminal
- Generate a precision loss visualization saved as `ieee754_precision_loss.png`
- Show test cases for both normal and edge cases

## Exercises Covered

1. **Decimal to IEEE 754 Conversion** - Binary representation of floating-point numbers
2. **Arithmetic Operations** - Demonstrates precision limitations (e.g., 0.1 + 0.2 ≠ 0.3)
3. **Special Values** - Handling of infinity and NaN
4. **Rounding Modes** - Different rounding strategies and banker's rounding
5. **Overflow/Underflow** - Boundary conditions and denormalized numbers
6. **Precision Visualization** - Graphical representation of precision loss
7. **Comparative Study** - How different programming languages handle IEEE 754