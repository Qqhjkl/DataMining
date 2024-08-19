class Logic:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right

    def __str__(self):
        if self.left is None and self.right is None:
            return self.name
        elif self.right is None:
            return f"{self.name}({self.left})"
        else:
            return f"({self.left} {self.name} {self.right})"


def distribute(expr):
    if expr.name == "And":
        if expr.left.name == "Or":
            return Logic("Or", distribute(Logic("And", expr.left.left, expr.right)), distribute(Logic("And", expr.left.right, expr.right)))
        elif expr.right.name == "Or":
            return Logic("Or", distribute(Logic("And", expr.left, expr.right.left)), distribute(Logic("And", expr.left, expr.right.right)))
    elif expr.name == "Or":
        return Logic("Or", distribute(expr.left), distribute(expr.right))
    elif expr.name == "Not":
        return Logic("Not", distribute(expr.left))
    elif expr.name in ["Imply", "Equiv"]:
        return Logic(expr.name, distribute(expr.left), distribute(expr.right))
    else:
        return expr


def deMorgan(expr):
    if expr.name == "Not":
        if expr.left.name == "And":
            return Logic("Or", deMorgan(Logic("Not", expr.left.left)), deMorgan(Logic("Not", expr.left.right)))
        elif expr.left.name == "Or":
            return Logic("And", deMorgan(Logic("Not", expr.left.left)), deMorgan(Logic("Not", expr.left.right)))
    elif expr.name == "And" or expr.name == "Or" or expr.name == "Imply" or expr.name == "Equiv":
        return Logic(expr.name, deMorgan(expr.left), deMorgan(expr.right))
    else:
        return expr


# Example usage:
# Define the expressions
A = Logic("A")
B = Logic("B")
C = Logic("C")

# Create complex expressions
expression1 = Logic("And", Logic("Or", A, B), C)
expression2 = Logic("Not", Logic("And", A, B))

# Apply the distribute function
distributed_expression1 = distribute(expression1)
print("Distributed expression:", distributed_expression1)

# Apply the deMorgan function
demorgan_expression2 = deMorgan(expression2)
print("DeMorgan expression:", demorgan_expression2)
