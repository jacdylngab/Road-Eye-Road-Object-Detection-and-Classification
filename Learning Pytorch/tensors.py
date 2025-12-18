import torch
import numpy as np

"""
Tensors are a specialized data structure that are very similar to arrays 
and matrices. In PyTorch, we use tensors to encode the inputs 
and outputs of a model, as well as the model’s parameters.
"""

######################################################################
######################## TENSOR INITIALIZATION #######################
######################################################################

#Tensors can be initialized in various ways. Take a look at the following examples:

# Directly from data:
# Tensors can be created directly from data.
# The data type is automatically inferred.
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From a NumPy array:
# Tensors can be created from NumPy arrays (ans vice versa)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print("==================== TENSOR INITIALIZATION1 ==================")
# From another tensor:
# The new tensor retains the properties (shape, datatype) of the argument tensor,
# unless explicitly overridden
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# With random or constant values:
# 'shape' is a tuple of tensor dimensions. In the functions below, it determines the dimensionality
# of the output tensor
shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print("==================== TENSOR INITIALIZATION2 ==================")
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

######################################################################
########################### TENSOR ATTRIBUTES ########################
######################################################################

# Tensor attributes describe their shapes, datatype, and the device on which they are stored

tensor = torch.rand(3, 4)

print("==================== TENSOR ATTRIBUTES ======================")
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

######################################################################
########################### TENSOR OPERATIONS ########################
######################################################################

print("==================== TENSOR OPERATIONS ======================")

# There are over 100 tensor operations, including transposing, indexing, slicing, mathematical operations, linear
# algebra, random sampling, and more, etc..

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}")

# Standard numpy-like indexing and slicing:
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print("ONES:")
print(tensor)

# Joining tensors. 
# 'torch.cat' can be used to concatenate a sequence of tensors along a given
# dimension.

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print("JOINING TENSORS:")
print(t1)

# Multiplying tensors

print("MULTIPLYING TENSORS:")
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# This computes the matrix multiplication between two tensors
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In-place operations: 
# Operations that have a _ suffix are in-place. For example: x.copy_(y), x.t_(), will change x.

print(tensor, "\n")
tensor.add_(5)
print(tensor)


######################################################################
########################### BRIDGE WITH NUMPY ########################
######################################################################

print("==================== BRIDGE WITH NUMPY ======================")

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
