"""Module providing some useful functions."""
import numpy as np


def rotZ(angle):
    """Return the rotation matrix for rotations around the Z axis.

    :param angle: Rotation angle in radians
    """
    matrix = np.identity(3)
    matrix[0, 0] = np.cos(angle)
    matrix[0, 1] = np.sin(angle)
    matrix[1, 0] = -np.sin(angle)
    matrix[1, 1] = np.cos(angle)
    return matrix


def rotX(angle):
    """Return the rotation matrix for rotations around the X axis.

    :param angle: Rotation angle in radians
    """
    matrix = np.identity(3)
    matrix[1, 1] = np.cos(angle)
    matrix[1, 2] = np.sin(angle)
    matrix[2, 1] = -np.sin(angle)
    matrix[2, 2] = np.cos(angle)
    return matrix


def rotY(angle):
    """Return the rotation matrix for rotations around the Y axis.

    :param angle: Rotation angle in radians
    """
    matrix = np.identity(3)
    matrix[2, 2] = np.cos(angle)
    matrix[2, 0] = np.sin(angle)
    matrix[0, 2] = -np.sin(angle)
    matrix[0, 0] = np.cos(angle)
    return matrix


def rot_matrix(sequence):
    """Construct the rotation matrix corresponding to a sequence of rotations.

    :param sequence: List of tuples [("x", 30), ("y", 10), ("x", -10)]
                     First item in the tuples is the axis.
                     Second item in the tuple is angle in degrees
    """
    mat = np.identity(3)
    for rot in sequence:
        angle = np.radians(rot[1])
        axis = rot[0]
        if axis == "x":
            mat = rotX(angle).dot(mat)
        elif axis == "y":
            mat = rotY(angle).dot(mat)
        elif axis == "z":
            mat = rotZ(angle).dot(mat)
        else:
            raise ValueError("Axis has to be x, y or z!")
    return mat


def to_mandel(tensor):
    """Convert 3x3 tensor to mandel vector."""
    if tensor.shape[0] != 3 or tensor.shape[1] != 3:
        raise ValueError("The provided tensor has to be 3x3!")

    mandel = np.zeros(6)
    mandel[0] = tensor[0, 0]
    mandel[1] = tensor[1, 1]
    mandel[2] = tensor[2, 2]
    mandel[3] = np.sqrt(2.0)*tensor[1, 2]
    mandel[4] = np.sqrt(2.0)*tensor[0, 2]
    mandel[5] = np.sqrt(2.0)*tensor[0, 1]
    return mandel


def to_full_tensor(mandel):
    """Convert from mandel representation back to full tensor."""
    if len(mandel) != 6:
        raise ValueError("The mandel vector has to be of length 6!")

    tensor = np.zeros((3, 3))
    tensor[0, 0] = mandel[0]
    tensor[1, 1] = mandel[1]
    tensor[2, 2] = mandel[2]

    tensor[1, 2] = mandel[3]/np.sqrt(2.0)
    tensor[2, 1] = mandel[3]/np.sqrt(2.0)
    tensor[0, 2] = mandel[4]/np.sqrt(2.0)
    tensor[2, 0] = mandel[4]/np.sqrt(2.0)
    tensor[0, 1] = mandel[5]/np.sqrt(2.0)
    tensor[1, 0] = mandel[5]/np.sqrt(2.0)
    return tensor


def rotate_tensor(tensor, rot_matrix):
    """Rotate a tensor."""
    assert len(tensor.shape) == 2
    return rot_matrix.dot(tensor).dot(rot_matrix.T)


def rotate_rank4_tensor(tensor, rot_matrix):
    """Rotate a rank for tensor."""
    assert len(tensor.shape) == 4
    tensor = np.einsum("lp,mnqp->mnql", rot_matrix, tensor)
    tensor = np.einsum("kq,mnql->mnkl", rot_matrix, tensor)
    tensor = np.einsum("jn,mnkl->mjkl", rot_matrix, tensor)
    tensor = np.einsum("im,mjkl->ijkl", rot_matrix, tensor)
    # rot = rot_matrix
    # T = tensor
    # tensor = np.einsum('ai,bj,ck,dl,abcd->ijkl', rot, rot, rot, rot, T)
    return tensor


def rotate_rank4_mandel(tensor, rot_matrix):
    assert len(tensor.shape) == 2
    rank4_tensor = to_full_rank4(tensor)
    rank4_tensor = rotate_rank4_tensor(rank4_tensor, rot_matrix)
    return to_mandel_rank4(rank4_tensor)


def rot_matrix_spherical_coordinates(phi, theta):
    """Find the rotation matrix in spherical coordinates.abs

    After the rotation the z-axis points along
    z' = [cos(phi) sin(theta), sin(phi)sin(theta), cos(theta)]

    :param float phi: Azimuthal angle
    :param float theta: polar angle
    """
    phi_deg = 180.0 * phi / np.pi
    theta_deg = 180.0 * theta / np.pi
    #seq = [("z", phi_deg), ("y", theta_deg)]
    seq = [("y", -theta_deg), ("z", -phi_deg)]
    matrix = rot_matrix(seq)

    # Make sure that the rotation is correct
    # by checking how the z-axis is rotated
    target_z = np.array([np.cos(phi)*np.sin(theta),
                         np.sin(phi)*np.sin(theta),
                         np.cos(theta)])

    z_from_matrix = matrix.dot([0, 0, 1])
    assert np.allclose(z_from_matrix, target_z)
    return matrix


def to_mandel_rank4(tensor):
    """Convert rank 4 tensor to mandel notation."""
    from itertools import product
    assert len(tensor.shape) == 4
    out = np.zeros((6, 6))
    mandel_lut = {
        (0, 0): 0,
        (1, 1): 1,
        (2, 2): 2,
        (1, 2): 3,
        (0, 2): 4,
        (0, 1): 5
    }
    for ind in product([0, 1, 2], repeat=4):
        if ind[1] < ind[0] or ind[3] < ind[2]:
            continue
        row = mandel_lut[(ind[0], ind[1])]
        col = mandel_lut[(ind[2], ind[3])]

        if ind[0] != ind[1] and ind[2] != ind[3]:
            value = 2.0*tensor[ind[0], ind[1], ind[2], ind[3]]
        elif ind[0] != ind[1]:
            value = np.sqrt(2.0)*tensor[ind[0], ind[1], ind[2], ind[3]]
        elif ind[2] != ind[3]:
            value = np.sqrt(2.0)*tensor[ind[0], ind[1], ind[2], ind[3]]
        else:
            value = tensor[ind[0], ind[1], ind[2], ind[3]]
        out[row, col] = value
    return out


def to_full_rank4(mandel_tensor):
    """Convert Mandel representation to full tensor."""
    from itertools import product
    assert len(mandel_tensor.shape) == 2
    out = np.zeros((3, 3, 3, 3))
    mandel_lut = [(0, 0), (1, 1), (2, 2), 
                    (1, 2), (0, 2), (0, 1)]
    
    for ind in product(range(6), repeat=2):
        if ind[0] > 2 and ind[1] > 2:
            value = mandel_tensor[ind[0], ind[1]]/2.0
        elif ind[0] > 2 or ind[1] > 2:
            value = mandel_tensor[ind[0], ind[1]]/np.sqrt(2.0)
        else:
            value = mandel_tensor[ind[0], ind[1]]

        row = mandel_lut[ind[0]]
        col = mandel_lut[ind[1]]

        out[row[0], row[1], col[0], col[1]] = value
        out[row[0], row[1], col[1], col[0]] = value
        out[row[1], row[0], col[1], col[0]] = value
        out[row[1], row[0], col[0], col[1]] = value
    return out
