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


def to_voigt(tensor):
    """Convert 3x3 tensor to voigt vector."""
    if tensor.shape[0] != 3 or tensor.shape[1] != 3:
        raise ValueError("The provided tensor has to be 3x3!")

    voigt = np.zeros(6)
    voigt[0] = tensor[0, 0]
    voigt[1] = tensor[1, 1]
    voigt[2] = tensor[2, 2]
    voigt[3] = tensor[1, 2]
    voigt[4] = tensor[0, 2]
    voigt[5] = tensor[0, 1]
    return voigt


def to_full_tensor(voigt):
    """Convert from Voigt representation back to full tensor."""
    if len(voigt) != 6:
        raise ValueError("The Voigt vector has to be of length 6!")

    tensor = np.zeros((3, 3))
    tensor[0, 0] = voigt[0]
    tensor[1, 1] = voigt[1]
    tensor[2, 2] = voigt[2]
    tensor[1, 2] = voigt[3]
    tensor[2, 1] = voigt[3]
    tensor[0, 2] = voigt[4]
    tensor[2, 0] = voigt[4]
    tensor[0, 1] = voigt[5]
    tensor[1, 0] = voigt[5]
    return tensor


def rotate_tensor(tensor, rot_matrix):
    """Rotate a tensor."""
    return rot_matrix.dot(tensor).dot(rot_matrix.T)


def rot_matrix_spherical_coordinates(phi, theta):
    """Find the rotation matrix in spherical coordinates.abs

    After the rotation the z-axis points along
    z' = [cos(phi) sin(theta), sin(phi)sin(theta), cos(theta)]

    :param float phi: Azimuthal angle
    :param float theta: polar angle
    """
    phi_deg = 180.0 * phi / np.pi
    theta_deg = 180.0 * theta / np.pi
    seq = [("z", phi_deg), ("y", theta_deg)]
    seq = [("y", -theta_deg), ("z", -phi_deg)]
    matrix = rot_matrix(seq)

    # Make sure that the rotation is correct
    target_z = np.array([np.cos(phi)*np.sin(theta),
                         np.sin(phi)*np.sin(theta),
                         np.cos(theta)])

    z_from_matrix = matrix.dot([0, 0, 1])
    assert np.allclose(z_from_matrix, target_z)
    return matrix
