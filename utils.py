import numpy as np
import trimesh


def calculate_contact_pose(mesh, start_frame_for_push, goal_frame_for_push):
    start_position = start_frame_for_push[:3, 3]
    goal_position = goal_frame_for_push[:3, 3]

    push_dir = goal_position - start_position
    push_dir_length = np.linalg.norm(push_dir)
    push_type = ""
    if push_dir_length == 0:
        contact_pose = calculate_rotation_pose(start_frame_for_push, mesh)
        push_type = "rotational"
    else:
        normed_push_dir = push_dir / push_dir_length
        orientation_for_start_frame_for_push = start_frame_for_push[:3, :3]
        contact_pose = calculate_translation_pose(
            start_position, normed_push_dir, orientation_for_start_frame_for_push, mesh
        )
        push_type = "translational"
    return contact_pose, push_type


def calculate_rotation_pose(start_pose, object_mesh):
    orientation_of_start_pose = start_pose[:3, :3]
    position_of_start_pose = start_pose[:3, 3]
    # The approach dir is always the z-direction of the gripper
    approach_dir = orientation_of_start_pose[:, 2]
    gripper_approach_ray_intersection_with_mesh = ray_mesh_intersection(
        position_of_start_pose,
        approach_dir,
        object_mesh,
    )
    pose = np.eye(4)
    pose[:3, 3] = gripper_approach_ray_intersection_with_mesh
    pose[:3, :3] = orientation_of_start_pose
    return pose


def calculate_translation_pose(start_position, push_dir, push_orientation, object_mesh):
    mesh_ray_intersection = ray_mesh_intersection(start_position, push_dir, object_mesh)
    position_for_pre_manipulation = translate_point(
        mesh_ray_intersection, push_dir, 0.02
    )
    pose = np.eye(4)
    pose[:3, 3] = position_for_pre_manipulation
    pose[:3, :3] = push_orientation
    # The y_axis of the push_orientation always need to be parallel to the push_dir. If this is not
    # the case, we need to rotate the push_orientation around the push_dir
    y_axis = push_orientation[:, 1]
    angle = angle_between_vectors(y_axis, push_dir)
    if angle > (np.pi / 180) * 5:
        # We will always rotate around the z-axis of the push_orientation
        rotation_axis = push_orientation[:, 2]
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
        pose[:3, :3] = rotation_matrix[:3, :3] @ push_orientation
        # Round pose[:3, :3] to 6 decimal places
        pose[:3, :3] = np.round(pose[:3, :3], 6)
    return pose


def ray_mesh_intersection(P, D, trimesh_obj):
    closest_intersection = None
    closest_t = float("inf")
    for face in trimesh_obj.faces:
        v0 = trimesh_obj.vertices[face[0]]
        v1 = trimesh_obj.vertices[face[1]]
        v2 = trimesh_obj.vertices[face[2]]
        intersection = ray_triangle_intersection(P, D, v0, v1, v2)
        if intersection is not None:
            t = np.dot(intersection - P, D)
            if 0 <= t < closest_t:
                closest_t = t
                closest_intersection = intersection
    return closest_intersection


def translate_point(point, vector, t=1.0):
    translated_point = point + t * vector
    return translated_point


def ray_triangle_intersection(P, D, v0, v1, v2):
    # Vertices of the triangle: v0, v1, v2
    # Ray: P + t*D
    epsilon = 1.0e-6

    # Compute triangle edges
    e1 = v1 - v0
    e2 = v2 - v0

    # Compute determinant
    h = np.cross(D, e2)
    det = np.dot(e1, h)

    # If determinant is near zero, ray is in the plane of the triangle
    if abs(det) < epsilon:
        return None

    inv_det = 1.0 / det
    s = P - v0
    u = np.dot(s, h) * inv_det

    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, e1)
    v = np.dot(D, q) * inv_det

    if v < 0.0 or u + v > 1.0:
        return None

    t = np.dot(e2, q) * inv_det

    if t > epsilon:
        return P + D * t  # Intersection point

    return None


def angle_between_vectors(vec_a, vec_b, degrees=False):
    assert (
        np.round(np.linalg.norm(vec_a), 6) == np.round(np.linalg.norm(vec_b), 6) == 1
    ), "Input vectors are not normalized"

    cos_theta = vec_a.dot(vec_b)
    cos_theta = np.clip(cos_theta, -1, 1)

    angle = np.arccos(cos_theta)
    if degrees:
        angle = np.rad2deg(angle)
    return angle


def load_mesh(mesh_path):
    mesh = trimesh.load(mesh_path)
    return mesh


def quat2rotmat(q):
    """
    Convert a quaternion to a rotation matrix.

    Args:
        q: A 4-element unit quaternion.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2
    R[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
    R[0, 2] = 2 * (q[0] * q[2] + q[1] * q[3])
    R[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
    R[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2
    R[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
    R[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
    R[2, 1] = 2 * (q[0] * q[1] + q[2] * q[3])
    R[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2
    return R


def rotmat2quat(R):
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R.flatten()
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz])
    assert (R == np.round(quat2rotmat(q))).all()
    return q
