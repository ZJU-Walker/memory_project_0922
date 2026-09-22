"""Forward kinematics of the Franka Panda in plain numpy (base frame): joint angles -> tool-center position.

Craig's modified DH parameters of the Panda arm (Franka's published values) plus the hand: link8 (flange) is
0.107 m past joint 7, the hand is rotated -45 degrees about the flange z axis and the tool center point sits
0.1034 m further along the hand z axis (ManiSkill's `panda_hand_tcp`). `tcp_positions(q[..., 7])` -> xyz[..., 3].

Check (home pose q = [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]): the flange sits near (0.307, 0, 0.590) and the tool
center near (0.307, 0, 0.487); `python panda_fk.py` prints both.
"""
import numpy as np

# (a_{i-1}, d_i, alpha_{i-1}) per joint, then the fixed flange transform
_DH = [
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
]
_FLANGE_D = 0.107
_HAND_YAW = -np.pi / 4
_TCP_D = 0.1034


def _mdh(a, d, alpha, theta):
    """Modified DH transform RotX(alpha) TransX(a) RotZ(theta) TransZ(d), vectorized over theta[...]."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.zeros(theta.shape + (4, 4))
    T[..., 0, 0] = ct
    T[..., 0, 1] = -st
    T[..., 0, 3] = a
    T[..., 1, 0] = st * ca
    T[..., 1, 1] = ct * ca
    T[..., 1, 2] = -sa
    T[..., 1, 3] = -sa * d
    T[..., 2, 0] = st * sa
    T[..., 2, 1] = ct * sa
    T[..., 2, 2] = ca
    T[..., 2, 3] = ca * d
    T[..., 3, 3] = 1.0
    return T


def frames(q):
    """Homogeneous transforms of the flange and of the tool center for q[..., 7] (radians)."""
    q = np.asarray(q, dtype=np.float64)
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for i, (a, d, alpha) in enumerate(_DH):
        T = T @ _mdh(a, d, alpha, q[..., i])
    flange = T @ _mdh(0.0, _FLANGE_D, 0.0, np.zeros(q.shape[:-1]))
    hand = flange @ _mdh(0.0, 0.0, 0.0, np.full(q.shape[:-1], _HAND_YAW))
    tcp = hand @ _mdh(0.0, _TCP_D, 0.0, np.zeros(q.shape[:-1]))
    return flange, tcp


def tcp_positions(q):
    """Tool-center xyz in the robot base frame for joint angles q[..., 7] (extra columns, e.g. the gripper, ignored)."""
    q = np.asarray(q, dtype=np.float64)[..., :7]
    return frames(q)[1][..., :3, 3]


if __name__ == "__main__":
    home = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
    fl, tcp = frames(home)
    print("flange", np.round(fl[:3, 3], 4), "tcp", np.round(tcp[:3, 3], 4))
