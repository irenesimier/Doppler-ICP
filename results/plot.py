import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import glob
import os
import open3d as o3d

# ---------- Functions ----------

def load_tum_poses(filename):
    """Load TUM trajectory: timestamp x y z qx qy qz qw"""
    data = np.loadtxt(filename)
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]  # [qx, qy, qz, qw]
    r = R.from_quat(quaternions)
    euler = r.as_euler('xyz', degrees=True)  # roll, pitch, yaw
    return positions, euler

def load_doppler_speeds(pcd_dir, n_frames=None):
    """Load all .bin files and extract Doppler speeds (mean per frame)"""
    files = sorted(glob.glob(os.path.join(pcd_dir, "*.bin")))
    if n_frames is not None:
        files = files[:n_frames]
    speeds = []
    for f in files:
        data = np.fromfile(f, dtype=np.float32).reshape(-1,4)
        speeds.append(np.mean(data[:,3]))
    return np.array(speeds)

# ---------- Paths ----------

# ref_file = "dataset/carla-town04-straight-walls/ref_poses.txt"
# doppler_file = "results/town04-output-doppler/icp_poses.txt"
# ptp_file = "results/town04-output-ptp/icp_poses.txt"

ref_file = "dataset/carla-town05-curved-walls/ref_poses.txt"
doppler_file = "results/town05-output-doppler/icp_poses.txt"
ptp_file = "results/town05-output-ptp/icp_poses.txt"

# ---------- Load data ----------

ref_pos, ref_rpy = load_tum_poses(ref_file)
dop_pos, dop_rpy = load_tum_poses(doppler_file)
ptp_pos, ptp_rpy = load_tum_poses(ptp_file)

# ---------- 1) 3D Trajectories ----------

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(ref_pos[:,0], ref_pos[:,1], ref_pos[:,2], color='k', label='Reference')
ax.plot(dop_pos[:,0], dop_pos[:,1], dop_pos[:,2], color='r', linestyle='--', label='Doppler ICP')
ax.plot(ptp_pos[:,0], ptp_pos[:,1], ptp_pos[:,2], color='b', linestyle='-.', label='Point-to-plane ICP')
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
ax.set_title('3D Trajectories')
ax.legend()
plt.tight_layout()
plt.show()

# ---------- 2) RPY 3 subplots ----------

fig, axs = plt.subplots(3,1, figsize=(12,10), sharex=True)

# Roll
axs[0].plot(ref_rpy[:,0], color='k', label='Ref Roll')
axs[0].plot(dop_rpy[:,0], color='r', linestyle='--', label='Doppler Roll')
axs[0].plot(ptp_rpy[:,0], color='b', linestyle='-.', label='Point-to-plane Roll')
axs[0].set_ylabel('Roll [deg]'); axs[0].legend()

# Pitch
axs[1].plot(ref_rpy[:,1], color='k', label='Ref Pitch')
axs[1].plot(dop_rpy[:,1], color='orange', linestyle='--', label='Doppler Pitch')
axs[1].plot(ptp_rpy[:,1], color='cyan', linestyle='-.', label='Point-to-plane Pitch')
axs[1].set_ylabel('Pitch [deg]'); axs[1].legend()

# Yaw
axs[2].plot(ref_rpy[:,2], color='k', label='Ref Yaw')
axs[2].plot(dop_rpy[:,2], color='brown', linestyle='--', label='Doppler Yaw')
axs[2].plot(ptp_rpy[:,2], color='magenta', linestyle='-.', label='Point-to-plane Yaw')
axs[2].set_xlabel('Frame'); axs[2].set_ylabel('Yaw [deg]'); axs[2].legend()

plt.tight_layout()
plt.show()
