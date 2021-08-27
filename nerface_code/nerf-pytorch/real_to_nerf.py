import os
os.environ['PYOPENGL_PLATFORM'] = "egl"
#os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']
import pyrender

import subprocess
proc1 = subprocess.Popen(['scontrol', 'show', 'job', os.environ['SLURM_JOBID'], '-d'], stdout=subprocess.PIPE)
process = subprocess.run(['grep', '-oP', 'GRES=.*IDX:\K\d'], stdin=proc1.stdout, capture_output=True, text=True)
os.environ['EGL_DEVICE_ID'] = process.stdout.rstrip()
proc1.stdout.close()



import argparse
import numpy as np
import os
from PIL import Image
import json
from tqdm import tqdm, trange
import pyrender
import trimesh
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

def visualize(im):
    plt.imshow(im)
    plt.show()

def normalize(vector):
    return vector / np.linalg.norm(vector)

def lookAt(cam_pos_world, to_pos_world, up = np.array([0, 1, 0])):

    forward = normalize( cam_pos_world - to_pos_world) # pos if on unit sphere and model is centered

    right = normalize(np.cross(normalize(up), forward))
    up = normalize(np.cross(forward, right))

    camToWorld = np.zeros((4,4))

    camToWorld[0,:-1] = right
    camToWorld[1,:-1] = up
    camToWorld[2,:-1] = forward

    camToWorld[3,:-1] = cam_pos_world
    camToWorld[3,3] = 1
    return camToWorld.T


def lookAt_like_other_cam_from_different_pos(cam_pos_world, orig_cam_matrix, up = np.array([0, 1, 0])):
    gt_rot = orig_cam_matrix[:3, :3] # original rotation
    orig_pos_lookat_rot = lookAt(orig_cam_matrix[:3, -1], np.zeros(3), up)[:3, :3] # cam matrix that looks to 0 from orig point
    new_lookat_rot = lookAt(cam_pos_world, np.zeros(3),up)[:3, :3] # cam matrix that looks at 0 from new point

    new_rot = np.matmul(gt_rot, np.transpose(orig_pos_lookat_rot))
    new_rot = np.matmul(new_rot, new_lookat_rot)
    new_pose = np.eye(4)
    new_pose[:3, :3] = new_rot
    new_pose[:3, -1] = cam_pos_world
    return new_pose
    #color = render_debug_camera_matrix(new_pose, intrinsics, scale)
    #visualize(color)


def read_intrinsics(path_to_intrinsics_txt, im_size=None, center_crop_fix_intrinsics=False):
    all_intrinsics = np.genfromtxt(path_to_intrinsics_txt, dtype=None)
    if im_size:
        h,w = im_size
        fx = all_intrinsics[0][0] * -w  # TODO opposite?
        fy = all_intrinsics[0][1] * -h
        cx = all_intrinsics[0][2] * w  # TODO opposite?
        if center_crop_fix_intrinsics:
            cx = all_intrinsics[0][2] * w * 0.5625 # Fix for center cropping 1280 -> 720 1:1 ratio
        cy = (1-all_intrinsics[0][3]) * h # TODO opposite?
        return np.array([fx,fy,cx,cy]) # fx fy cx cy relative to im size unless provided im size
    else:
        return all_intrinsics[0]

def read_rigid_poses(path_to_rigid_poses_txt, mean_scale=True):
    all_rigids = np.genfromtxt(path_to_rigid_poses_txt, dtype=None)
    all_rigids = all_rigids.reshape(-1,4,4)
    all_rigids[:,:,0] *= -1 # fix guy
    all_rigids[:,:,2] *= -1 # fix guy
    scale = 0.5/np.mean(all_rigids[:,2,-1]) # mean z_val # Scaling such that the camera is at z = ~0.5
    if mean_scale:
        print("scaling scene by %f" % scale)
        all_rigids[:,0:3,-1] *= scale
    print("rigids shape ", all_rigids.shape)
    return all_rigids, scale

def read_expressions(path_to_expressions_txt):
    expressions = np.genfromtxt(path_to_expressions_txt, dtype=None)
    print("expressions shape ", expressions.shape)
    return expressions

def read_img_folder(path_to_imgs_folder = "./images"):
    im_path_list = os.listdir(path_to_imgs_folder)
    im_path_list.sort()
    N = len(im_path_list)
    print("found %d images in image folder" % N)
    # get size of first image
    im0 = Image.open(os.path.join(path_to_imgs_folder, im_path_list[0]))
    im_size = im0.size

    return im_path_list, N, im_size

def train_val_partition(N, n_train, n_val, n_test):
    perm = np.random.permutation(N)
    i_train = perm[:n_train]
    i_val = perm[n_train:n_train+n_val]
    i_test = perm[n_train+n_val:n_train+n_val+n_test]
    return  {'train':i_train, 'val':i_val, 'test':i_test}

def create_subfolders(folder_name):
    ## Create all necessary subfolders
    subfolders = []
    modes = ['train', 'test', 'val']
    for mode_name in modes:
        subfolders += [mode_name]
    for subfolder in subfolders:
        path = os.path.join(folder_name, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)

def render_debug_camera_matrix(matrix, intrinsics, scale=1.0):
    rot = np.array([[0,-1,0,0],
                    [1, 0, 0, 0],
                    [0,0,1,0],
                    [0,0,0,1]])

    #print("matrix: \n", matrix)
    scene = pyrender.Scene(ambient_light=(0.5,0.5,0.5))
    sphere = trimesh.creation.uv_sphere(radius=0.005)
    box = trimesh.creation.box((0.05,0.15,0.1))
    face = trimesh.load_mesh('./average.off')
    face.apply_scale(1.0/1000000.0)
    face.apply_scale(scale)

    sphere.visual.vertex_colors = [1.0, 0.0, 0.0]
    #box.visual.vertex_colors = [1.0, 0., 0.]
    face.visual.vertex_colors = [1.0, 0.5, 0.5]

    #pose_sphere = np.eye(4)
    #pose_box = np.eye(4)
    pose_mesh = np.eye(4)
    #pose_mesh[:,1]*=1
    #pose_sphere[:,-1] = [0,0.0,0.0,1]
    #pose_sphere = matrix

    #pose_cam = np.eye(4)
    #pose_cam[:,3] = matrix[:,3]
    # new_mat = np.zeros_like(pose_cam)
    # new_mat[:,0] = matrix[:,0]
    # new_mat[:,1] = matrix[:,1]
    # new_mat[:,2] = matrix[:,2]
    # new_mat[:,3] = matrix[:,3]
    # new_mat[0,3] = matrix[1,3]
    # new_mat[1,3] = matrix[0,3]

    pose_cam = matrix
    #pose_cam = np.eye(4)
    #pose_cam[:,3] = [0,0,2.4,1]
    #pose_cam[:,0] *= -1
    #pose_cam[:,2] *= -1
    #pose_cam = lookAt(matrix[:3,-1],[0,0,0])
    #pose_cam = lookAt(np.array([0.0,0.0,2.422]),[0,0,0])


    #pose_cam = (matrix)
    #print("pose_cam: (cam2world) \n", pose_cam)
    #print("pose_sphere: \n", pose_sphere)
    #print("pose_mesh: \n", pose_mesh)


    #sphere_mesh = pyrender.Mesh.from_trimesh(sphere, poses=pose_sphere)
    #box_mesh = pyrender.Mesh.from_trimesh(box, poses=pose_box)
    face_mesh = pyrender.Mesh.from_trimesh(face, poses=pose_mesh)
    #scene.add(sphere_mesh)
    #scene.add(box_mesh)
    scene.add(face_mesh)
    light = pyrender.SpotLight(color=np.ones(3), intensity=24.0,
                               innerConeAngle=np.pi / 16.0)

    light_node = scene.add(light, pose=lookAt(np.array([2, 2, 2]), [0, 0, 0]))
    camera = pyrender.IntrinsicsCamera(intrinsics[0], intrinsics[1], intrinsics[2],
                                       intrinsics[3], 0.01, 100000)
    #print("intrinsics: ", intrinsics[0], intrinsics[1], intrinsics[2],intrinsics[3])
    camera_node = scene.add(camera, pose=pose_cam)
    r = pyrender.OffscreenRenderer(512,512)
    color, depth = r.render(scene, flags=16|1024)
    #color, depth = r.render(scene, flags= 512)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(color)
    # plt.figure()
    # plt.show()
    return color

def show_im(im):
    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show()

def find_bbox(im):
    """
    function that takes synthetic rendering of the track head, finds its bbox, enlarges it, and returns the coords
    relative to image size (multiply by im dimensions to get pixel location of enlarged bbox

    :param im:
    :return:
    """

    H, W, _ = np.shape(im)
    where = np.where(im[:, :, 0] < 255)
    h_min = np.min(where[0])
    h_max = np.max(where[0])
    w_min = np.min(where[1])
    w_max = np.max(where[1])

    h_span = h_max - h_min
    w_span = w_max - w_min
    ratio = 0.3
    h_min -= ratio * 0.9 * h_span
    h_max += ratio * 0.5 * h_span
    w_min -= ratio * 0.5 * w_span
    w_max += ratio * 0.5 * w_span

    h_min = int(np.clip(h_min, 0, H - 1))
    h_max = int(np.clip(h_max, 0, H - 1))
    w_min = int(np.clip(w_min, 0, W - 1))
    w_max = int(np.clip(w_max, 0, W - 1))

    # uncomment for visualization
    # synth[h_min:h_max,w_min] = 150
    # synth[h_min:h_max,w_max] = 150
    # synth[h_min,w_min:w_max] = 150
    # synth[h_max,w_min:w_max] = 150
    return np.array([h_min/H, h_max/H, w_min/W, w_max/W])


def custom_sequence(neutral_pose):

    x_neutral = neutral_pose[0,-1]
    y_neutral = neutral_pose[1,-1]
    z_neutral = neutral_pose[2,-1]
    # Camera point
    x = np.linspace(-0.6,0.6,60)
    y = np.linspace(-0.3,0.5,60)
    #z = 1.6
    camera_points = np.zeros((120,3))
    camera_points[:,0] = x_neutral
    camera_points[:,1] = y_neutral
    camera_points[:,2] = z_neutral

    camera_points[0:60,0] = x
    camera_points[60:120,1] = y
    #camera_points[:,2]=z
    rigids = np.zeros((120,4,4))
    for i in range(rigids.shape[0]):
        rigids[i] = lookAt(camera_points[i],np.zeros(3))

    # Expressions
    beta = np.linspace(-0.2,0.4,20)
    expressions = np.zeros((120,76))
    #expressions[120:140,1] = beta
    #expressions[140:160,2] = beta
    #expressions[160:180,3] = beta
    #expressions[180:200,4] = beta

    return expressions, rigids



def ellipse(a,b,N,half=False):
    x0 = np.linspace(-a,a, int(N//2))
    x = np.concatenate((x0, np.linspace(a,-a, int(N//2))))
    #z = np.sqrt(a**2-np.power(x))
    y0 = np.sqrt(b**2 - (b**2)/(a**2) * np.power(x0,2))
    y = np.concatenate((y0, -y0))
    if half:
        return x0,y0
    return x,y

def circle(r_squared,N,half=False):
    r = np.sqrt(r_squared)
    x0 = np.linspace(-0.4*r,0.4*r, int(N//2))
    x = np.concatenate((x0, -x0))
    #z = np.sqrt(a**2-np.power(x))
    #y0 = np.sqrt(b**2 - (b**2)/(a**2) * np.power(x0,2))
    y0 = np.linspace(-0.05*r,0.05*r, int(N//2))
    y = np.concatenate((y0, -y0))
    z0 = np.sqrt(r_squared-np.power(x0,2) - np.power(y0,2))
    z = np.concatenate((z0,z0))
    if half:
        return x0,y0,z0
    return x,y,z



def custom_sequence_circle(neutral_pose, xmin,xmax,ymin,ymax):
    n_pts = 120
    # elliptical motion for x and y
    x_neutral = neutral_pose[0,-1]
    y_neutral = neutral_pose[1,-1]
    z_neutral = neutral_pose[2,-1]
    dx = xmax - xmin
    dy = ymax - ymin

    # Camera point
    #x = np.linspace(-dx/2,dx/2,n_pts//2)
    #y = np.sqrt(dy**2 - (dy**2)/(dx**2) * np.power(x,2)) + y_neutral
    #x += x_neutral
    x,y = ellipse(dx/2,dy/2, n_pts)
    x += x_neutral
    y += y_neutral
    #np.linspace(-0.3,0.5,60)
    #z = 1.6
    half_n_points= int(n_pts//2)
    camera_points = np.zeros((n_pts,3))
    camera_points[:,0] = x
    camera_points[:,1] = y
    #camera_points[:half_n_points,1] = y
    #camera_points[half_n_points:,1] = -np.sqrt(dy**2 - (dy**2)/(dx**2) * np.power(x,2)) + y_neutral
    camera_points[:,2] = z_neutral-0.1

    #camera_points[0:60,0] = x
    #camera_points[60:120,1] = y
    #camera_points[:,2]=z
    rigids = np.zeros((n_pts,4,4))
    for i in range(rigids.shape[0]):
        rigids[i] = lookAt(camera_points[i],np.zeros(3))

    # Expressions
    return None, rigids


def custom_seq_presentation(rigid_poses, expressions_orig):
    N_same_start = 50
    N_ellipse = 100
    N_same_at_pose = 150
    acc = 0

    neutral_pose = rigid_poses.mean(0)
    out_poses = np.zeros((1000,4,4))
    expressions = expressions_orig
    x_mean = neutral_pose[0,-1]
    y_mean = neutral_pose[1,-1]
    z_mean = neutral_pose[2,-1]
    radius_squared = np.linalg.norm(neutral_pose[:3,-1])
    radius_squared = 0.5**2
    #z_mean = rigid_poses[20][2,-1]
    intrinsics = np.array([-2831.62112,   2559.32928 ,   258.428928  , 264.114688])
    scale = 0.3099918717132938

    # same as sequence
    out_poses[:N_same_start] = rigid_poses[:N_same_start]
    acc+= N_same_start
    last_matrix = out_poses[acc-1]

    ### Turn

    #x,y = ellipse(0.6,0.3,N_ellipse)
    x,y,z = circle(radius_squared, N_ellipse)

    #z = z_mean * np.abs(np.sin(x * np.pi/ 0.6))
    x += x_mean
    y += y_mean
    #z = np.sqrt(radius_squared) - np.sqrt(radius_squared - np.power(x,2)-np.power(y,2))

    # Go to ellipse start
    line = np.linspace(out_poses[acc-1,0:3,-1], np.array([x[0],y[0],z[0]]),int(N_ellipse//2))

    for i in range(len(line)):
        #out_poses[acc + i] = lookAt(np.array(line[i]), np.zeros(3))
        out_poses[acc + i] = lookAt_like_other_cam_from_different_pos(line[i],last_matrix)
    acc += int(N_ellipse//2)

    # Do Ellipse
    x0, x1 = x[:N_ellipse//2], x[N_ellipse//2:]
    y0, y1 = y[:N_ellipse//2], y[:N_ellipse//2]
    z0, z1 = z[:N_ellipse//2], z[N_ellipse//2:]

    for i in range(len(x0)):
        #out_poses[acc + i] = lookAt(np.array([x0[i],y0[i],z_mean]), np.zeros(3))
        out_poses[acc + i] = lookAt_like_other_cam_from_different_pos(np.array([x0[i],y0[i],z0[i]]), last_matrix)
    acc += len(x0)

    # Continue fixed pose
    out_poses[acc:acc+N_same_at_pose] = out_poses[acc-1]
    acc += N_same_at_pose

    # Turn
    for i in range(len(x1)):
        #out_poses[acc + i] = lookAt(np.array([x1[i],y1[i],z_mean]), np.zeros(3))
        out_poses[acc + i] = lookAt_like_other_cam_from_different_pos(np.array([x1[i],y1[i],z1[i]]), last_matrix)
    acc += len(x1)

    # Continue fixed pose
    out_poses[acc:acc+N_same_at_pose] = out_poses[acc-1]
    acc += N_same_at_pose

    # Fix expressions and move camera around
    x,y = ellipse(0.6,0.3,N_ellipse)
    x,y,z = circle(radius_squared, N_ellipse)
    x += x_mean
    y += y_mean
    for i in range(len(x)):
        #out_poses[acc + i] = lookAt(np.array([x[i],y[i],z_mean]), np.zeros(3))
        out_poses[acc + i] = lookAt_like_other_cam_from_different_pos(np.array([x[i],y[i],z[i]]), last_matrix)
    expressions[acc:acc + int(N_ellipse)] = expressions[acc-1]
    acc += int(N_ellipse)

    line = np.linspace(out_poses[acc-1,0:3,-1], rigid_poses[20][:3,-1],int(N_ellipse//2))
    for i in range(len(line)):
        #out_poses[acc + i] = lookAt(np.array(line[i]), np.zeros(3))
        out_poses[acc + i] = lookAt_like_other_cam_from_different_pos(np.array(line[i]), last_matrix)
    expressions[acc:acc + int(N_ellipse//2)] = expressions[acc-1]
    acc += int(N_ellipse//2)

    # Continue fixed pose, with expressions
    out_poses[acc:acc+N_same_at_pose] = out_poses[acc-1]
    acc += N_same_at_pose

    return  expressions[:acc], out_poses[:acc]


def custom_seq_presentation_v2(rigid_poses, expressions_orig):
    N_same_start = 50
    N_ellipse = 100
    N_same_at_pose = 150
    acc = 0
    from scipy.spatial.transform import Rotation as R
    xyz_angles = np.zeros((len(rigid_poses),3))
    poses_inv = []
    for i,pose in enumerate(rigid_poses):
        pose_inv = np.linalg.inv(pose)
        poses_inv.append(pose_inv)
        r = R.from_matrix(pose_inv[:3,:3]) # move to euler angles. fix cam rotating head
        xyz_angles[i] = r.as_euler('xyz', degrees=True)

    # get range of angles
    x_min, x_max = np.min(xyz_angles[:,0]), np.max(xyz_angles[:,0])
    y_min, y_max = np.min(xyz_angles[:,1]), np.max(xyz_angles[:,1])
    z_min, z_max = np.min(xyz_angles[:,2]), np.max(xyz_angles[:,2])

    # choose some random angles and interpolate between them
    N_waypoints = 4
    x=np.hstack(( np.array(xyz_angles[0,0]), np.array(xyz_angles[0,0]), np.random.uniform(x_min, x_max, N_waypoints)))
    y=np.hstack(( np.array(xyz_angles[0,1]), np.array(xyz_angles[0,1]), np.random.uniform(y_min, y_max, N_waypoints)))
    z=np.hstack(( np.array(xyz_angles[0,2]), np.array(xyz_angles[0,2]), np.random.uniform(z_min, z_max, N_waypoints)))



    N_waypoints = 2
    x = np.hstack( (np.array(xyz_angles[0,0]), x_min*0.5, x_max*0.5, x_max*0.5))
    y = np.hstack( (np.array(xyz_angles[0,1]), y_min*0.5, y_max*0.5, y_min*0.5))
    z = np.hstack( (np.array(xyz_angles[0,2]), [0], [0], [0]))

    out_angles=[]
    for i in range(N_waypoints+1):
        start = np.array([x[i], y[i], z[i]])
        end = np.array([x[i + 1], y[i + 1], z[i + 1]])
        # interpolate to next set of angles
        out_angles.append(np.linspace(start,end , 60))
        # Stay for a while
        out_angles.append(np.repeat(end[np.newaxis,:], 100, axis=0))
    # stack them all up
    out_angles = np.concatenate(out_angles,axis=0)
    N_out = out_angles.shape[0]
    print("number of angles: ", N_out)

    print("angles range in video: X: [%f,%f], Y: [%f,%f]: Z: [%f,%f]" % (x_min, x_max, y_min, y_max, z_min, z_max))

    # invert back to matrices and to moving camera fixed head
    r_out = R.from_euler('xyz', out_angles, degrees=True)
    rotations = r_out.as_matrix() # this assumes static camera moving head, need to invert

    out_poses = np.zeros((N_out, 4, 4))
    #out_poses[:,-1,-1] = 1
    #rot = np.eye(4)
    rot_inv = np.eye(4)
    # P is camera pose with head at 0. Then R is rotation to head.
    # In turning head space, R * P_inv is new head pose
    # Back to fixed head space, camera pose is P * R_inv = P * R_t
    for i in range(N_out):
        #rot[:3,:3] = rotations[i]
        rot_inv[:3,:3] = np.transpose(rotations[i])
        #rotated_head = np.matmul(rot,poses_inv[i])
        camera_matrix = np.matmul(rigid_poses[0], rot_inv)
        camera_matrix = np.matmul(rot_inv,rigid_poses[0])
        out_poses[i] = camera_matrix

    expressions = expressions_orig
    return  expressions[:N_out], out_poses


def custom_seq_driving(rigid_poses_driving,rigid_poses_target,expressions_driving, expressions_target):
    N_same_start = 50
    N_ellipse = 100
    N_same_at_pose = 150
    acc = 0
    from scipy.spatial.transform import Rotation as R

    # Get angles of Driving sequence
    xyz_angles = np.zeros((len(rigid_poses_driving),3))
    poses_inv = []
    for i,pose in enumerate(rigid_poses_driving):
        pose_inv = np.linalg.inv(pose)
        poses_inv.append(pose_inv)
        r = R.from_matrix(pose_inv[:3,:3]) # move to euler angles. fix cam rotating head
        xyz_angles[i] = r.as_euler('xyz', degrees=True)

    # Find most frontal pose in target vid
    # Get angles of target sequence
    xyz_angles_target = np.zeros((len(rigid_poses_target),3))
    poses_target_inv = []
    for i,pose in enumerate(rigid_poses_target):
        pose_inv = np.linalg.inv(pose)
        poses_inv.append(pose_inv)
        r = R.from_matrix(pose_inv[:3,:3]) # move to euler angles. fix cam rotating head
        xyz_angles_target[i] = r.as_euler('xyz', degrees=True)

    xyz_angles_target[:,0]*=0.5 # make up/down less important in finding fronatl pose
    angles_norm = np.sum(np.abs(xyz_angles_target) ** 2, axis=-1) ** (1. / 2)
    index_frontal_pose = np.argmin(angles_norm)
    print("frontal most pose of target sequence is frame %d" % index_frontal_pose)


    # get range of angles
    x_min, x_max = np.min(xyz_angles[:,0]), np.max(xyz_angles[:,0])
    y_min, y_max = np.min(xyz_angles[:,1]), np.max(xyz_angles[:,1])
    z_min, z_max = np.min(xyz_angles[:,2]), np.max(xyz_angles[:,2])

    # choose some random angles and interpolate between them
    N_waypoints = 4
    out_angles=[]
    # for i in range(N_waypoints+1):
    #     start = np.array([x[i], y[i], z[i]])
    #     end = np.array([x[i + 1], y[i + 1], z[i + 1]])
    #     # interpolate to next set of angles
    #     out_angles.append(np.linspace(start,end , 60))
    #     # Stay for a while
    #     out_angles.append(np.repeat(end[np.newaxis,:], 100, axis=0))
    # # stack them all up
    # out_angles = np.concatenate(out_angles,axis=0)

    #out_angles = xyz_angles[-1000:] # use angles from driving
    out_angles = xyz_angles[:] # use angles from driving
    N_out = out_angles.shape[0]
    print("number of angles: ", N_out)

    print("angles range in video: X: [%f,%f], Y: [%f,%f]: Z: [%f,%f]" % (x_min, x_max, y_min, y_max, z_min, z_max))

    # invert back to matrices and to moving camera fixed head
    r_out = R.from_euler('xyz', out_angles, degrees=True)
    rotations = r_out.as_matrix() # this assumes static camera moving head, need to invertmv

    out_poses = np.zeros((N_out, 4, 4))
    #out_poses[:,-1,-1] = 1
    #rot = np.eye(4)
    rot_inv = np.eye(4)
    # P is camera pose with head at 0. Then R is rotation to head.
    # In turning head space, R * P_inv is new head pose
    # Back to fixed head space, camera pose is P * R_inv = P * R_t
    for i in range(N_out):
        #rot[:3,:3] = rotations[i]
        rot_inv[:3,:3] = np.transpose(rotations[i])
        #rotated_head = np.matmul(rot,poses_inv[i])
        #camera_matrix = np.matmul(rigid_poses[0], rot_inv)
        camera_matrix = np.matmul(rot_inv,rigid_poses_target[index_frontal_pose])
        out_poses[i] = camera_matrix

    transfer_delta_expressions_from_netural = True
    if transfer_delta_expressions_from_netural:
        # yawar 1278
        # barbara 6050
        # Pablo2 6560
        neutral_driving = expressions_driving[6560]

        #neutral_driving = expressions_driving[6050]
        #neutral_driving = expressions_driving[1278]
        # manuel 25
        #neutral_driving = expressions_driving[25]
        #NORMAN: (also 6500 previously 4715)
        neutral_target = expressions_target[6500]
        #Dave:
        #neutral_target = expressions_target[4718]
        # dejan 3180
        #neutral_target = expressions_target[4718]
        # ji 5937
        #neutral_target = expressions_target[3180]
        delta_to_transfer = expressions_driving[-N_out:] - np.tile(neutral_driving,(N_out,1))
        expressions_out = np.tile(neutral_target, (N_out,1)) + delta_to_transfer

        #expressions_out = 0.5*expressions_out + 0.5*expressions_driving[-N_out:]
    else:
        expressions_out = expressions_driving[-N_out:]

        #expressions_out = n

    return  expressions_out, out_poses


def custom_seq_xyz(rigid_poses, expressions_orig):
    N_same_start = 50
    N_ellipse = 100
    N_same_at_pose = 150
    acc = 0
    from scipy.spatial.transform import Rotation as R
    xyz_angles = np.zeros((len(rigid_poses),3))
    poses_inv = []
    for i,pose in enumerate(rigid_poses):
        pose_inv = np.linalg.inv(pose)
        poses_inv.append(pose_inv)
        r = R.from_matrix(pose_inv[:3,:3]) # move to euler angles. fix cam rotating head
        xyz_angles[i] = r.as_euler('xyz', degrees=True)

    # get range of angles
    x_min, x_max = np.min(xyz_angles[:,0]), np.max(xyz_angles[:,0])
    y_min, y_max = np.min(xyz_angles[:,1]), np.max(xyz_angles[:,1])
    z_min, z_max = np.min(xyz_angles[:,2]), np.max(xyz_angles[:,2])

    # choose some random angles and interpolate between them
    x = [0,x_min*0.6,x_max*0.6,0,
         0,0,0,0]

    y = [0,0,0,0,
         y_max*0.4,0,y_min*0.4,0]

    z = np.zeros(8)

    # Abdrei: TEST_SEQ_START = 5506
    #neutral_expression = expressions_orig[TEST_SEQ_START + 987]
    #neutral_expression = expressions_orig[TEST_SEQ_START + 987]


    #Norman:
    TEST_SEQ_START = 5509

    neutral_expression = expressions_orig[TEST_SEQ_START+979]
    neutral_expression[68] -= 0.3

    smile_expression = (0.2*expressions_orig[TEST_SEQ_START+460] + 0.8*expressions_orig[5450])
    #smile_expression = expressions_orig[5450]
    open_mouth_expression = np.copy(neutral_expression)
    open_mouth_expression[68] = 0.5
    open_mouth_expression[12] = 0.4

    #expressions = expressions_orig[[979,979,979,979,5450,5450,5450,5450],:]
    #expressions = expressions_orig[[979,979,979,5680,5680,
    #                                5450,5450,5450,5680,5450],:]
    #expressions = expressions_orig[[979,979,979,2359,2359,2359],:]
    #out_poses = rigid_poses[[6308, 5450, 6338, 5644, 6129,
    #                         6308, 5450, 6338,5644, 6129], ...]
    #N_out = out_poses.shape[0]

    expression_waypoints = []
    #expression_waypoints.append(expressions_orig)
    # expression_waypoints.append(neutral_expression)
    # expression_waypoints.append(expressions_orig[TEST_SEQ_START+309])
    # expression_waypoints.append(neutral_expression)
    # expression_waypoints.append(expressions_orig[TEST_SEQ_START+630])
    # expression_waypoints.append(neutral_expression)
    # expression_waypoints.append(expressions_orig[TEST_SEQ_START+708])
    # expression_waypoints.append(neutral_expression)
    # expression_waypoints.append(expressions_orig[TEST_SEQ_START+935])
    # expression_waypoints.append(neutral_expression)
    expression_waypoints.append(neutral_expression)
    expression_waypoints.append(smile_expression)
    expression_waypoints.append(open_mouth_expression)
    expression_waypoints.append(smile_expression)
    expression_waypoints.append(neutral_expression)
    expression_waypoints.append(open_mouth_expression)
    expression_waypoints.append(smile_expression)
    #expression_waypoints.append(open_mouth_expression)
    expression_waypoints.append(neutral_expression)


    #expression_waypoints.append(expressions_orig[TEST_SEQ_START+708])
    # expression_waypoints.append(neutral_expression)
    # expression_waypoints.append(expressions_orig[TEST_SEQ_START+935])
    # expression_waypoints.append(neutral_expression)
    expressions_out = []
    for i in range(len(expression_waypoints)-1):
        expressions_out.append(np.linspace(expression_waypoints[i],expression_waypoints[i+1],15))
        #expressions_out.append(np.repeat(expression_waypoints[i+1][np.newaxis,:], 15, axis=0))


    N_waypoints=len(x)
    out_angles=[]
    # show rigid NVS
    for i in range(N_waypoints-1):
        start = np.array([x[i], y[i], z[i]])
        end = np.array([x[i + 1], y[i + 1], z[i + 1]])
        # interpolate to next set of angles
        out_angles.append(np.linspace(start,end , 15))
        # Stay for a while
        #out_angles.append(np.repeat(end[np.newaxis,:], 100, axis=0))
    # stack them all up


    out_angles = np.concatenate(out_angles,axis=0)



    N_out = out_angles.shape[0]


    print("number of angles: ", N_out)

    print("angles range in video: X: [%f,%f], Y: [%f,%f]: Z: [%f,%f]" % (x_min, x_max, y_min, y_max, z_min, z_max))

    # invert back to matrices and to moving camera fixed head
    r_out = R.from_euler('xyz', out_angles, degrees=True)
    rotations = r_out.as_matrix() # this assumes static camera moving head, need to invert

    # Dave
    #expressions_to_render = [5956,6261,6481, 5805]
    #expressions_to_render = [0,0,0, 0]
    #n_different_expr = len(expressions_to_render)

    #out_poses = np.zeros((N_out*n_different_expr, 4, 4))
    out_poses = np.zeros((N_out,4,4))
    #out_poses[:,-1,-1] = 1
    #rot = np.eye(4)
    # P is camera pose with head at 0. Then R is rotation to head.
    # In turning head space, R * P_inv is new head pose
    # Back to fixed head space, camera pose is P * R_inv = P * R_t
    rot_inv = np.eye(4)
    for i in range(N_out):
        #rot[:3,:3] = rotations[i]
        rot_inv[:3,:3] = np.transpose(rotations[i])
        #rotated_head = np.matmul(rot,poses_inv[i])
        camera_matrix = np.matmul(rigid_poses[0], rot_inv)
        camera_matrix = np.matmul(rot_inv,rigid_poses[0])
        #for j in range(n_different_expr):
        out_poses[i]=camera_matrix

    #out_poses = np.concatenate(out_poses,0)
    expressions_out = np.concatenate(expressions_out,0)
    #expressions_out = np.concatenate((expressions_out,expressions_orig[TEST_SEQ_START+641:987]),0)

    n_play_with_expressions = expressions_out.shape[0]
    n_play_with_view = out_poses.shape[0]

    #out_poses = np.concatenate((out_poses, np.tile(out_poses[-1],(n_play_with_expressions,1,1)) ), axis=0)


    #expressions_out = np.concatenate(expressions_out,0)
    #expressions_out = np.concatenate((np.tile(expressions_out[0], (n_play_with_view, 1)), expressions_out), axis=0)
    #indices = expressions_to_render * N_out
    #expressions_out = expressions_orig[indices,:]
    #expressions_out = np.tile(smile_expression,(n_play_with_view,1))
    # use neutral exp
    #expressions_out = np.repeat(expressions_orig[111][np.newaxis,:], out_poses.shape[0], axis=0)
    #out_poses = np.concatenate((out_poses, np.tile(out_poses[-1],(n_play_with_expressions,1,1)) ), axis=0)
    out_poses = np.tile(out_poses[0], (n_play_with_expressions,1,1))
    return  expressions_out, out_poses


def custom_seq_open_mouth(rigid_poses, expressions_orig):
    N_same_start = 50
    N_ellipse = 100
    N_same_at_pose = 150
    acc = 0
    from scipy.spatial.transform import Rotation as R
    xyz_angles = np.zeros((len(rigid_poses),3))
    poses_inv = []
    for i,pose in enumerate(rigid_poses):
        pose_inv = np.linalg.inv(pose)
        poses_inv.append(pose_inv)
        r = R.from_matrix(pose_inv[:3,:3]) # move to euler angles. fix cam rotating head
        xyz_angles[i] = r.as_euler('xyz', degrees=True)

    # get range of angles
    x_min, x_max = np.min(xyz_angles[:,0]), np.max(xyz_angles[:,0])
    y_min, y_max = np.min(xyz_angles[:,1]), np.max(xyz_angles[:,1])
    z_min, z_max = np.min(xyz_angles[:,2]), np.max(xyz_angles[:,2])

    # choose some random angles and interpolate between them
    x = [0,x_min*0.7,x_max*0.7,0,
         0,0,0,0]

    y = [0,0,0,0,
         y_max*0.5,0,y_min*0.5,0]

    z = np.zeros(8)

    TEST_SEQ_START = 5506
    neutral_expression = expressions_orig[TEST_SEQ_START + 987]
    #neutral_expression[[12,13]] *= 0.5
    open_mouth_expression = np.copy(neutral_expression)
    open_mouth_expression[68] = 0.4

    closed_mouth_expression = np.copy(neutral_expression)
    closed_mouth_expression[68] = -0.5

    right_mouth_expression = np.copy(neutral_expression)
    right_mouth_expression[12] = 0.4
    right_mouth_expression[13] = -0.1

    left_mouth_expression = np.copy(neutral_expression)
    left_mouth_expression[12] = -0.4
    left_mouth_expression[13] = 0.4

    smile_mouth_expression = np.copy(neutral_expression)
    smile_mouth_expression[14] = 0.4
    smile_mouth_expression[68] = 0.4
    #smile_mouth_expression[12] = 0.4
    #smile_mouth_expression[12] = 0.4


    expression_waypoints = []
    expression_waypoints.append(neutral_expression)
    expression_waypoints.append(open_mouth_expression)
    expression_waypoints.append(closed_mouth_expression)
    expression_waypoints.append(neutral_expression)
    expression_waypoints.append(smile_mouth_expression)
    expression_waypoints.append(closed_mouth_expression)
    #expression_waypoints.append(neutral_expression)
    expressions_out = []

    for i in range(len(expression_waypoints)-1):
        expressions_out.append(np.linspace(expression_waypoints[i],expression_waypoints[i+1],15))
        #expressions_out.append(np.repeat(expression_waypoints[i+1][np.newaxis,:], 1, axis=0))


    N_waypoints=len(x)
    out_angles=[]
    # show rigid NVS
    for i in range(N_waypoints-1):
        start = np.array([x[i], y[i], z[i]])
        end = np.array([x[i + 1], y[i + 1], z[i + 1]])
        # interpolate to next set of angles
        out_angles.append(np.linspace(start,end , 1))
        # Stay for a while
        #out_angles.append(np.repeat(end[np.newaxis,:], 100, axis=0))
    # stack them all up


    out_angles = np.concatenate(out_angles,axis=0)



    N_out = out_angles.shape[0]


    print("number of angles: ", N_out)

    print("angles range in video: X: [%f,%f], Y: [%f,%f]: Z: [%f,%f]" % (x_min, x_max, y_min, y_max, z_min, z_max))

    # invert back to matrices and to moving camera fixed head
    r_out = R.from_euler('xyz', out_angles, degrees=True)
    rotations = r_out.as_matrix() # this assumes static camera moving head, need to invert

    # Dave
    #expressions_to_render = [5956,6261,6481, 5805]
    #expressions_to_render = [0,0,0, 0]
    #n_different_expr = len(expressions_to_render)

    #out_poses = np.zeros((N_out*n_different_expr, 4, 4))
    out_poses = np.zeros((N_out,4,4))
    #out_poses[:,-1,-1] = 1
    #rot = np.eye(4)
    # P is camera pose with head at 0. Then R is rotation to head.
    # In turning head space, R * P_inv is new head pose
    # Back to fixed head space, camera pose is P * R_inv = P * R_t
    rot_inv = np.eye(4)
    for i in range(N_out):
        #rot[:3,:3] = rotations[i]
        rot_inv[:3,:3] = np.transpose(rotations[i])
        #rotated_head = np.matmul(rot,poses_inv[i])
        camera_matrix = np.matmul(rigid_poses[0], rot_inv)
        camera_matrix = np.matmul(rot_inv,rigid_poses[0])
        #for j in range(n_different_expr):
        out_poses[i]=camera_matrix


    #out_poses = np.concatenate(out_poses,0)
    expressions_out = np.concatenate(expressions_out,0)
    #expressions_out = np.concatenate((expressions_out,expressions_orig[TEST_SEQ_START+641:987]),0)

    n_play_with_expressions = expressions_out.shape[0]
    n_play_with_view = out_poses.shape[0]

    #out_poses = np.concatenate((out_poses, np.tile(out_poses[-1],(n_play_with_expressions,1,1)) ), axis=0)
    out_poses = np.tile(out_poses[0],(n_play_with_expressions,1,1))

    #expressions_out = np.concatenate(expressions_out,0)
    #expressions_out
    #expressions_out = np.concatenate((np.tile(expressions_out[0], (n_play_with_view, 1)), expressions_out), axis=0)
    #indices = expressions_to_render * N_out
    #xpressions_out = expressions_orig[indices,:]
    # use neutral exp
    #expressions_out = np.repeat(expressions_orig[111][np.newaxis,:], out_poses.shape[0], axis=0)


    return  expressions_out, out_poses


def custom_seq_open_mouth_xyz(rigid_poses, expressions_orig):
    N_same_start = 50
    N_ellipse = 100
    N_same_at_pose = 150
    acc = 0
    from scipy.spatial.transform import Rotation as R
    xyz_angles = np.zeros((len(rigid_poses),3))
    poses_inv = []
    for i,pose in enumerate(rigid_poses):
        pose_inv = np.linalg.inv(pose)
        poses_inv.append(pose_inv)
        r = R.from_matrix(pose_inv[:3,:3]) # move to euler angles. fix cam rotating head
        xyz_angles[i] = r.as_euler('xyz', degrees=True)

    # get range of angles
    x_min, x_max = np.min(xyz_angles[:,0]), np.max(xyz_angles[:,0])
    y_min, y_max = np.min(xyz_angles[:,1]), np.max(xyz_angles[:,1])
    z_min, z_max = np.min(xyz_angles[:,2]), np.max(xyz_angles[:,2])

    # choose some random angles and interpolate between them
    x = [0,x_max*0.4,x_min*0.3,x_max*0.4,x_max*0.3,0]

    y = [0,y_min*0.4,y_max*0.1,y_max*0.40, y_min*0.1,0]

    z = [0, 0, 0, 0,0,0]


    x = [0,x_max*0.4,x_min*0.4,0,
         0,0,0,0]

    y = [0,0,0,0,
         y_max*0.4,0,y_min*0.4,0]

    z = [0, 0,0, 0,0,0,0,0]
    # DAVE:
    TEST_SEQ_START = 5506
    # Andrei:
    #TEST_SEQ_START = 0
    neutral_expression = expressions_orig[TEST_SEQ_START + 987]
    #neutral_expression[[12,13]] *= 0.5
    open_mouth_expression = np.copy(neutral_expression)
    open_mouth_expression[68] = 0.4

    closed_mouth_expression = np.copy(neutral_expression)
    closed_mouth_expression[68] = -0.5

    right_mouth_expression = np.copy(neutral_expression)
    right_mouth_expression[12] = 0.4
    right_mouth_expression[13] = -0.1

    left_mouth_expression = np.copy(neutral_expression)
    left_mouth_expression[12] = -0.4
    left_mouth_expression[13] = 0.4

    smile_mouth_expression = np.copy(neutral_expression)
    smile_mouth_expression[14] = 0.4
    smile_mouth_expression[68] = 0.4
    #smile_mouth_expression[12] = 0.4
    #smile_mouth_expression[12] = 0.4


    expression_waypoints = []
    expression_waypoints.append(neutral_expression)
    expression_waypoints.append(open_mouth_expression)
    expression_waypoints.append(closed_mouth_expression)
    expression_waypoints.append(neutral_expression)
    expression_waypoints.append(smile_mouth_expression)
    expression_waypoints.append(closed_mouth_expression)
    #expression_waypoints.append(neutral_expression)
    expressions_out = []

    for i in range(len(expression_waypoints)-1):
        expressions_out.append(np.linspace(expression_waypoints[i],expression_waypoints[i+1],15))
        #expressions_out.append(np.repeat(expression_waypoints[i+1][np.newaxis,:], 1, axis=0))


    N_waypoints=len(x)
    out_angles=[]
    # show rigid NVS
    for i in range(N_waypoints-1):
        start = np.array([x[i], y[i], z[i]])
        end = np.array([x[i + 1], y[i + 1], z[i + 1]])
        # interpolate to next set of angles
        out_angles.append(np.linspace(start,end , 15))
        # Stay for a while
        #out_angles.append(np.repeat(end[np.newaxis,:], 100, axis=0))
    # stack them all up


    out_angles = np.concatenate(out_angles,axis=0)



    N_out = out_angles.shape[0]


    print("number of angles: ", N_out)

    print("angles range in video: X: [%f,%f], Y: [%f,%f]: Z: [%f,%f]" % (x_min, x_max, y_min, y_max, z_min, z_max))

    # invert back to matrices and to moving camera fixed head
    r_out = R.from_euler('xyz', out_angles, degrees=True)
    rotations = r_out.as_matrix() # this assumes static camera moving head, need to invert

    # Dave
    #expressions_to_render = [5956,6261,6481, 5805]
    #expressions_to_render = [0,0,0, 0]
    #n_different_expr = len(expressions_to_render)

    #out_poses = np.zeros((N_out*n_different_expr, 4, 4))
    out_poses = np.zeros((N_out,4,4))
    #out_poses[:,-1,-1] = 1
    #rot = np.eye(4)
    # P is camera pose with head at 0. Then R is rotation to head.
    # In turning head space, R * P_inv is new head pose
    # Back to fixed head space, camera pose is P * R_inv = P * R_t
    rot_inv = np.eye(4)
    for i in range(N_out):
        #rot[:3,:3] = rotations[i]
        rot_inv[:3,:3] = np.transpose(rotations[i])
        #rotated_head = np.matmul(rot,poses_inv[i])
        camera_matrix = np.matmul(rigid_poses[0], rot_inv)
        camera_matrix = np.matmul(rot_inv,rigid_poses[5506+987])
        #for j in range(n_different_expr):
        out_poses[i]=camera_matrix


    #out_poses = np.concatenate(out_poses,0)
    expressions_out = np.concatenate(expressions_out,0)
    #expressions_out = np.concatenate((expressions_out,expressions_orig[TEST_SEQ_START+641:987]),0)

    #n_play_with_expressions = expressions_out.shape[0]
    n_play_with_view = out_poses.shape[0]

    #out_poses = np.concatenate((out_poses, np.tile(out_poses[-1],(n_play_with_expressions,1,1)) ), axis=0)
    #out_poses = np.tile(out_poses[0],(n_play_with_expressions,1,1))

    #expressions_out = np.concatenate(expressions_out,0)
    #expressions_out
    expressions_out = np.concatenate((np.tile(expressions_out[0], (n_play_with_view, 1)), expressions_out), axis=0)
    #indices = expressions_to_render * N_out
    #xpressions_out = expressions_orig[indices,:]
    # use neutral exp
    #expressions_out = np.repeat(expressions_orig[111][np.newaxis,:], out_poses.shape[0], axis=0)
    # expressions_out = expressions_orig[5317+190:5317+320]
    # out_poses = np.tile(rigid_poses[900], (130,1,1))
    # out_poses[:,0, -1] = -0.04
    # out_poses[:,1, -1] = 0.08
    # out_poses[:,2, -1] = 0.42
    #out_poses = rigid_poses[5317+190:5317+320]

    return  expressions_out, out_poses


def custom_seq_teaser(rigid_poses, expressions_orig):
    N_same_start = 50
    N_ellipse = 100
    N_same_at_pose = 150
    acc = 0
    from scipy.spatial.transform import Rotation as R
    xyz_angles = np.zeros((len(rigid_poses),3))
    poses_inv = []
    for i,pose in enumerate(rigid_poses):
        pose_inv = np.linalg.inv(pose)
        poses_inv.append(pose_inv)
        r = R.from_matrix(pose_inv[:3,:3]) # move to euler angles. fix cam rotating head
        xyz_angles[i] = r.as_euler('xyz', degrees=True)

    # get range of angles
    x_min, x_max = np.min(xyz_angles[:,0]), np.max(xyz_angles[:,0])
    y_min, y_max = np.min(xyz_angles[:,1]), np.max(xyz_angles[:,1])
    z_min, z_max = np.min(xyz_angles[:,2]), np.max(xyz_angles[:,2])

    # choose some random angles and interpolate between them
    N_waypoints = 4
    x=np.hstack(( np.array(xyz_angles[0,0]), np.array(xyz_angles[0,0]), np.random.uniform(x_min, x_max, N_waypoints)))
    y=np.hstack(( np.array(xyz_angles[0,1]), np.array(xyz_angles[0,1]), np.random.uniform(y_min, y_max, N_waypoints)))
    z=np.hstack(( np.array(xyz_angles[0,2]), np.array(xyz_angles[0,2]), np.random.uniform(z_min, z_max, N_waypoints)))



    N_waypoints = 6
    x = np.array((x_min*0.5, 0, x_max*0.5))
    x = np.hstack((x,x))
    y = np.array( (y_max*0.7, 0, y_min*0.7))
    y = np.hstack((y, y))
    z = np.array( (z_min*0.1, 0, z_max*0.1) )
    z = np.hstack((z, z))

    #out_angles=[]
    out_angles = np.vstack((x,y,z)).transpose()
    N_out = out_angles.shape[0]
    print("number of angles: ", N_out)

    print("angles range in video: X: [%f,%f], Y: [%f,%f]: Z: [%f,%f]" % (x_min, x_max, y_min, y_max, z_min, z_max))

    # invert back to matrices and to moving camera fixed head
    r_out = R.from_euler('xyz', out_angles, degrees=True)
    rotations = r_out.as_matrix() # this assumes static camera moving head, need to invert

    out_poses = np.zeros((N_out, 4, 4))
    #out_poses[:,-1,-1] = 1
    #rot = np.eye(4)
    rot_inv = np.eye(4)
    # P is camera pose with head at 0. Then R is rotation to head.
    # In turning head space, R * P_inv is new head pose
    # Back to fixed head space, camera pose is P * R_inv = P * R_t
    for i in range(N_out):
        #rot[:3,:3] = rotations[i]
        rot_inv[:3,:3] = np.transpose(rotations[i])
        #rotated_head = np.matmul(rot,poses_inv[i])
        camera_matrix = np.matmul(rigid_poses[0], rot_inv)
        camera_matrix = np.matmul(rot_inv,rigid_poses[0])
        out_poses[i] = camera_matrix
    #Norman
    expressions = expressions_orig[[979,979,979,979,5450,5450,5450,5450],:]
    expressions = expressions_orig[[979,979,979,5680,5680,
                                    5450,5450,5450,5680,5450],:]
    #expressions = expressions_orig[[979,979,979,2359,2359,2359],:]
    out_poses = rigid_poses[[6308, 5450, 6338, 5644, 6129,
                             6308, 5450, 6338,5644, 6129], ...]
    N_out = out_poses.shape[0]

    #expressions[4] = (expressions[2359]+expressions[5450])/2
    # #Dave
    # expressions = expressions_orig[[5506+382,5506+382,5506+382,5506+980,5506+980,5506+980],:]
    # out_poses = rigid_poses[[5506+135, 5506+980, 5506+641, 5506+135, 5506+980, 5506+641], ...]



    intrinsics = np.array([-1991.4496  ,  2074.81344   , 258.46272 ,   257.676288])
    visualize(render_debug_camera_matrix(out_poses[0],intrinsics,0.33))
    visualize(render_debug_camera_matrix(out_poses[1],intrinsics,0.33))
    visualize(render_debug_camera_matrix(out_poses[2],intrinsics,0.33))
    visualize(render_debug_camera_matrix(out_poses[3],intrinsics,0.33))

    #out_poses= rigid_poses[[6308,6129,6338,6308,6129,6338],...]
    return  expressions[:N_out], out_poses

def generate_driven_test_sequence(args, N_max):
    "generating original sequence as 'test':"
    im_path_list, N, im_size = read_img_folder(args.source + "/images")
    if N_max:
        N = min(N,N_max)

    intrinsics = read_intrinsics(os.path.join(args.source, "intrinsics.txt"), im_size)
    expressions_driving = read_expressions(os.path.join(args.driving, "expression.txt"))
    expressions_target = read_expressions(os.path.join(args.source, "expression.txt"))
    rigid_poses_target, scale = read_rigid_poses(os.path.join(args.source, "rigid.txt"))
    rigid_poses_driving, scale_driving = read_rigid_poses(os.path.join(args.driving, "rigid.txt"))
    neutral_pose = rigid_poses_target.mean(0)
    #_, rigid_poses = custom_sequence(neutral_pose)
    #_, rigid_poses = custom_sequence_circle(neutral_pose,-0.6,0.6,-0.2,0.2)
    #expressions, rigid_poses = custom_seq_presentation_v2(rigid_poses,expressions)#,-0.6,0.6,-0.2,0.2)
    #expressions, rigid_poses = custom_seq_teaser(rigid_poses,expressions)#,-0.6,0.6,-0.2,0.2)
    DVP_PARTITION = False
    expressions, rigid_poses = custom_seq_driving(rigid_poses_driving,rigid_poses_target,expressions_driving, expressions_target)#,-0.6,0.6,-0.2,0.2)
    #expressions = expressions[500:620]
    create_subfolders(args.target)
    N = rigid_poses.shape[0]
    if N_max:
        N = min(N,N_max)

    camera_angle = 2 * np.arctan(im_size[0] / (2 * (intrinsics[0] )))
    mode = 'test'
    frames_data_dump = []
    idxs=range(N)
    print("Processing %d %s data..." % (len(idxs), mode))
    # for i, idx in enumerate(tqdm(idxs)) :
    #     filename = im_path_list[idx]
    #     #frames_data_dump.append({'file_path': './%s/%s' % (mode, filename.split(".")[0]),
    #     frames_data_dump.append({'file_path': './%s/%s' % (mode, 'f_%04d'%i),
    #                              #'transform_matrix': np.linalg.inv(rigid_poses[idx]).tolist(),
    #                              'transform_matrix': (rigid_poses[idx]).tolist(),
    #                              'expression': expressions[idx].tolist()
    #                              })
    #     real = True
    #     synth = not real
    #     if real:
    #         im = Image.open(os.path.join(args.source , 'images' , filename))
    #         # Resize:
    #         #im.thumbnail((300, 200), Image.ANTIALIAS)
    #         #im.save(args.target + ('/%s/%s' % (mode, filename.split(".")[0] + ".png")), "png")
    #         im.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")
    #     if synth:
    #         synth = render_debug_camera_matrix(rigid_poses[idx], intrinsics)
    #         synth = PIL.Image.fromarray(synth.astype(np.uint8))
    #         synth.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")

    for i, idx in enumerate(tqdm(idxs)) :
        filename = im_path_list[idx]
        #frames_data_dump.append({'file_path': './%s/%s' % (mode, filename.split(".")[0]),
        real = True
        save_synthetic = not real
        output_bbox = False
        bbox = np.array([0.0, 1.0, 0.0, 1.0])

        if real:
            im = Image.open(os.path.join(args.source , 'images' , filename))
            # Resize:
            #im.thumbnail((300, 200), Image.ANTIALIAS)
            #im.save(args.target + ('/%s/%s' % (mode, filename.split(".")[0] + ".png")), "png")
            im.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")
        # if synth:
        #     synth = render_debug_camera_matrix(rigid_poses[idx], intrinsics)
        #     synth = PIL.Image.fromarray(synth.astype(np.uint8))
        #     synth.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")
        if save_synthetic or output_bbox:
            synth = render_debug_camera_matrix(rigid_poses[idx], intrinsics, scale)

            if output_bbox:  # find bbox of head
                bbox = find_bbox(synth)

            if save_synthetic:
                synth = PIL.Image.fromarray(synth.astype(np.uint8))
                synth.save(args.target + ('/%s/%s' % (mode, 'f_%04d' % i + ".png")), "png")

        frames_data_dump.append({'file_path': './%s/%s' % (mode, 'f_%04d' % i),
                                 'bbox': bbox.tolist(),

                                 # 'transform_matrix': np.linalg.inv(rigid_poses[idx]).tolist(),
                                 'transform_matrix': (rigid_poses[idx]).tolist(),
                                 'expression': expressions[idx].tolist()
                                 })

    intrinsics_to_output = np.copy(intrinsics)
    intrinsics_to_output[3] /= im_size[0] #fy
    intrinsics_to_output[2] /= im_size[1] #fx

    with open(os.path.join(args.target, ('transforms_%s.json' % mode)), 'w') as fp:
        json.dump({'camera_angle_x': camera_angle,
                   'frames': frames_data_dump,
                   'intrinsics': intrinsics_to_output.tolist()},
                  fp, indent=4, )
    print("Done.")



def generate_custom_test_sequence(args, N_max):
    "generating original sequence as 'test':"
    im_path_list, N, im_size = read_img_folder(args.source + "/images")
    if N_max:
        N = min(N,N_max)

    intrinsics = read_intrinsics(os.path.join(args.source, "intrinsics.txt"), im_size)
    expressions = read_expressions(os.path.join(args.source, "expression.txt"))
    rigid_poses, scale = read_rigid_poses(os.path.join(args.source, "rigid.txt"))
    neutral_pose = rigid_poses.mean(0)
    #_, rigid_poses = custom_sequence(neutral_pose)
    #_, rigid_poses = custom_sequence_circle(neutral_pose,-0.6,0.6,-0.2,0.2)
    #expressions, rigid_poses = custom_seq_presentation_v2(rigid_poses,expressions)#,-0.6,0.6,-0.2,0.2)
    #expressions, rigid_poses = custom_seq_teaser(rigid_poses,expressions)#,-0.6,0.6,-0.2,0.2)
    #expressions, rigid_poses = custom_seq_xyz(rigid_poses,expressions)#,-0.6,0.6,-0.2,0.2)
    #expressions, rigid_poses = custom_seq_geom(rigid_poses,expressions)#,-0.6,0.6,-0.2,0.2)
    expressions, rigid_poses = custom_seq_open_mouth_xyz(rigid_poses,expressions)#,-0.6,0.6,-0.2,0.2)
    #expressions = expressions[500:620]
    create_subfolders(args.target)
    N = rigid_poses.shape[0]
    if N_max:
        N = min(N,N_max)

    camera_angle = 2 * np.arctan(im_size[0] / (2 * (intrinsics[0] )))
    mode = 'test'
    frames_data_dump = []
    idxs=range(N)
    print("Processing %d %s data..." % (len(idxs), mode))
    # for i, idx in enumerate(tqdm(idxs)) :
    #     filename = im_path_list[idx]
    #     #frames_data_dump.append({'file_path': './%s/%s' % (mode, filename.split(".")[0]),
    #     frames_data_dump.append({'file_path': './%s/%s' % (mode, 'f_%04d'%i),
    #                              #'transform_matrix': np.linalg.inv(rigid_poses[idx]).tolist(),
    #                              'transform_matrix': (rigid_poses[idx]).tolist(),
    #                              'expression': expressions[idx].tolist()
    #                              })
    #     real = True
    #     synth = not real
    #     if real:
    #         im = Image.open(os.path.join(args.source , 'images' , filename))
    #         # Resize:
    #         #im.thumbnail((300, 200), Image.ANTIALIAS)
    #         #im.save(args.target + ('/%s/%s' % (mode, filename.split(".")[0] + ".png")), "png")
    #         im.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")
    #     if synth:
    #         synth = render_debug_camera_matrix(rigid_poses[idx], intrinsics)
    #         synth = PIL.Image.fromarray(synth.astype(np.uint8))
    #         synth.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")

    for i, idx in enumerate(tqdm(idxs)) :
        filename = im_path_list[idx]
        #frames_data_dump.append({'file_path': './%s/%s' % (mode, filename.split(".")[0]),
        real = True
        save_synthetic = not real
        output_bbox = False
        bbox = np.array([0.0, 1.0, 0.0, 1.0])

        if real:
            im = Image.open(os.path.join(args.source , 'images' , filename))
            # Resize:
            #im.thumbnail((300, 200), Image.ANTIALIAS)
            #im.save(args.target + ('/%s/%s' % (mode, filename.split(".")[0] + ".png")), "png")
            im.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")
        # if synth:
        #     synth = render_debug_camera_matrix(rigid_poses[idx], intrinsics)
        #     synth = PIL.Image.fromarray(synth.astype(np.uint8))
        #     synth.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")
        if save_synthetic or output_bbox:
            synth = render_debug_camera_matrix(rigid_poses[idx], intrinsics, scale)

            if output_bbox:  # find bbox of head
                bbox = find_bbox(synth)

            if save_synthetic:
                synth = PIL.Image.fromarray(synth.astype(np.uint8))
                synth.save(args.target + ('/%s/%s' % (mode, 'f_%04d' % i + ".png")), "png")

        frames_data_dump.append({'file_path': './%s/%s' % (mode, 'f_%04d' % i),
                                 'bbox': bbox.tolist(),

                                 # 'transform_matrix': np.linalg.inv(rigid_poses[idx]).tolist(),
                                 'transform_matrix': (rigid_poses[idx]).tolist(),
                                 'expression': expressions[idx].tolist()
                                 })

    intrinsics_to_output = np.copy(intrinsics)
    intrinsics_to_output[3] /= im_size[0] #fy
    intrinsics_to_output[2] /= im_size[1] #fx

    with open(os.path.join(args.target, ('transforms_%s.json' % mode)), 'w') as fp:
        json.dump({'camera_angle_x': camera_angle,
                   'frames': frames_data_dump,
                   'intrinsics': intrinsics_to_output.tolist()},
                  fp, indent=4, )
    print("Done.")

def generate_original_test_sequence(args, N_max = None):
    "generating original sequence as 'test':"
    im_path_list, N, im_size = read_img_folder(args.source + "/images")
    if N_max:
        N = min(N,N_max)
    intrinsics = read_intrinsics(os.path.join(args.source, "intrinsics.txt"), im_size)
    expressions = read_expressions(os.path.join(args.source, "expression.txt"))

    rigid_poses, scale = read_rigid_poses(os.path.join(args.source, "rigid.txt"))

    if DVP_PARTITION:
        expressions = expressions[-1000:]
        rigid_poses = rigid_poses[-1000:]
        im_path_list = im_path_list[-1000:]

    create_subfolders(args.target)
    camera_angle = 2 * np.arctan(im_size[0] / (2 * (intrinsics[0] )))
    mode = 'test'
    frames_data_dump = []
    idxs=range(N)
    print("Processing %d %s data..." % (len(idxs), mode))
    for i, idx in enumerate(tqdm(idxs)) :
        filename = im_path_list[idx]
        #frames_data_dump.append({'file_path': './%s/%s' % (mode, filename.split(".")[0]),
        real = True
        save_synthetic = not real
        output_bbox = False
        bbox = np.array([0.0, 1.0, 0.0, 1.0])

        if real:
            im = Image.open(os.path.join(args.source , 'images' , filename))
            # Resize:
            #im.thumbnail((300, 200), Image.ANTIALIAS)
            #im.save(args.target + ('/%s/%s' % (mode, filename.split(".")[0] + ".png")), "png")
            im.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")
        # if synth:
        #     synth = render_debug_camera_matrix(rigid_poses[idx], intrinsics)
        #     synth = PIL.Image.fromarray(synth.astype(np.uint8))
        #     synth.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")
        if save_synthetic or output_bbox:
            synth = render_debug_camera_matrix(rigid_poses[idx], intrinsics, scale)

            if output_bbox:  # find bbox of head
                bbox = find_bbox(synth)

            if save_synthetic:
                synth = PIL.Image.fromarray(synth.astype(np.uint8))
                synth.save(args.target + ('/%s/%s' % (mode, 'f_%04d' % i + ".png")), "png")
        frames_data_dump.append({'file_path': './%s/%s' % (mode, 'f_%04d'%i),
                                 'bbox': bbox.tolist(),

                                 #'transform_matrix': np.linalg.inv(rigid_poses[idx]).tolist(),
                                 'transform_matrix': (rigid_poses[idx]).tolist(),
                                 'expression': expressions[idx].tolist()
                                 })

    intrinsics_to_output = np.copy(intrinsics)
    intrinsics_to_output[3] /= im_size[0] #fy
    intrinsics_to_output[2] /= im_size[1] #fx

    with open(os.path.join(args.target, ('transforms_%s.json' % mode)), 'w') as fp:
        json.dump({'camera_angle_x': camera_angle,
                   'frames': frames_data_dump,
                   'intrinsics': intrinsics_to_output.tolist()},
                  fp, indent=4, )
    print("Done.")


def main(args):

    im_path_list, N, im_size = read_img_folder(args.source + "/images")
    intrinsics = read_intrinsics(os.path.join(args.source, "intrinsics.txt"), im_size)
    expressions = read_expressions(os.path.join(args.source, "expression.txt"))
    rigid_poses, scale = read_rigid_poses(os.path.join(args.source, "rigid.txt"))
    print("scaling = ", scale)

    if DVP_PARTITION:
        expressions = expressions[:-1000]
        rigid_poses = rigid_poses[:-1000]
        im_path_list = im_path_list[:-1000]
        N = len(im_path_list)

    if LESS_DATA > 0:  # 0.5 0.25 ratio of data to use
        # Get rid of test set
        expressions = expressions[:-1000]
        rigid_poses = rigid_poses[:-1000]
        im_path_list = im_path_list[:-1000]
        n_full = rigid_poses.shape[0]
        n_trim = int(LESS_DATA * n_full)
        expressions = expressions[:n_trim]
        rigid_poses = rigid_poses[:n_trim]
        im_path_list = im_path_list[:n_trim]
        N = len(im_path_list)
    create_subfolders(args.target)
    n_train = N-6
    n_val = 5
    n_test = 1
    indices = train_val_partition(N, n_train, n_val, n_test) # train val test indices
    camera_angle = 2 * np.arctan(im_size[0] / (2 * (intrinsics[0] )))
    # create a map Nx2 from image index to index in train set (for reconstructing the video with latent codes)
    map = -np.ones((N,2))
    map[:,0] = np.arange(N)
    for mode in indices.keys(): # Train Val Test
        if mode == 'test':
            continue
        idxs = indices[mode]
        frames_data_dump = []
        print("Processing %d %s data..." % (len(idxs), mode))
        for i, idx in enumerate(tqdm(idxs)) :
            filename = im_path_list[idx]
            #frames_data_dump.append({'file_path': './%s/%s' % (mode, filename.split(".")[0]),
            if mode == 'train': map[idx,1] = i
            real = True
            save_synthetic = not real
            output_bbox = True
            bbox = np.array([0.0, 1.0, 0.0, 1.0])
            if real:
                im = Image.open(os.path.join(args.source , 'images' , filename))
                # Resize:
                #im.thumbnail((300, 200), Image.ANTIALIAS)
                #im.save(args.target + ('/%s/%s' % (mode, filename.split(".")[0] + ".png")), "png")
                im.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")
            if save_synthetic or output_bbox:
                synth = render_debug_camera_matrix(rigid_poses[idx], intrinsics, scale)

                if output_bbox: # find bbox of head
                    bbox = find_bbox(synth)

                if save_synthetic:
                    synth = PIL.Image.fromarray(synth.astype(np.uint8))
                    synth.save(args.target + ('/%s/%s' % (mode, 'f_%04d'%i + ".png")), "png")

            frames_data_dump.append({'file_path': './%s/%s' % (mode, 'f_%04d' % i),
                                     # 'transform_matrix': np.linalg.inv(rigid_poses[idx]).tolist(),
                                     'bbox': bbox.tolist(),
                                     'transform_matrix': (rigid_poses[idx]).tolist(),
                                     'expression': expressions[idx].tolist()
                                     })

        intrinsics_to_output = np.copy(intrinsics)
        intrinsics_to_output[3] /= im_size[0] #fy
        intrinsics_to_output[2] /= im_size[1] #fx

        with open(os.path.join(args.target, ('transforms_%s.json' % mode)), 'w') as fp:
            json.dump({'camera_angle_x': camera_angle,
                       'frames': frames_data_dump,
                       'intrinsics': intrinsics_to_output.tolist()},
                      fp, indent=4, )
        np.save(os.path.join(args.target, 'index_map.npy'), map)
        print("Done.")

def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)

if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGUSR1, handle_pdb)
    print("Reading tracked face data")
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('109.125.110.118', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    import PIL.Image

    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", type=str, required=True, help="Path to (.yml) config file.")
    #parser.add_argument("--save-disparity-image", action="store_true", help="Save disparity images too."    )
    parser.add_argument("--source", type=str,  help="path to source dir. Render images of this person.",
                        default="/rdata/guygafni/projects/cnerf/nerf-pytorch/real_data/dave")
    parser.add_argument("--target", type=str,  help="where to save",
                        default="/rdata/guygafni/projects/cnerf/nerf-pytorch/real_data/dave_xyz_neww")
    parser.add_argument("--driving", type=str, help="path to source dir of person DRIVING (uses expressions and rotations)",
                        default="/rdata/guygafni/projects/cnerf/nerf-pytorch/real_data/yawar2")
    parser.add_argument("--LESS_DATA", type=float,  help="use % first frames of video for train. 0 is full train set, 0.5 is half",
                        default="0")

    DVP_PARTITION=True
    configargs = parser.parse_args()
    LESS_DATA = configargs.LESS_DATA
    os.makedirs(configargs.target, exist_ok=True)
    os.makedirs(os.path.join(configargs.target, 'debug_vis'), exist_ok=True)
    #main(configargs)
    #generate_original_test_sequence(configargs,1000)
    #generate_custom_test_sequence(configargs, 1000)
    generate_driven_test_sequence(configargs, 1000)
    visualize = True
    Normalize_scene = True
    if visualize:
        intrinsics = read_intrinsics(os.path.join(configargs.source, "intrinsics.txt"), (512,512))
        expressions = read_expressions(os.path.join(configargs.source, "expression.txt"))
        rigid_poses, scale = read_rigid_poses(os.path.join(configargs.source, "rigid.txt"), mean_scale=Normalize_scene)
        for i in trange(4000,5000,1):
        #for i in trange(0,rigid_poses.shape[0]):
            color = render_debug_camera_matrix(rigid_poses[i],intrinsics, scale)
            im_real = PIL.Image.open(os.path.join(configargs.source,'images','%05d.png'%i ))
            im_real = np.asarray(im_real)

            overlay = (2*im_real + 2*color)/3
            overlay = np.copy(im_real)
            idx= np.where(color<255)
            overlay[np.where(color<255)] = 0.8 * color[idx] + 0.2 * overlay[idx]
            overlay = PIL.Image.fromarray(overlay.astype(np.uint8))
            overlay.save((os.path.join(configargs.target, 'debug_vis') + '/r_%04d.png' %  i))
            show = False
            if show:
                plt.figure()
                plt.imshow(overlay)
                plt.figure()
                plt.show()
