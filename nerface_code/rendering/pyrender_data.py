# Render offscreen -- make sure to set the PyOpenGL platform
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np
import trimesh
import pyrender
import torch
from options import Options
from tqdm import tqdm
import PIL.Image
import numpy.random
import json
from spherical_sampler import SphericalSampler

def blender_to_GL(input):
    y = input[1]
    z = input[2]
    return np.array([input[0],z,-y])

def GL_to_blender(input):
    x = input[0]
    y = input[1]
    z = input[2]
    return np.array([+x,y,z])

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

def create_cam_matrices_from_position(posistions, look_at=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    # list version of LookAt

    camera_poses = []
    for position in posistions:
        camera_poses.append(lookAt(position), look_at, up)
    # view_sampler = SphericalSampler(opt.n_views_test, 'SPIRAL')
    # all_viewpoints['test'] = view_sampler.points
    # mp.save(os.path.join(opt.folder_name, 'poses_test.npy'), view_sampler.points)
    return camera_poses


class Renderer:

    def __init__(self, opt):

        self.name = opt.name
        self.target_name = opt.target_name
        self.folder_name = opt.folder_name
        self.n_views = opt.n_views
        #self.n_views_train = 10
        self.n_views_train = opt.n_views_train
        self.n_views_test = opt.n_views_test
        self.n_points = opt.train
        self.render = opt.render
        self.render = opt.render
        self.im_size = opt.im_size
        self.anti_alias = opt.anti_alias
        self.all_viewpoints = {}
        self.all_viewpoints['train'] = []
        self.all_viewpoints['test'] = []
        #self.light_positions

        self.fx = 300 * self.anti_alias
        self.fy = 300 * self.anti_alias
        self.width = 256
        self.height = 256
        self.cx = self.width*self.anti_alias/2
        self.cy = self.width*self.anti_alias/2
        self.camera_angle = 2 * np.arctan(self.width/(2*(self.fx/self.anti_alias)))

        self.target_path = '../nerf-pytorch/cache/%s/' % (self.target_name).lower()
        if self.folder_name is '':
            self.folder_name = self.name

        create_subfolders(self.target_path)

        #extrinsics = np.load('./renders/teapot_normal_dense_all_views.npy')
        #intrinsics = np.load('./renders/teapot_normal_dense_all_views_intrinsics.npy')

    def set_train_views(self, train_views):
        for view in train_views:
            view[0][:, 2] *= -1
            self.all_viewpoints['train'].append(view[0])

    def set_test_views(self, test_views):
        for view in test_views:
            view[0][:, 2] *= -1
            self.all_viewpoints['test'].append(view[0])

    def set_intrinsics(self, matrix):
        self.fx = matrix[0][0] * opt.anti_alias
        self.fy = -matrix[1][1] * opt.anti_alias
        self.cx = matrix[0][2] * opt.anti_alias
        self.cy = matrix[1][2] * opt.anti_alias

    def process_mesh(self):

        mesh_path = os.path.join(self.folder_name, self.name + '.ply')

        print('Attempting to load mesh: %s ...' % self.name)
        if not os.path.exists(mesh_path):
            print("ERROR: mesh file does not exist: %s" % mesh_path)


        # Load the mesh and put it in a scene
        #input_trimesh = trimesh.load('./example_data/pointclouds/teapot_normal_dense.obj')
        input_trimesh = trimesh.load_mesh(mesh_path)
        trimesh_scene = input_trimesh.scene()

        input_trimesh.vertices -= trimesh_scene.centroid
        volume = input_trimesh.bounding_sphere.volume
        radius = np.power((3*volume/(4*np.pi)),1/3)

        #input_trimesh.vertices *= (1/(4*radius))
        input_trimesh.apply_scale(1/(1.2*input_trimesh.scale))

        print("Mesh has %d vertices" % input_trimesh.vertices.shape[0] )
        #scene = fuze_trimesh.scene()
        self.mesh = pyrender.Mesh.from_trimesh(input_trimesh)
        self.scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        #scene = pyrender.Scene.from_trimesh_scene(fuze_trimesh)
        self.scene.add(self.mesh)
        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up

                # Add lights
        light = pyrender.SpotLight(color=np.ones(3), intensity=24.0,
                                   innerConeAngle=np.pi / 16.0)
        light_node = self.scene.add(light, pose= lookAt(np.array([2,2,2]), [0,0,0]))
        light_node = self.scene.add(light, pose= lookAt(np.array([2,6,3]), [0,0,0]))
        light_node = self.scene.add(light, pose= lookAt(np.array([2,-1,-3]), [0,0,0]))
        light_node = self.scene.add(light, pose= lookAt(np.array([-4,4,-3]), [0,0,0]))
        light_node = self.scene.add(light, pose= lookAt(np.array([-2,-2,-3]), [0,0,0]))

    def get_viewpoints(self):
        self.all_viewpoints={}
        view_sampler = SphericalSampler(self.n_views, 'RANDOM')
        np.random.shuffle(view_sampler.points)
        #points = random.shuffle(view_sampler.points)
        self.all_viewpoints['train'] = view_sampler.points[0:int(0.6*self.n_views),:]
        self.all_viewpoints['val']= view_sampler.points[int(0.6*self.n_views):int(0.8*self.n_views),:]
        self.all_viewpoints['test']= view_sampler.points[int(0.8*self.n_views):,:]

    def get_test_sequence(self):
        test_view_sampler = SphericalSampler(self.n_views_test, 'HELIX')
        self.all_viewpoints['test'] = test_view_sampler.points

    def render_images(self, mode):

        if self.all_viewpoints[mode] is None:
            print("Error: list of points to render %s is None" % mode)

        r = pyrender.OffscreenRenderer(self.width*self.anti_alias, self.height*self.anti_alias)

        if mode is 'train':
            N = min(len(self.all_viewpoints['train']), self.n_views_train)
        elif mode is 'test':
            N = min(len(self.all_viewpoints['test']), self.n_views_test)
        elif mode is 'val':
            N = min(len(self.all_viewpoints['val']), self.n_views_train)

        print("Rendering %s from %d views ..." % (mode,N))
        frames_data_dump = []
        camera = pyrender.IntrinsicsCamera(self.fx, self.fx, self.cx,
                                           self.cy, 0.0001, 1000)
        camera_node = self.scene.add(camera, pose=np.eye(4))
        #camera_node = self.scene.add(camera, self.all_viewpoints['train'][1][0])

        for i in tqdm(range(N)):
            #camera_pose = lookAt(np.array([-9.0,-4.0,13]),np.array([0,0,0]))

            camera_position = self.all_viewpoints[mode][i]

            camera_pose = lookAt(camera_position, [0,0,0])
            self.scene.set_pose(self.scene.main_camera_node, pose = camera_pose)
            # Render the scene
            color, depth = r.render(self.scene, flags=1024) #skip backface culling
            #print('\n',self.scene.main_camera_node.matrix , '\n')
            #print('\n', np.max(color))
            #print(np.min(color))
            if np.max(color) == np.min(color):
                print("rendered nothing for %d-th pose" % i)
            #np.save(os.path.join(opt.folder_name, 'poses_train.npy'), view_sampler.points)

            # save png file
            im = PIL.Image.fromarray(color)
            # Down sample high resolution version
            if self.anti_alias > 1:
                im.thumbnail([self.height, self.width], PIL.Image.ANTIALIAS)

            # I/O - Json and image
            im.save((self.target_path + '%s/r_%04d.png' % (mode,i)))
            frames_data_dump.append({'file_path': './%s/r_%04d' % (mode,i),
                                    'transform_matrix':camera_pose.tolist()})
            with open(self.target_path + '/transforms_%s.json' % mode, 'w') as fp:
                json.dump({'camera_angle_x':self.camera_angle,'frames':frames_data_dump},
                          fp, indent=4, )

            # Change to true for visualization
            if True:
                import matplotlib.pyplot as plt
                if i<3:
                    plt.figure()
                    plt.imshow(color)
                    plt.figure()
                    plt.show()

            #view_sampler = SphericalSampler(opt.n_views_test, 'SPIRAL')
            #all_viewpoints['test'] = view_sampler.points
            #np.save(os.path.join(opt.folder_name, 'poses_test.npy'), view_sampler.points)


#camera_pose =  lookAt(  GL_to_blender(pose_dss)     , np.array([0,0,0]), np.array([0,0,1])   )

# Show the images
#import matplotlib.pyplot as plt
#plt.figure()
#plt.subplot(1,2,1)
#plt.axis('off')
#plt.imshow(color)
#plt.subplot(1,2,2)
#plt.axis('off')
#plt.imshow(depth, cmap=plt.cm.gray_r)
#plt.show()


if __name__ == "__main__":
    opt = Options().parse()
    opt.anti_alias = 4
    render = Renderer(opt)
    render.folder_name = './FLAME_sample/'
    #render.target_name = 'FLAME_overfit'
    #render.folder_name = './DSS/example_data/pointclouds/full_body_simp_tex/'
    render.process_mesh()
    render.get_viewpoints()
    #render.render_images('val')
    #render.render_images('train')
    render.get_test_sequence()
    render.render_images('test')


