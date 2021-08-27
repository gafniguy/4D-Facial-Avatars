import platform
print(platform.python_version())
import os
import time
import trimesh
#import pyrender
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import torch
import skimage.io
import os.path
from spherical_sampler import SphericalSampler
from tqdm import tqdm
from options import Options
import PIL
from PIL import Image
import logging
import io



import torch
import os
import argparse
import numpy as np
from itertools import chain
from glob import glob
from DSS.core.renderer import createSplatter
from DSS.utils.splatterIo import saveAsPng, readScene, readCloud, getBasename, writeCameras
from DSS.utils.trainer import CameraSampler
from DSS.options.render_options import RenderOptions
from rendering.spherical_sampler import SphericalSampler
from pytorch_points.utils.pc_utils import save_ply


def create_subfolders(folder_name):
    ## Create all necessary subfolders
    subfolders = []
    modes = ['train', 'test']
    for mode_name in modes:
        subfolders += [mode_name, os.path.join(mode_name, 'A'), os.path.join(mode_name, 'B')]
    subfolders += ['depth']
    for subfolder in subfolders:
        path = os.path.join(opt.folder_name, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)



def normalize(vector):
    return vector / np.linalg.norm(vector)

def lookAt(cam_pos_world, to_pos_world, tmp = np.array([0, 1, 0])):


    forward = normalize( cam_pos_world - to_pos_world) # pos if on unit sphere and model is centered

    right = normalize(np.cross(normalize(tmp), forward))
    up = normalize(np.cross(forward, right))

    camToWorld = np.zeros((4,4))

    camToWorld[0,:-1] = right
    camToWorld[1,:-1] = up
    camToWorld[2,:-1] = forward

    camToWorld[3,:-1] = cam_pos_world
    camToWorld[3,3] = 1
    return camToWorld.T

class Renderer:

    def __init__(self, opt):

        self.name = opt.name
        self.folder_name = opt.folder_name
        self.n_views_train = opt.n_views_train
        self.n_views_test = opt.n_views_test
        self.n_points = opt.train
        self.test = opt.test
        self.splat = opt.splat
        self.render = opt.render
        self.save_cam_space_coords = opt.save_cam_space_coords
        self.save_world_space_coords = opt.save_world_space_coords
        self.render = opt.render
        self.im_size = opt.im_size
        self.simple_mesh = opt.simple_mesh
        self.anti_alias = opt.anti_alias

        if self.folder_name is '':
            self.folder_name = self.name

        create_subfolders(self.folder_name)


    def process_mesh(self):
        # load a mesh
        #mesh = trimesh.load('armchair/armchair.obj')

        if self.simple_mesh:
            mesh_name = os.path.join(self.folder_name ,self.name + '_downsampled_2000.obj')
        else:
            mesh_name = os.path.join(self.folder_name, self.name+'.obj')

        print('Attempting to load mesh: %s ...' % mesh_name)
        if not os.path.exists(mesh_name):
            print("ERROR: mesh file does not exist")

        mesh = trimesh.load_mesh(mesh_name)

        scene = mesh.scene()

        self.n_points =  mesh.vertices.shape[0]

        print('Succesfully loaded. Mesh has %d vertices' % self.n_points)

                ## Mean centre and normalize mesh points ##

        #mesh.vertices -= mesh.center_mass
        mesh.vertices -= scene.centroid

        volume = mesh.bounding_sphere.volume
        radius = np.power((3*volume/(4*np.pi)),1/3)
        mesh.vertices = (1/(2*radius)) * mesh.vertices

        # homogeneous coords
        N = mesh.vertices.shape[0]
        verts_hom = np.ones((N,mesh.vertices.shape[1]+1), dtype=float)
        verts_hom[:,:-1] = mesh.vertices

        # to GPU
        self.verts_torch = torch.from_numpy(verts_hom).type(torch.cuda.FloatTensor)

        # look_dirs = trimesh.util.attach_to_log()


                ## Set up scene camera and canvas ##

        # get a scene object containing the mesh, this is equivalent to:
        # scene = trimesh.scene.Scene(mesh)

        camera = scene.camera
        camera.resolution = [self.im_size, self.im_size]
        camera_extrinsics = scene.camera_transform
        # camera.K = np.array([[1, 0, 256],[0, 1, 256],[0, 0, 1.]])

        camera_intrinsics = camera.K

        camera_intrinsics_hom = np.zeros((3, 4))
        camera_intrinsics_hom[:, :-1] = camera_intrinsics

        # camera_extrinsics = np.matmul(flip_z, camera_extrinsics)
        # camera_extrinsics = lookAt(np.array([0,0,-1]), np.array([0,0,0]))

        # print("camera_extrinsics \n", camera_extrinsics)
        # print("camera_intrinsics \n", camera.K)



        # zero = np.array([0,0,0,1])

        '''
        camera_extrinsics = np.array([
            [1.0, 0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ])
         '''

        self.camera_intrinsics_1 = np.array([
            [200.0 * self.anti_alias, 0, self.im_size * self.anti_alias / 2],
            [0.0, 200.0 * self.anti_alias, self.im_size * self.anti_alias / 2],
            [0.0, 0, 1.0]
        ])

        self.camera_intrinsics_1_hom = np.array([
            [0.0, 200.0, self.im_size / 2, 0.0],
            [-200.0, 0.0, self.im_size / 2, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        camera.K = self.camera_intrinsics_1
        # '''
        self.scene = scene
    # Top of main python script
    #os.environ['PYOPENGL_PLATFORM'] = 'egl'


    # Cam to World matrix!!


    def project_world_to_image_torch(self, cam2world, intrinsics, verts_torch, IM_SIZE):
        world2cam = np.linalg.inv(cam2world)

        intrinsics_torch = torch.from_numpy(intrinsics).type(torch.cuda.FloatTensor)
        world2cam_torch = torch.from_numpy(world2cam).type(torch.cuda.FloatTensor)
        #print("world2cam_torch \n", world2cam_torch)
        #print("intrinsics_torch \n", intrinsics_torch)

        matrix = torch.mm(intrinsics_torch, world2cam_torch )

        projected_points = torch.mm( matrix , verts_torch.t())

        projected_points = projected_points.t()
        projected_points_cam_space = projected_points.clone()
        #with torch.no_grad:

        z_vals = projected_points[:, 2]
        projected_points[:, 0] /= projected_points[:, 2]
        projected_points[:, 1] /= projected_points[:, 2]
        projected_points[:, 2] /= projected_points[:, 2]
        selection_mask_1 = ~torch.isnan(projected_points[:, 2])

        projected_points = torch.round(projected_points[selection_mask_1])
        z_vals = z_vals[selection_mask_1]

        # remove if out of frame bounds
        #                               u  > 0                  u < IM_SIZE
        selection_mask_2 = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < IM_SIZE) & (
            projected_points[:, 1] >= 0) & (projected_points[:, 1] < IM_SIZE)
        #                               v  > 0                  v < IM_SIZE

        # Valid splats
        projected_points = projected_points[selection_mask_2]
        pixels = projected_points[:, :2]
        pixels = pixels.long()
        z_vals = z_vals[selection_mask_2]

        #print(len(z_vals), " valid vertices projected to screen")
        result = torch.cuda.FloatTensor(IM_SIZE, IM_SIZE).fill_(float("Inf"))
        coords = torch.cuda.FloatTensor(IM_SIZE, IM_SIZE, 3).fill_(-1)  # x,y,z of the point, in world coords
        vert_ids = torch.cuda.FloatTensor(IM_SIZE, IM_SIZE).fill_(float(0))  # vertex_id of projected vertex

        verts_torch = verts_torch[selection_mask_1]
        verts_torch = verts_torch[selection_mask_2]
        projected_points_cam_space[selection_mask_1]
        projected_points_cam_space[selection_mask_2]

        for p in range(len(pixels)):
            # depth test
            if z_vals[p] < result[pixels[p][0], pixels[p][1]]:

                # Update depth buffer
                result[pixels[p][0], pixels[p][1]] = z_vals[p]

                # vert_id = selection_mask[p]
                # vert_ids[pixels[p]] = vert_id
                if self.save_world_space_coords:
                    coords[pixels[p][0], pixels[p][1],:] = verts_torch[p][:3]
                if self.save_cam_space_coords:
                    coords[pixels[p][0], pixels[p][1], :] = projected_points_cam_space[p][:3]
                    #coords[pixels[p][0], pixels[p][1], 0:1] = z_vals[p]

                # output one channel of vert_idx+1
                vert_ids[pixels[p][0], pixels[p][1]] = p+1   #### shifting vid by one!! for DL pipeline
        #coords = (coords + 1) / 2

        result[result == float("Inf")] = 0
        result = abs(result)
        result = result/torch.max(result) * 255
        return result, coords , vert_ids
    # print logged messages


    def splat_points_to_image(self, poses, mode, IM_SIZE = 0 ):

        if IM_SIZE == 0:
            IM_SIZE = self.im_size

        subfolder_save = os.path.join(self.folder_name, mode)

        for i in tqdm(range(len(poses))):
            camera_extrinsics = lookAt(poses[i], np.array([0, 0, 0]))

            #print("pose\n", camera_extrinsics)

            splat, coords, vert_ids = self.project_world_to_image_torch(camera_extrinsics, self.camera_intrinsics_1_hom,
                                                                        self.verts_torch, IM_SIZE)

            splat = splat.detach().cpu().numpy()
            coords = coords.detach().cpu().numpy()

            splat.astype("int8")


            file_name_depth = os.path.join(self.folder_name, os.path.join("depth", 'depth_%d.png' % i))

            #file_name_xyz = os.path.join(SUBFOLDER_NAME, os.path.join("xyz", 'xyz_%d.png' % i))
            file_name_xyz_npy = os.path.join(subfolder_save, os.path.join("A", 'pose_%d' % i))

            skimage.io.imsave(file_name_depth, splat.astype('uint8'))
            np.save(file_name_xyz_npy, np.dstack((coords, vert_ids.cpu().numpy())))
            #skimage.io.imsave(file_name_xyz, coords)

            #plt.imshow(splat, cmap='gray')
            #plt.imshow(coords)
            #plt.show()

    def render_color_images(self, poses, scene , mode, IM_SIZE = 0):

        if IM_SIZE == 0:
            IM_SIZE = self.im_size

        subfolder_save = os.path.join(self.folder_name, mode)
        scene.show()
        for i in tqdm(range(len(poses))):

            camera_extrinsics = lookAt(poses[i], np.array([0, 0, 0]))

            #trimesh.constants.log.info('Saving image %d', i)

            # rotate the camera view transform
            camera_old, _geometry = scene.graph[scene.camera.name]
            #camera_new = lookAt(poses[i], np.array([0,0,0]))
            camera_new = camera_extrinsics
            # apply the new transform
            scene.graph[scene.camera.name] = camera_new
            #camera.K = camera_intrinsics_1
            #print(np.matmul(scene.camera.K,np.matmul(camera_extrinsics,np.array([0,0,0,1]))[:3]))

            #aa, bb = scene.graph[scene.camera.name]
            #print("trimesh renders from \n", aa)
            #print("intrinsics\n" , scene.camera.K)
            #scene.show()
            # saving an image requires an opengl context, so if -nw
            # is passed don't save the image
            try:
                # increment the file name
                file_name = os.path.join(subfolder_save, os.path.join("B", 'pose_%d.png' % i))

                # save a render of the object as a png
                png = scene.save_image(resolution=[IM_SIZE * self.anti_alias, IM_SIZE * self.anti_alias], visible=True)
                #image = Image.frombytes("RGB", (IM_SIZE * self.anti_alias, IM_SIZE * self.anti_alias), png, 'raw')
                image = Image.open(io.BytesIO(png))
                # Down sample high resolution version
                if self.anti_alias > 1:
                    image.thumbnail([IM_SIZE, IM_SIZE], PIL.Image.ANTIALIAS)
                image.save(file_name, 'PNG')

                # with open(file_name, 'wb') as f:
                #     f.write(png)
                #     f.close()
                #
            except BaseException as E:
                print("unable to save image", str(E))
            time.sleep(0.05)




if __name__ == '__main__':

    opt = Options().parse()

    create_subfolders(opt.folder_name)
    #mode_name = ['test', 'train']
    #np.save('poses_%s.npy' % mode_name[TRAIN], poses)

    renderer = Renderer(opt)
    renderer.process_mesh()

    all_viewpoints = {}


    if opt.render_from_splat:
        all_viewpoints = np.load(os.path.join(opt.folder_name, opt.name+'_all_views'))
        intrinsics = np.load(os.path.join(opt.folder_name, opt.name+'_all_views_intrinsics'))

    if opt.train:
        view_sampler = SphericalSampler(opt.n_views_train, 'LATTICE')
        all_viewpoints['train'] = view_sampler.points
        np.save(os.path.join(opt.folder_name, 'poses_train.npy'), view_sampler.points)

        if opt.render:
            renderer.render_color_images(all_viewpoints['train'], renderer.scene, 'test')
        if opt.splat:
            renderer.splat_points_to_image(all_viewpoints['train'], 'train')


    if opt.test:
        view_sampler = SphericalSampler(opt.n_views_test, 'SPIRAL')
        all_viewpoints['test'] = view_sampler.points
        np.save(os.path.join(opt.folder_name, 'poses_test.npy'), view_sampler.points)

        if opt.render:
            renderer.render_color_images(all_viewpoints['test'], renderer.scene, 'test')
        if opt.splat:
            renderer.splat_points_to_image(all_viewpoints['test'], 'test')

   # SIMPLIFIED_MESH = GENERATE_SPLATTING


#opt.name = "full_body"









if __name__ == "__main__":
    opt = RenderOptions().parse()
    points_paths = list(chain.from_iterable(glob(p) for p in opt.points))
    assert(len(points_paths) > 0), "Found no point clouds in with path {}".format(points_paths)
    points_relpaths = None
    if len(points_paths) > 1:
        points_dir = os.path.commonpath(points_paths)
        points_relpaths = [os.path.relpath(p, points_dir) for p in points_paths]
    else:
        points_relpaths = [os.path.basename(p) for p in points_paths]

    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(24)

    view_sampler = SphericalSampler(300, 'SPIRAL')
    points = view_sampler.points
    save_ply(points,'example_data/pointclouds/spiral_300.ply',normals=points)


    scene = readScene(opt.source, device="cpu")
    opt.genCamera = 300
    if opt.genCamera > 0:
        camSampler = CameraSampler(opt.genCamera, opt.camOffset, opt.camFocalLength, points=scene.cloud.localPoints,
                                   camWidth=opt.width, camHeight=opt.height, filename="example_data/pointclouds/spiral_300.ply")
        camSampler.closer = False
    with torch.no_grad():
        splatter = createSplatter(opt, scene=scene)
        #splatter.shading = 'depth'
        if opt.genCamera > 0:
            cameras = []
            for i in range(opt.genCamera):
                cam = next(camSampler)
                cameras.append(cam)

            splatter.initCameras(cameras=cameras)
            writeCameras(scene, os.path.join(opt.output, 'cameras.ply'))
        else:
            for i in range(len(scene.cameras)):
                scene.cameras[i].width = opt.width
                scene.cameras[i].height = opt.height

        splatter.initCameras(cameras=scene.cameras, genSunMode="triColor")

        for pointPath, pointRelPath in zip(points_paths, points_relpaths):
            keyName = os.path.join(os.path.join(opt.output, pointRelPath[:-4]))
            points = readCloud(pointPath, device="cpu")
            scene.loadPoints(points)
            fileName = getBasename(pointPath)
            splatter.setCloud(scene.cloud)
            successful_views= []
            rendered = []
            for i, cam in enumerate(scene.cameras):
                splatter.setCamera(i)
                result = splatter.render()
                if result is None:
                    print("splatted a None")
                    continue
                result = result.detach()[0]
                rendered.append(result)
            print(pointRelPath)
            for i, gt in enumerate(rendered):
                if splatter.shading == "albedo":
                    cmax = 1
                else:
                    cmax = None
                saveAsPng(gt.cpu(), keyName + '_cam%02d.png' % i, cmin=0, cmax=cmax)
            # stacked = torch.stack(rendered, dim=0)
            # np.save(keyName+'_views.npy', stacked.cpu().numpy())

    all_views = [camer.world2CameraMatrix_test(camer.rotation, camer.position).detach().cpu().numpy() for camer in scene.cameras]
    all_views_intrinsics = [camer.projectionMatrix().detach().cpu().numpy() for camer in scene.cameras]
    np.save(keyName+'_all_views', all_views)
    np.save(keyName+'_all_views_intrinsics', all_views_intrinsics)

    #print(all_views[0])