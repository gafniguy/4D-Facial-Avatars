#! /usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SphericalSampler:
    """
    Sample N points on unit sphere, to use these as camera positions in data generation process.
    """

    def __init__(self, N, sampling="LATTICE"):
        self.N = N
        if sampling == "LATTICE":
            self.points = self.sphere_fibonacci_grid_points(N)
        elif sampling == "RANDOM":
            self.points = self.sphere_sample_gaussian(N)
        elif sampling == "CURVE":
            self.points = self.sphere_sample_curve(N)
        elif sampling == "SPIRAL":
            self.points = self.sphere_sample_spiral(N)
        elif sampling == "HELIX":
            self.points = self.sphere_sample_helix(N)
        elif sampling == "ARC":
            self.points = self.sphere_sample_arc(N)

        else:
            raise NameError('Sampling of type: %s not supported. Use: LATTICE | RANDOM' % sampling)
        print("Sampled %d views from unit sphere with %s sampling." % (N, sampling))

    def sphere_fibonacci_grid_points(self, ng):
        """
         SPHERE_FIBONACCI_GRID_POINTS: Fibonacci spiral gridpoints on a sphere.

          Licensing:

            This code is distributed under the GNU LGPL license.

          Modified:

            15 May 2015
            Author:
                John Burkardt

          Reference:

            Richard Swinbank, James Purser,
            Fibonacci grids: A novel approach to global modelling,
            Quarterly Journal of the Royal Meteorological Society,
            Volume 132, Number 619, July 2006 Part B, pages 1769-1793.

          Parameters:

            Input, integer NG, the number of points.

            Output, real XG(3,N), the grid points.
        """
        phi = (1.0 + np.sqrt(5.0)) / 2.0

        theta = np.zeros(ng)
        sphi = np.zeros(ng)
        cphi = np.zeros(ng)

        for i in range(0, ng):
            i2 = 2 * i - (ng - 1)
            theta[i] = 2.0 * np.pi * float(i2) / phi
            sphi[i] = float(i2) / float(ng)
            cphi[i] = np.sqrt(float(ng + i2) * float(ng - i2)) / float(ng)

        xg = np.zeros((ng, 3))

        for i in range(0, ng):
            xg[i, 0] = cphi[i] * np.sin(theta[i])  # x
            xg[i, 1] = cphi[i] * np.cos(theta[i])  # y
            xg[i, 2] = sphi[i]  # z

        return xg

    def sphere_sample_gaussian(self, N):
        points = np.zeros((N,3))
        for i in range(N):
            x = np.random.normal()
            y = np.random.normal()
            z = np.random.normal()
            point = np.array([x,y,z])
            points[i,:] = point / np.linalg.norm(point)
            points[i,2] = abs(points[i,2])
        return points

    def sphere_sample_curve(self,N,  theta = np.pi/2):

        phi = np.linspace(0, 2*np.pi, num=N, endpoint=False, retstep=False, dtype=float, axis=0)
        points = np.zeros((N,3))

        x = np.sin(theta) * np.cos(phi)
        z = np.sin(theta) * np.sin(phi)
        y = np.cos(theta)
        points[:,0] = x
        points[:,1] = y
        points[:,2] = z

        print(points.shape)

        return points

    def sphere_sample_spiral(self,N):

        phi = np.linspace(0,1, num=N, endpoint=False, retstep=False, dtype=float, axis=0)
        points = np.zeros((N,3))

        x = phi * np.cos(16*phi)
        z = phi *  np.sin(16*phi)

        y = np.sqrt(1 - np.square(x) - np.square(z))


        #z = np.sin(theta) * np.sin(phi)
        points[:,0] = x
        points[:,1] = y
        points[:,2] = z

        for i in range(N):
            points[i,:] = points[i,:] / np.linalg.norm(points[i,:])
        print(points.shape)

        return points

    def sphere_sample_arc(self,N):

        t = np.linspace(0,1, num=N, endpoint=False, retstep=False, dtype=float, axis=0)
        points = np.zeros((N,3))
        theta_x = np.linspace(-0.5,0.5, num=N, endpoint=False, retstep=False, dtype=float, axis=0)
        theta_y = np.linspace(-0.2,0.2, num=N, endpoint=False, retstep=False, dtype=float, axis=0)
        #x = np.sin(theta_x)
        #y = np.cos(theta_y)
        #z = 0.7

        points[:,0] = theta_x
        points[:,1] = theta_y
        points[:,2] = 0.7

        # for i in range(N):
        #     points[i,:] = points[i,:] / np.linalg.norm(points[i,:])
        print(points.shape)

        return points

    def sphere_sample_helix(self,N):

        t = np.linspace(0,1, num=N, endpoint=False, retstep=False, dtype=float, axis=0)
        points = np.zeros((N,3))

        x = np.cos(3*t*np.pi)
        y = np.sin(3*t*np.pi)
        z = t

        points[:,0] = x
        points[:,1] = y
        points[:,2] = z

        # for i in range(N):
        #     points[i,:] = points[i,:] / np.linalg.norm(points[i,:])
        print(points.shape)

        return points

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[ :,0], self.points[:,1], self.points[:,2], zdir='z', s=5, c=None, depthshade=True)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


if __name__ == "__main__":
    sampler = SphericalSampler(200, 'LATTICE')
    sampler.visualize()
    print(sampler.points.shape)
