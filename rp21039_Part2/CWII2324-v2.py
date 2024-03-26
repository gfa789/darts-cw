'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) #gets rid of deprecation warning in python 3.10
'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == '__main__': 

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')    
    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10, 
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16, 
                        help='max sphere  radius x10')
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4, 
                       help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8, 
                       help='max sphere  separation')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')

    args = parser.parse_args()

    if args.num<=0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min>=args.sph_rad_max or args.sph_rad_min<=0:
        print('invalid max and min sphere radii')
        exit()
    	
    if args.sph_sep_min>=args.sph_sep_max or args.sph_sep_min<=0:
        print('invalid max and min sphere separation')
        exit()
	
    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sphere with random radius
        size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2)/10
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # create random sphere location
        step = random.randrange(args.sph_sep_min,args.sph_sep_max,1)
        x = random.randrange(-h/2+2, h/2-2, step)
        z = random.randrange(-w/2+2, w/2-2, step)
        # print(x, size,z)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(-h/2+2, h/2-2, step)
            z = random.randrange(-w/2+2, w/2-2, step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
                    [[1, 0, 0, x],
                     [0, 1, 0, size],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]
                )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * (80+random.uniform(-10, 10))/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]

#####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.
    
    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width=640
    img_height=480
    f=415 # focal length
    # image centre in pixel coordinates
    ox=img_width/2-0.5 
    oy=img_height/2-0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)
    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()
    
##################################################

    # load in the images for post processings
    img0 = cv2.imread('view0.png', -1)
    dep0 = cv2.imread('depth0.png', -1)
    img1 = cv2.imread('view1.png', -1)
    dep1 = cv2.imread('depth1.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()

    
    ###################################
    '''
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    ###################################
    #Convert image to grayscale

    gray_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    #Apply Gaussian Blur to image
    gray_img0_blurred = cv2.medianBlur(gray_img0, 5)
    gray_img1_blurred = cv2.medianBlur(gray_img1, 5)
    # gray_img0_blurred = cv2.medianBlur(gray_img0_blurred, 5)
    gray_img1_blurred = cv2.medianBlur(gray_img1_blurred, 5)
    
    gray_img0_blurred = cv2.GaussianBlur(gray_img0_blurred, (5,5), 0)
    gray_img1_blurred = cv2.GaussianBlur(gray_img1_blurred, (5,5), 0)

    cv2.imshow('./img0_blurred.jpg', gray_img0_blurred)
    cv2.imshow('./img1_blurred.jpg', gray_img1_blurred)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    edges0 = cv2.Canny(gray_img0_blurred, 20, 50)
    edges1 = cv2.Canny(gray_img1_blurred, 20, 50)

    cv2.imshow('canny_edges0', edges0)
    cv2.imshow('canny_edges1', edges1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Find Hough Circles in Image - Parameters so that it aims to find more than 6
    circles_img0 = cv2.HoughCircles(gray_img0_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2 =20, minRadius= 10, maxRadius= 50)
    circles_img1 = cv2.HoughCircles(gray_img1_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2 =20, minRadius= 10, maxRadius= 50)

    #Take the first 6 found by cv2
    no_of_spheres = args.num

    circles_img0 = circles_img0[:,:no_of_spheres]
    circles_img1 = circles_img1[:,:no_of_spheres]

    #If less than 6 found in one image, make there the same amount in the other image
    if circles_img0.shape[1] < 6:
        circles_img1 = circles_img1[:, :circles_img0.shape[1]]

    if circles_img1.shape[1] < 6:
        circles_img0 = circles_img0[:, :circles_img1.shape[1]]

    #Create images to draw on
    output_hough_img0 = np.copy(img0)
    output_hough_img1 = np.copy(img1)

    #Draw Hough Circles
    if circles_img0 is not None:
        # circles_img0 = np.array(circles_img0)
        for i in range(circles_img0.shape[1]):
            x,y,r = np.round(circles_img0[0,i]).astype(int)
            output_hough_img0 = cv2.circle(output_hough_img0, (x,y),r, (0,0,255), 2)
    
    if circles_img1 is not None:
        # circles_img0 = np.array(circles_img0)
        for i in range(circles_img1.shape[1]):
            x,y,r = np.round(circles_img1[0,i]).astype(int)
            output_hough_img1 = cv2.circle(output_hough_img1, (x,y),r, (0,0,255), 2)
            
    #Save images
    cv2.imshow('Detected Circles IMG0', output_hough_img0)
    cv2.imshow('Detected Circles IMG1', output_hough_img1)
    cv2.imwrite("./hough_img0.jpg", output_hough_img0)
    cv2.imwrite("./hough_img1.jpg", output_hough_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ###################################
    '''
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    ###################################


    # Take x and y image coordinates from list of hough circles
    pts0 = circles_img0[0,:, :2]
    pts1 = circles_img1[0,:, :2]

    # Convert x and y coordinates to integers
    pts0 = np.round(pts0).astype(int)
    pts1 = np.round(pts1).astype(int)
    
    #Computer matrix which converts pixel coordinates to image coordinates
    sx=sy=1
    Mpi=np.array([[sx, 0, -ox/f],[0, sy, -oy/f],[0, 0, 1]])
    #Compute matrix which converts image coordinates to pixel coordinates
    Mip=np.linalg.inv(Mpi)

    #Find matrix transformation from camera 1 to camera 0
    H_10 = np.matmul(H0_wc, np.linalg.inv(H1_wc))

    #Calculate fundamental matrix
    R = H_10[:3, :3].T # R10^T
    T = H_10[:3, 3] # T10
    S = np.array([
        [0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]
    ])
    E = np.matmul(R, S)
    F = np.matmul(np.matmul(Mpi.T, E), Mpi)

    lines = []

    #Drawing lines on 2D-image captured by camera 1
    for i in range(len(pts0)):
        # get point from image 0
        pt1_img0 = np.array([pts0[i][0], pts0[i][1], f])

        #turn point into image 1 epipolar line
        u = np.matmul(F,pt1_img0).reshape(3,)

        #calculate left-most point 
        p0 = np.array([0, -f*u[2]/u[1] ]).astype(int)

        #calculate right-most point
        p1 = np.array([img_width, -(f*u[2]+u[0]*img_width)/u[1] ]).astype(int)
        #draw line
        img = cv2.line(img1, p0, p1, (255,0,0), 1)
        #append F*pr ready for correspondence
        lines.append(u)
    
    cv2.imshow("lines on image0", img0)
    cv2.imwrite("./lines_img0.jpg", img0)
    cv2.imshow("lines on image1", img1)
    cv2.imwrite("./lines_img1.jpg", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ###################################
    '''
    Task 5: Find correspondences

    Write your code here
    '''
    ###################################

    matches = []
    #for loop to minmise pl*F*pr using lines matrix from previous part
    for j in range(len(lines)):
        lowest_value = 9999999999
        lowest_index = None
        for i in range(len(pts1)):
            pt1_img1 = np.array([pts1[i][0], pts1[i][1], f])
            correspondence = np.matmul(pt1_img1.T, lines[j])
            if abs(correspondence) < lowest_value:
                lowest_index = i 
                lowest_value = abs(correspondence)
        matches.append([j, lowest_index])
    pts_corr = []
    for i in range(len(pts0)):
        pts_corr.append([pts0[i], pts1[matches[i][1]]])
    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''
    ###################################
    points = []
    #Apply 3D reconstruction equation
    for i in range(len(pts_corr)):
        # get corresponding points and convert into x=sx(x-ox) and y=sx(y-ox)
        # sx=1 so ignore
        pl = pts_corr[i][0]
        pl = np.array([pl[0] - ox, pl[1] - oy, f])
        pr = pts_corr[i][1]
        pr = np.array([pr[0] - ox, pr[1] - oy, f])
        Rtpr = np.matmul(R.T, pr)
        pl_x_Rtpr = np.cross(pl, Rtpr)
        H = np.vstack([pl, -Rtpr, -pl_x_Rtpr]).T

        # a,b,c = H^-1 * T
        HT = np.matmul(np.linalg.inv(H), T)

        #use a b and c to find point
        P = (HT[0] * pl + HT[1] * Rtpr + T)/2
        P = np.array([P[0], P[1], P[2], 1])
        P = np.matmul(np.linalg.inv(H0_wc), P)
        points.append(P[:3])
    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################
    #get rid of all meshes other than the plane
    obj_meshes = [obj_meshes[0]]
    mesh_list, H_list, RGB_list = [], [], []

    #plot ground truth centers
    for i in range(len(GT_cents)):
        name_list.append(f'ground truth{i}')
        radius =0.01
        size = radius
        sph_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([1.,0.,0.])
        x, y, z, _= GT_cents[i]
        sph_H = np.array([[1, 0, 0, x],
                          [0, 1, 0, y],
                          [0, 0, 1, z],
                          [0, 0, 0, 1]])
        H_list.append(sph_H)

    #plot calculated centers
    for i in range(len(points)):
        name_list.append(f'sphere{i}')
        radius = 0.01
        size = radius
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 1., 0.])
        # get sphere centre
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        sph_H = np.array(
                    [[1, 0, 0, x],
                        [0, 1, 0, y],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]]
                )
        H_list.append(sph_H)

    #for all ground truths - find which ones correspond to which point
    #draw line between the points
    center_distance = []
    gt_radius_corr = []
    gt_center_corr = []
    for p in points:
        x0, y0, z0 = p
        lowest_dist = 999999999
        lowest_point = None
        lowest_point_rad = None
        for (gt,gtr) in zip(GT_cents,GT_rads):
            distance = np.linalg.norm(p - gt[:3])
            if distance < lowest_dist:
                lowest_dist = distance
                lowest_point = gt
                lowest_point_rad = gtr
        x1, y1, z1, _ = lowest_point
        gt_radius_corr.append(lowest_point_rad)
        gt_center_corr.append([x1,y1,z1])
        center_distance.append(lowest_dist)
        line = o3d.geometry.LineSet()
        end_points = o3d.utility.Vector3dVector(np.array([[x0,y0,z0], [x1,y1,z1]]))
        line.points = end_points
        lines = o3d.utility.Vector2iVector([[0,1]])
        line.lines = lines
        line.colors = o3d.utility.Vector3dVector([np.array([0.,0.,1.])])
        obj_meshes.append(line)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)

    #draw all shapes
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    for m in obj_meshes:
        vis.add_geometry(m)

    vis.run()
    vis.destroy_window()


    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################
    #reset obj meshes to only include plane
    obj_meshes = [obj_meshes[0]]
    mesh_list, H_list, RGB_list = [], [], []
    geometries = []

    #plot ground truths
    for i in range(len(GT_cents)):
        name_list.append(f'ground truth{i}')
        radius = GT_rads[i]
        size = radius
        x, y, z, _= GT_cents[i]

        #create mesh
        sph_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)

        #translate mesh to center
        translation = np.array([x, size, z]) - np.array(sph_mesh.get_center())
        sph_mesh.translate(translation, relative=False)
        
        #convert sphere to point cloud - easier to interpret
        sph_mesh_pc = sph_mesh.sample_points_poisson_disk(number_of_points=50)
        colors = [np.array([1, 0, 0]) for _ in range(len(sph_mesh_pc.points))]
        sph_mesh_pc.colors = o3d.utility.Vector3dVector(colors)
        sph_mesh_pc.normals = o3d.utility.Vector3dVector(np.zeros((1, 3))) 
        geometries.append([sph_mesh_pc])

    radius_calc0 = []
    radius_calc1 = []
    radius_calc = []

    #plot detected spheres
    for i in range(len(points)):
        #calculate radius by scaling it using camera matrix 
        radius0 = circles_img0[0,i,2]
        radius0 *= (1 / np.linalg.norm(H0_wc))
        radius_calc0.append(radius0)

        radius1 = circles_img1[0,i,2]
        radius1 *= (1/np.linalg.norm(H1_wc))
        radius_calc1.append(radius1)

        #mean radii from both images to get a better calculation
        radius = 0.5 * (radius0 + radius1)
        # radius = np.max([radius0,radius1]
        radius_calc.append(radius)
        size = radius

        x = points[i][0]
        z = points[i][2]

        #create sphere mesh
        sph_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size, resolution=20)

        #translate the mesh to correct center 
        translation = np.array([x, size, z]) - np.array(sph_mesh.get_center())
        sph_mesh.translate(translation, relative=False)
        sph_mesh_pc = sph_mesh.sample_points_poisson_disk(number_of_points=50)
        colors = [np.array([0, 1, 0]) for _ in range(len(sph_mesh_pc.points))]
        sph_mesh_pc.colors = o3d.utility.Vector3dVector(colors)
        sph_mesh_pc.normals = o3d.utility.Vector3dVector(np.zeros((1, 3))) 

        geometries.append([sph_mesh_pc])

    
    ##################################
    '''
    DATA WRITING INTERLUDE
    '''
    ##################################
    #export data to csv
    with open('./results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Radius IMG1', 'Radius IMG2', 'AVG Radius', 'GT Radius', 'Radius difference', 'CALC CENTER', 'GT CENTER', 'CENTER DISTANCE', 'VOLUME OVERLAP'])
        for i in range(len(radius_calc)):
            d = center_distance[i]
            r0 = radius_calc[i]
            r1 = gt_radius_corr[i]
            volume_overlap = 0
            if d < (r0 + r1):
                volume_overlap_0 = (np.pi / (12 * d)) * np.square(r0 + r1 - d)
                volume_overlap_1 = (d**2 + 2*d*(r0 + r1) - 3*(r0**2 + r1**2) + 6 *(r0 + r1))
                volume_overlap = volume_overlap_0 * volume_overlap_1
            writer.writerow([radius_calc0[i], radius_calc1[i], radius_calc[i], gt_radius_corr[i], abs(gt_radius_corr[i] - radius_calc[i]), points[i], gt_center_corr[i], center_distance[i], volume_overlap])

    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    ###################################
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)

    #draw geometries
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)
    
    for m in obj_meshes:
        vis.add_geometry(m)

    for g in geometries:
        for p in g:
            vis.add_geometry(p)
    vis.run()
    vis.destroy_window()
    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ###################################
