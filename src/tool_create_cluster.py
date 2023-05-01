#!/usr/bin/env python3

import sys
import numpy as np

def calc_cluster_langevin(eta, pos):
    """Compute the effective translational and rotational damping acting on a CM of a cluster of N particles"""

    N = pos.shape[0] # Size of the cluster
    # CM translational viscosity [fKg/ms]
    etat_eff = eta*N
    # CM rotational viscosity [micron^2*fKg/ms]
    etar_eff = eta*np.sum(pos**2) # Prop to N^2. Varying with shape.
    #etar_eff = eta*N**2 # Not varying with shape.
    return etat_eff, etar_eff

def save_xyz(pos, outfname='cluster.xyz', elem='X'):
    """Save cluster as xyz in given file (default 'cluster.xyz')"""
    N = pos.shape[0]
    xyz_out = open(outfname, 'w')
    xyz_out.write(str(N) + '\n#\n')
    for i in range(N):
        print(elem + " %20.15f %20.15f %20.15f" % tuple(pos[i,:3]), file=xyz_out)
    xyz_out.close()

def load_cluster(input_hex, angle=0, center=False):
    """Load cluster form numpy file data. Optionally adjust CM and rotate."""

    #pos = np.loadtxt(input_hex)
    pos = np.load(input_hex)
    if center: pos -= np.average(pos, axis=0 )
    pos = rotate(pos, angle)
    return pos

def get_rotomatr(angle):
    """Get ACW rotation matrix of an angle [degree] """
    roto_mtr = np.array([[np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)],
                         [-np.sin(angle/180*np.pi), np.cos(angle/180*np.pi)]])
    return roto_mtr

def rotate(pos, angle, c=[0,0]):
    """Rotate positions pos of angle [degree] with respect to center c (default 0,0)"""
    # Equivalent to EP below, but faster. Or should be! - AS
    #for i in range(pos.shape[0]):
    #    newx = pos[i,0] * np.cos(angle/180*np.pi) - pos[i,1] * np.sin(angle/180*np.pi)
    #    newy = pos[i,0] * np.sin(angle/180*np.pi) + pos[i,1] * np.cos(angle/180*np.pi)
    #    pos[i,0] = newx
    #    pos[i,1] = newy
    roto_mtr = np.array([[np.cos(angle/180*np.pi), -np.sin(angle/180*np.pi)],
                         [np.sin(angle/180*np.pi), np.cos(angle/180*np.pi)]])
    pos = np.dot(roto_mtr, (pos-c).T).T # NumPy inverted convention on row/col
    return pos + c

def create_cluster(input_cluster, angle=0):
    """Create clusters taking as input the two primitive vectors a1 and a2 and the indices of lattice points.
    Put center of mass in zero."""

    file = open(input_cluster, 'r')
    N = int(file.readline())
    a1 = [float(x) for x in file.readline().split()]
    a2 = [float(x) for x in file.readline().split()]
    pos = np.zeros((N,2))
    for i in range(N):
        index = [float(x) for x in file.readline().split()]
        pos[i,0] = index[0]*a1[0]+index[1]*a2[0]
        pos[i,1] = index[0]*a1[1]+index[1]*a2[1]
    pos -= np.average(pos,axis=0)
    pos = rotate(pos, angle)
    return pos

def create_input_hex(N1, N2, clgeom_fname='in.hex', a1=np.array([4.45, 0]), a2=np.array([-4.45/2, 4.45*np.sqrt(3)/2])):
    """Create input file in EP .hex format

    Cluster is created from Bravais lattice a1 a2 with N1 repetition along a1 and N2 repetitions along N2.
    Default is Xin colloids: triangular with spacing 4.45"""

    clgeom_file = open(clgeom_fname, 'w')
    print("%i %i" % (N1, N2), file=clgeom_file)
    print("%15.10g %15.10g" % (a1[0], a1[1]), file=clgeom_file)
    print("%15.10g %15.10g" % (a2[0], a2[1]), file=clgeom_file)
    clgeom_file.close() # Close file so MD function can read it

    return clgeom_fname

def load_input_hex(instream):
    """Load .hex file defining lattice and size"""
    N1, N2 = [int(x) for x in instream.readline().split()]
    a1 = np.array([float(x) for x in instream.readline().split()])
    a2 = np.array([float(x) for x in instream.readline().split()])
    return N1, N2, a1, a2

def create_cluster_circle(input_hex, outstream=sys.stdout, X0=0, Y0=0):
    """Circle cluster

    The 6 dimensions are for backward compatibility with EP."""
    # Load lattice
    file = open(input_hex, 'r')
    N1, N2, a1, a2 = load_input_hex(file)
    file.close()
    a1norm, a2norm = np.linalg.norm(a1), np.linalg.norm(a2)

    # Create the positions
    pos, ipos = np.zeros((0,6)), np.zeros((0,2), dtype='int')
    iN = 0
    for i in range(-N1*2+1, N1*2+1):
        for j in range(-N2*2+1, N2*2+1):
            X = j*a1[0]+i*a2[0]
            Y = j*a1[1]+i*a2[1]
            if ((X*X + Y*Y) < N1/2*N2/2*a1norm*a2norm):
                pos = np.append(pos,[[X,Y,0,0,0,0]], axis=0)
                ipos = np.append(ipos,[[j, i]], axis=0)
                iN = iN+1
    # Save the positions
    print(iN, file=outstream)
    print(a1[0], a1[1], file=outstream)
    print(a2[0], a2[1], file=outstream)
    for i in range(iN):
        print(ipos[i,0], ipos[i,1], file=outstream)
    # CM in X0,Y0
    pos = pos - np.mean(pos,axis=0) + np.array([X0, Y0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    return pos

def create_cluster_hex(input_hex, outstream=sys.stdout, X0=0, Y0=0):
    """Hexagonal cluster

    The 6 dimensions are for backward compatibility with EP."""
    # Load lattice
    file = open(input_hex, 'r')
    N1, N2, a1, a2 = load_input_hex(file)
    file.close()

    N = int(N1*N2+((N2-1)*N2)/2+N2*(N1-1)+((N1-2)*(N1-1))/2)
    pos = np.zeros((N,6))
    # Create and print at the same time
    iN = 0
    print('{}'.format(N), file=outstream)
    print('{} {}'.format(a1[0], a1[1]), file=outstream)
    print('{} {}'.format(a2[0], a2[1]), file=outstream)
    for i in range(N2):
        for j in range(N1+i):
            pos[iN,0] = j*a1[0]+i*a2[0]
            pos[iN,1] = j*a1[1]+i*a2[1]
            print('{} {}'.format(i, j), file=outstream)
            iN = iN+1
    for i in range(N2,(N2+N1-1)):
        for j in range(N1+N2-2,i-N2,-1):
            pos[iN,0] = j*a1[0]+i*a2[0]
            pos[iN,1] = j*a1[1]+i*a2[1]
            print('{} {}'.format(i,j), file=outstream)
            iN = iN+1
    # CM in X0,Y0
    pos = pos - np.mean(pos,axis=0) + np.array([X0, Y0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    return pos

def create_cluster_special(input_hex, outstream=sys.stdout, X0=0, Y0=0):
    """Special-parall cluster. See paper on XXX.

    The 6 dimensions are for backward compatibility with EP."""
    file = open(input_hex, 'r')
    N1, N2, a1, a2 = load_input_hex(file)
    file.close()

    if N1!=N2: raise ValueError("Need same N for special rect")
    if (np.sqrt(N1)-1)/2 % 1 > 1e-10: raise ValueError("Need N so that (sqrt(N)-1)/2 is integer! Now %.5g" % ((np.sqrt(N1)-1)/2))
    a1norm, a2norm = np.linalg.norm(a1), np.linalg.norm(a2)
    # Create
    pos, ipos = np.zeros((0,6)), np.zeros((0,2), dtype='int')
    iN = 0
    for i in np.arange(-(np.sqrt(N1)-1)/2, (np.sqrt(N1)-1)/2+1):
        for j in np.arange(-(np.sqrt(N2)-1)/2, (np.sqrt(N2)-1)/2+1):
            X = j*a1[0]+i*a2[0]
            Y = j*a1[1]+i*a2[1]
            if (np.abs(X) < N1/2*a1norm) and (np.abs(Y) < N2/2*a2norm):
                pos = np.append(pos,[[X,Y,0,0,0,0]], axis=0)
                ipos = np.append(ipos,[[j, i]], axis=0)
                iN = iN+1
    # Save
    print(iN, file=outstream)
    print(a1[0], a1[1], file=outstream)
    print(a2[0], a2[1], file=outstream)
    for i in range(iN):
        print(ipos[i,0], ipos[i,1], file=outstream)
    # CM in X0,Y0
    pos = pos - np.mean(pos,axis=0) + np.array([X0, Y0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    return pos


def create_cluster_rect(input_hex, outstream=sys.stdout, X0=0, Y0=0):
    """Rectangular cluster

    The 6 dimensions are for backward compatibility with EP."""
    file = open(input_hex, 'r')
    N1, N2, a1, a2 = load_input_hex(file)
    file.close()
    a1norm = np.linalg.norm(a1)
    a2norm = np.linalg.norm(a2)
    # Calculate
    pos, ipos = np.zeros((0,6)), np.zeros((0,2), dtype='int')
    iN = 0
    for i in range(-N1*2+1, N1*2+1):
        for j in range(-N2*2+1, N2*2+1):
            X = j*a1[0]+i*a2[0]
            Y = j*a1[1]+i*a2[1]
            if (np.abs(X) < N1/2*a1norm) and (np.abs(Y) < N2/2*a2norm):
                pos = np.append(pos,[[X,Y,0,0,0,0]], axis=0)
                ipos = np.append(ipos,[[j, i]], axis=0)
                iN = iN+1
    # Save
    print(iN, file=outstream)
    print(a1[0], a1[1], file=outstream)
    print(a2[0], a2[1], file=outstream)
    for i in range(iN):
        print(ipos[i,0], ipos[i,1], file=outstream)
    # CM in X0,Y0
    pos = pos - np.mean(pos,axis=0) + np.array([X0, Y0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    return pos

def create_cluster_tri(input_hex, outstream=sys.stdout, X0=0, Y0=0):
    """Triangular cluster

    The 6 dimensions are for backward compatibility with EP."""
    file = open(input_hex, 'r')
    N1, N2, a1, a2 = load_input_hex(file)
    file.close()

    # Create with baricentric coordinates
    x1, y1 = N1*a1
    x2, y2 = N2*a2
    x3, y3 = -(N1*a1+N2*a2)
    pos = np.zeros((0,6))
    ipos = np.zeros((0,2), dtype='int')
    iN = 0
    for i in range(-N1*2+1, N1*2+1):
        for j in range(-N2*2+1, N2*2+1):
            x = j*a1[0]+i*a2[0]
            y = j*a1[1]+i*a2[1]
            denom = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
            a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
            b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
            c = 1 - a - b
            if ((0 < a and a < 1)
                and (0 < b and b < 1)
                and (0 < c and c< 1)):
                pos = np.append(pos,[[x,y,0,0,0,0]], axis=0)
                ipos = np.append(ipos,[[j, i]], axis=0)
                iN = iN+1
    # Save
    print(iN, file=outstream)
    print(a1[0], a1[1], file=outstream)
    print(a2[0], a2[1], file=outstream)
    for i in range(iN):
        print(ipos[i,0], ipos[i,1], file=outstream)
    pos = pos - np.mean(pos,axis=0) + np.array([X0, Y0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    return pos

def cluster_inhex_Nl(N1, N2,  a1=np.array([4.45, 0]), a2=np.array([-4.45/2, 4.45*np.sqrt(3)/2]),
                     clgeom_fname="input_pos.hex", cluster_f=create_cluster_circle, X0=0, Y0=0):
    """Create a cluster from lattice details (equivalent to .hex file) and specific shape function.

    The .hex file will be created and removed using tempfile.
    Returns only xy positions.

    Default values relate to Xin/EP colloids [Nat. Phys 2019, PRE 2021] and circular shape"""
    from tempfile import NamedTemporaryFile
    clgeom_file = open(clgeom_fname, 'w') # Save position file for MD/static/driver module to read it
    with NamedTemporaryFile(prefix='hex', suffix='tmp', delete=True) as tmp: # write input on tempfile. Delete once used
        tmp.write(bytes("%i %i\n" % (N1, N2), encoding='utf-8'))
        tmp.write(bytes("%15.10g %15.10g\n" % (a1[0], a1[1]), encoding='utf-8'))
        tmp.write(bytes("%15.10g %15.10g\n" % (a2[0], a2[1]), encoding='utf-8'))
        tmp.seek(0) # Reset 'reading needle'
        pos = cluster_f(tmp.name, clgeom_file, X0=X0, Y0=Y0)[:,:2] # Pass name of tempfile to create function. Only keep xy pos.
    clgeom_file.close() # Close file so MD function can read it

    return pos, clgeom_fname

def cluster_from_params(params):
    """Create cluster from parameters in dictionary.

    Return xy positions"""

    # Read params
    clt_shape = params['cluster_shape'] # Select function associate to different cluster shape
    # define cluster shape
    if clt_shape == 'circle':
        create_cluster = create_cluster_circle
    elif clt_shape == 'hexagon':
        create_cluster = create_cluster_hex
    elif clt_shape == 'rectangle':
        create_cluster = create_cluster_rect
    elif clt_shape == 'triangle':
        create_cluster = create_cluster_tri
    elif clt_shape == 'special':
        create_cluster = create_cluster_special
    elif clt_shape == 'polygon':
        # Get poly
        poly = get_poly(params['cl_poly'])
        direction = params['direction'] if 'direction' in params.keys() else 0
        return cluster_poly(poly, params, direction) # Create cluster by masking poly
    else:
        raise NotImplementedError("Shape %s not implemented" % clt_shape)
    # Define cluster
    a1, a2, = np.array(params['a1']), np.array(params['a2'])
    N1, N2 = params['N1'], params['N2']
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(prefix='inhex', suffix='tmp', delete=True) as tmp: # write input on tempfile. Delete once used
        pos, _ = cluster_inhex_Nl(N1, N2, a1=a1, a2=a2, clgeom_fname=tmp.name, cluster_f=create_cluster)

    pos = add_basis(pos, params['cl_basis'])
    return pos

def get_poly(points, scale=1, tho=0, c=[0,0], shift=0, cm=False):
    """Get a polygon (Shapely obejct) from a set of points.

    Optinally rotate, scale and shfit"""
    from tool_create_cluster import rotate
    from shapely.geometry import Polygon
    points = scale*np.asarray(points) # scale the points
    if cm: points -= np.mean(points) # rotate around the cm
    points = rotate(points, tho, c) # rotate the cutting poligon
    points += np.asarray(shift) # shift, useful to tailor the masking
    return Polygon(points) # Get a handy poligon object

def cluster_poly(polygon, params, direction=0):
    """Use a polygon (shapely object) to mask a lattice (defined in params).

    Direction = 0: select the interior of the polygon (use bounds of polygon to define lattice big enough for mask)
    Direction = 1: select the exterior of the polygon (up to the N1 N2 in params)

    Return xy pos"""
    # Build a grid large enough for the polygon.
    # Is grid shape specify?
    if 'masked_shape' not in params.keys(): params['cluster_shape'] = 'rectangle'
    else: params['cluster_shape'] = params['masked_shape']
    if direction == 0: # Get minimum size for
        l = np.max([np.linalg.norm(params['a1']), np.linalg.norm(params['a2'])])
        Nlbound = 3*int(np.ceil(np.max(np.abs(polygon.bounds)/l)))
        params['N1'], params['N2'] = Nlbound, Nlbound
    pos = cluster_from_params(params)
    if 'theta' in params.keys(): pos = rotate(pos, params['theta']) # rotate the lattice

    # Mask: direction = 0 include, direction = 1 exclude
    from shapely.geometry import MultiPoint
    Ppos = MultiPoint(pos) # Needs to be a set of point
    # This might be slow for large polygons
    mask = np.array([int(polygon.contains(i)) - bool(direction) # For each point, ask if it's in/out
                     for i in Ppos.geoms],
                    dtype=bool)
    # Masked positions
    return pos[mask]

def add_basis(lat_pos, basis):
    """Add a crystal basis to a simple Bravais lattice"""
    pos = lat_pos[:, np.newaxis,:] +  basis
    return np.reshape(pos, (pos.shape[0]*pos.shape[1], 2))

def params_from_ASE(ase_geom, cut_z=0, tol=0.9):
    """Create JSON parameters to create clusters from ASE Atoms object.

    Assume this is already a 2D surface oriented along z to extract 2x2 matrix:
    From:
      a b 0
      c d 0
      0 0 1
    Take:
      a b
      c d
    Positions are flattend out: (x,y) from (x,y,z)

    Return parameters dictionary and list of z coordinates considered
    """

    # Extarct 2D sys
    cell2d = ase_geom.cell.array[[0,1], :2]
    pos2d = ase_geom.positions[:,:2]
    # Mask single layer
    pos_z = ase_geom.positions[:,2]
    #pos_z -= np.min(pos_z)
    if cut_z == 0:
        cut_z = np.std(pos_z)
        print('No cut_z given, used std=%.4g' % cut_z, file=sys.stderr)
    mask = pos_z<cut_z*tol
    pos2d = pos2d[mask]
    print('%s: Selcted %i atoms at z:' % (__name__, len(pos_z[mask])), pos_z[mask], file=sys.stderr)
    if len(pos2d) == 0: raise RuntimeError('No atoms selected (cut_z=%.4g tol=%.4g)' % (cut_z, tol))
    params = {
        'a1': list(cell2d[0]),
        'a2': list(cell2d[1]),
        'cl_basis': [list(v) for v in pos2d], # Json encoder can't cope with numpy. Annoying.
    }
    return params, pos_z[mask]

def params_from_poscar(poscar_fname, cut_z=0):
    """Create JSON parameters to create clusters from POSCAR file.

    Use ASE to read.

    Return parameters file.
    """
    import ase
    ase_geom = ase.io.read(poscar_fname)
    return params_from_ASE(ase_geom, cut_z)

if __name__ == "__main__":
    input_hex = sys.argv[1]
    X0 = 0.0
    Y0 = 0.0
    clt_shape = sys.argv[2]
    if clt_shape == 'circle':
        create_cluster_func = create_cluster_circle
    elif clt_shape == 'hexagon':
        create_cluster_func = create_cluster_hex
    elif clt_shape == 'rect':
        create_cluster_func = create_cluster_rect
    elif clt_shape == 'tri':
        create_cluster_func = create_cluster_tri
    else:
        raise ValueError("Shape %s not implemented" % clt_shape)

    if len(sys.argv)>4:
        X0, Y0 = sys.argv[3], sys.argv[4]
    create_cluster_func(input_hex, X0=X0, Y0=Y0)
