import numpy as np
import matplotlib.pyplot as plt
import tifffile
import scipy.signal as scs
from scipy.fftpack import dct, idct
from scipy.ndimage import median_filter
import argparse
import scipy

"""Python implementation of https://github.com/vccimaging/PhaseIntensityMicroscope/tree/master/matlab/cws.m
- Does not use tilt removal when calculating OPD from phi
- Uses R2LASSO_simple instead of R2LASSO_complete (not implemented) in phase_updae_ADMM
- Return almost the same loss as cws.m
- OPD output in meters
- Amp: actual amplitude
- phi: just OPD without scale factor"""



class CWS():
    def __init__(self):
        # define operators in spatial domain
        self.x_k = np.array([[0, 0, 0], 
                        [-1, 1, 0],
                        [0, 0, 0]])

        self.y_k = np.array([[0, -1, 0], 
                        [0, 1, 0],
                        [0, 0, 0]])

        self.l_k = np.array([[0, 1, 0], 
                        [1, -4, 1],
                        [0, 1, 0]])

    def run(self, I0_path, I1_path, prior=None, iter=None):
        
        if type(I0_path) != np.ndarray:
            I0 = tifffile.imread(I0_path).astype(np.float64)[..., 0] # reference image
            I1 = tifffile.imread(I1_path).astype(np.float64)[..., 0] # object image
        else:
            # for videos, just send in loaded frames
            I0 = I0_path
            I1 = I1_path

        opt = {
            'priors': prior if prior is not None else [0.1, 0.1, 100, 5],
            'iter': iter if iter is not None else [10, 5, 10],
            'mu': [0.1, 100],
            'tol': 0.05,
            'L': np.array([(2**(np.ceil(np.log2(I0.shape))) - I0.shape)/2, [256, 256]]).min(axis=0).astype('int'),
            'verbose': [0, 0]
            }
        
        # tradeoff parameters
        alpha = opt['priors'][0]
        beta = opt['priors'][1]
        gamma = opt['priors'][2]
        tau = opt['priors'][3]

        # total number of alternation iterations
        iter = opt['iter'][0]

        # A-update parameters
        opt_A = {
            'isverbose': 0,
            'iter': opt['iter'][1],
            'mu_A': opt['mu'][0]
        }

        # phi-update parameters
        opt_phase = {
            'isverbose': 0,
            'L': opt['L'],
            'mu': opt['mu'][1],
            'iter': opt['iter'][2]
        }

        # tolerance
        tol = opt['tol']

        # compute A-update parameters
        gamma_new = gamma/np.square(np.abs(I0)).mean()
        tau_new = tau/np.square(np.abs(I0)).mean()

        # define sizes
        M = I0.shape
        self.L = opt_phase['L']
        N = M + 2*self.L

        # pre-calculated variables
        _, K_mat = self.prepare_DCT_basis(M)

        # inversion basis in DCT domain for A-update and phi-update
        mat_A_hat = 1 + (tau_new + opt_A['mu_A'])*K_mat


        # initialization

        A   = np.ones(M, dtype=np.float64)
        phi = np.zeros(N, dtype=np.float64)
        I_warp = I1

        # main loop
        objvaltotal = np.zeros(iter)
        self.disp_obj('iter = ', 0, self.obj_total(A, I_warp, phi, alpha, beta, gamma, tau, I0))

        for outer_loop in range(1, iter+1):
            # === A-update ===
            B = np.zeros(list(np.shape(A)) + [3], dtype=np.float64)
            zeta = np.zeros(list(np.shape(A)) + [3], dtype=np.float64)
            print('-- A-update')
            
            for i_A in range(1, opt_A['iter']+1):
                # A-update
                A = self.idct2(self.dct2(I_warp/I0 + opt_A['mu_A']*self.KT(B-zeta))/mat_A_hat)
                
                # pre-cache
                u = self.K(A) + zeta
                
                # B-update
                B = self.prox_A(u, gamma_new, opt_A)
                
                # zeta-update
                zeta = u - B
                
                # show objective function
                if opt_A['isverbose']:
                    self.disp_obj('---- iter = ', i_A, self.obj_A(A, I_warp))

            # median filter A
            A = median_filter(A, [3, 3], mode='mirror')
            
            print('-- phi-update')
            img = np.concatenate([(A*I0)[:, :, np.newaxis], I_warp[:, :, np.newaxis]], -1)
            _, Delta_phi, I_warp = self.phase_update_ADMM(img, alpha, beta, opt_phase)
            
            print(f'-- mean(|\Delta\phi|) = {np.abs(Delta_phi).mean():.3e}')
            
            if np.abs(Delta_phi).mean() < tol:
                print('-- mean(|\Delta\phi|) too small; quit')
                break
            phi += Delta_phi

            # === records ===
            objvaltotal[outer_loop-1] = self.obj_total(A, I_warp, phi, alpha, beta, gamma, tau, I0)
            self.disp_obj('iter = ', outer_loop, objvaltotal[outer_loop-1])    
            
        # compute wavefront Laplacian
        wavefront_lap = self.nabla2(phi)
        wavefront_lap = self.M1(np.concatenate([wavefront_lap[:, :, np.newaxis], wavefront_lap[:, :, np.newaxis]], -1))
        wavefront_lap = wavefront_lap[:, :, 0]

        # median filtering phi
        phi = median_filter(phi, [3, 3], mode='mirror')

        # return phi
        phi = self.M1(np.concatenate([phi[:, :, np.newaxis], phi[:, :, np.newaxis]], -1))
        phi = phi[:, :, 0]
        phi = phi - phi.mean()

        self.A = A
        self.phi = phi
        self.wavefront_lap = wavefront_lap
        
        return A, phi, objvaltotal
    
    def get_field(self, pixel_size=6.45e-6, z=1.43e-3, RI=2.):
        
        scale_factor = pixel_size**2/z
        # RI = 2 # for OPD

        Amp = np.sqrt(self.A* (1 + pixel_size/z*self.wavefront_lap)) # actual amplitude
        OPD = (self.phi/(RI-1)*scale_factor)

        return Amp, OPD
    
    def phase_update_ADMM(self, img, alpha, beta, opt):
        M = [img.shape[0], img.shape[1]]
        L = opt['L']
        N = M + 2*L
        
        # boundary mask
        lap_mat, _ = self.prepare_DCT_basis(N)
        mat_x_hat = (opt['mu'] + beta)*lap_mat + beta*(lap_mat**2)
        mat_x_hat[0, 0] = 1 # why??
        
        # initialization
        x    = np.zeros(N.astype('int'), dtype=np.float64)
        u    = np.zeros([int(N[0]), int(N[1]), 2], dtype=np.float64)
        zeta = np.zeros([int(N[0]), int(N[1]), 2], dtype=np.float64)
        
        # get the matrices
        gt, gx, gy = self.partial_deriv(img)
        
        # pre-compute
        mu  = opt['mu']
        gxy = gx*gy
        gxx = gx**2
        gyy = gy**2
        
        # store in memory at run-time
        denom = mu*(gxx + gyy + mu)
        A11 = (gyy + mu)/denom
        A12 = -gxy/denom
        A22 = (gxx + mu)/denom
        
        # proximal algorithm
        objval = np.zeros([opt['iter'], 1])
        time   = np.zeros([opt['iter'], 1])
        
        # the loop
        for k in range(1, opt['iter'] + 1):
            # x-update
            if k > 1:
                x = self.idct2(self.dct2(mu*self.nablaT(u - zeta))/mat_x_hat)
                
            # pre-compute nabla_x
            nabla_x = self.nabla(x)
            
            # u-update
            u = nabla_x + zeta
            u_temp = self.M1(u)
            
            # only coding simple for now
            # R2 LASSO (in practice, difference between the two solutions
            # are subtle; for speed's sake, the simple version is
            # recommended)
            w_opt_x, w_opt_y = self.R2LASSO_simple(gx, gy, gt, u_temp[:, :, 0], u_temp[:, :, 1], alpha, mu, A11, A12, A22)
            
            # update u
            u[L[0]:-L[0], L[1]:-L[1], 0] = w_opt_x
            u[L[0]:-L[0], L[1]:-L[1], 1] = w_opt_y
            
            # zeta-update
            zeta = zeta + nabla_x - u
            
            # not coding the following now because it is not vital
            # record
            # if opt.isverbose
            #         if k == 1
            #             % define operators
            #             G = @(u) gx.*u(:,:,1) + gy.*u(:,:,2);
                        
            #             % define objective function
            #             obj = @(phi) norm2(G(M1(nabla(phi)))+gt) + ...
            #                 alpha *  norm1(nabla(phi)) + ...
            #                 beta  * (norm2(nabla(phi)) + norm2(nabla2(phi)));
            #         end
            #         objval(k) = obj(x);
            #         time(k)   = toc;
            #         disp(['---- ADMM iter: ' num2str(k) ', obj = ' num2str(objval(k),'%.4e')])
            #     end
            # end
            
        # compute the warped image by Taylor expansion
        w = self.M1(self.nabla(x))
        I_warp = gx*w[:, :, 0] + gy*w[:, :, 1] + img[:, :, 1]
        
        # return masked x
        x = x - x.mean()
        x_full = x
        
        x = self.M1(np.concatenate([x[:, :, np.newaxis], x[:, :, np.newaxis]], -1))
        x = x[:, :, 0]
            
        return x, x_full, I_warp
    
    def R2LASSO_simple(self, a, b, c, ux, uy, alpha, mu, A11, A12, A22):
        # This function attempts to solve the R2 LASSO problem, in a fast but not accurate sense:
        # min (a x + b y + c)^2 + \mu [(x - ux)^2 + (y - uy)^2] + \alpha (|x| + |y|)
        # x,y

        # See also R2LASSO_complete for a more accurate but slower one.
        
        # not coding the following because it is not used
        # if ~exist('A11','var') || ~exist('A12','var') || ~exist('A22','var')
        # denom = 1 ./ ( mu * (a.*a + b.*b + mu) )
        # abcd = 4
        # A11 = denom * (b.*b + mu)
        # A12 = -a.*b .* denom;
        # A22 = denom .* (a.*a + mu);
        # end
        
        # temp
        tx = mu*ux - a*c
        ty = mu*uy - b*c

        # l2 minimum
        x0_x = A11*tx + A12*ty
        x0_y = A12*tx + A22*ty
        
        # 2. get the sign of optimal
        sign_x = np.sign(x0_x)
        sign_y = np.sign(x0_y)

        # 3. get the optimal
        x = x0_x - alpha/2 * (A11 * sign_x + A12 * sign_y)
        y = x0_y - alpha/2 * (A12 * sign_x + A22 * sign_y)
        
        # 4. check sign and map to the range
        x[np.sign(x) != sign_x] = 0
        y[np.sign(y) != sign_y] = 0
        
        return x, y

    def obj_total(self, A, b, phi, alpha, beta, gamma, tau, I0):
        return self.norm2(I0*A - b) + alpha*self.norm1(self.nabla(phi)) + beta*self.norm2(self.K(phi)) + \
            gamma*self.norm1(self.K(A)) + tau*self.norm2(self.K(A))

    def disp_obj(self, s, i, objval):
        print(f'{s}{i}, obj = {objval:.4e}')
        
    def dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def partial_deriv(self, images):
        # derivative kernel
        h = np.array([[1, -8, 0, 8, -1]])/12

        # temporal gradient
        It = images[:, :, 1] - images[:, :, 0]
        
        # First compute derivative then warp
        Ix = scs.correlate2d(images[:, :, 1], h, mode='same', boundary='symm') # replicate/nearest in matlab is not an option here. difference should only be at the edges.
        Iy = scs.correlate2d(images[:, :, 1], h.transpose(), mode='same', boundary='symm')
        
        return It, Ix, Iy
    
    def nabla(self, phi):
        # apply x_k and y_k to phi, concat and return the results
        assert phi.ndim == 2, "Dimensions of phi must be 2"
        
        temp1 = scs.correlate2d(phi, self.x_k, mode='same', boundary='symm')
        temp2 = scs.correlate2d(phi, self.y_k, mode='same', boundary='symm')
        
        return np.concatenate([temp1[:, :, np.newaxis], temp2[:, :, np.newaxis]], -1)


    def nabla2(self, phi):
        # apply l_k to phi and return the result
        assert phi.ndim == 2, "Dimensions of phi must be 2"
        
        return scs.correlate2d(phi, self.l_k, mode='same', boundary='symm')


    def nablaT(self, adj_phi):
        # apply rotated x_k and y_k to first and second channels of adj_phi and add the two and return 
        assert adj_phi.ndim == 3, "Dimensions of phi must be 3"
        
        return scs.correlate2d(adj_phi[:, :, 0], np.flip(self.x_k, [0, 1]), mode='same', boundary='symm') + \
            scs.correlate2d(adj_phi[:, :, 1], np.flip(self.y_k, [0, 1]), mode='same', boundary='symm')


    def K(self, x): 
        return np.concatenate([self.nabla(x), self.nabla2(x)[:, :, np.newaxis]], -1)

    def KT(self, u):
        assert u.ndim == 3, "Dimensions of phi must be 3"
        
        return self.nablaT(u[:, :, :2]) + self.nabla2(u[:, :, 2])


    def prepare_DCT_basis(self, M):
        # prepare DCT basis of size M
        assert len(M) == 2, "Length of M must be 2"
        
        H, W = int(M[0]), int(M[1])
        x_coord, y_coord = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        
        # normalized the width and height so that range of x and y in [0, 1]
        # 4 - 2*cos(pi*x) - 2*cos(pi*y)
        lap_mat = 4 - 2*np.cos(np.pi*x_coord/W) - 2*np.cos(np.pi*y_coord/H) 
        K_mat = lap_mat*(lap_mat + 1)
        
        return lap_mat, K_mat

    # define proximal operator of A
    def prox_A(self, u, gamma_new, opt_A):
        temp = np.abs(u) - gamma_new/(2*opt_A['mu_A'])
        return np.sign(u)*np.where(temp > 0, temp, 0)

    # define norms
    def norm2(self, x): 
        return (np.abs(x.ravel())**2).sum()
    
    def norm1(self, x): 
        return np.abs(x.ravel()).sum()

    # define objective functions
    def obj_A(self, A, b, gamma_new, tau_new):
        return self.norm2(A - b/I0) + gamma_new*self.norm1(self.K(A)) + tau_new*self.norm2(self.K(A))
    
    # boundary mask
    def M1(self, u): 
        return u[self.L[0]:-self.L[0], self.L[1]:-self.L[1], :]
        
    def M2(self, u): 
        return u[self.L[0]:-self.L[0], self.L[1]:-self.L[1]]
    
    
# 3rd order polynomoial fit through linear regession
def linearization(theta):
    
    theta = np.array(theta)
    
    X, Y = np.meshgrid(np.linspace(-1, 1, theta.shape[1]), np.linspace(-1, 1, theta.shape[0]))
    A = np.array([X.T.ravel(), Y.T.ravel(), 
                  X.T.ravel()*Y.T.ravel(), X.T.ravel()*X.T.ravel(), Y.T.ravel()*Y.T.ravel(), 
                  np.ones_like(X.T.ravel()), 
                  (X.T.ravel())**3, (Y.T.ravel())**3, X.T.ravel()*(Y.T.ravel())**2, Y.T.ravel()*(X.T.ravel())**2
                  ], dtype=np.float64).T

    a = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), theta.T.ravel())

    return (A@a).reshape(theta.shape[1], theta.shape[0]).T