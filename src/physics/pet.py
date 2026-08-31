import torch
import numpy as np
import parallelproj
import array_api_compat.torch as xp
from parallelproj import Array
from copy import copy
from tqdm.auto import tqdm


class PET2D:
    
    def __init__(self, ct, device='cpu'):
        # Assume ct is in range [-1, 1] and has shape (1, 1, 256, 256)

        self.device = device

        self.scanner = parallelproj.RegularPolygonPETScannerGeometry(
            xp,
            device,
            radius=400.0,
            num_sides=12,
            num_lor_endpoints_per_side=50,
            lor_spacing=4.14,
            ring_positions=np.array([0.0]),
            symmetry_axis=1,
        )

        self.lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
            self.scanner,
            sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
            max_ring_difference=3,
            radial_trim=10,
        )

        self.proj = parallelproj.RegularPolygonPETProjector(
            self.lor_desc, img_shape=(256, 1, 256), voxel_size=(2.0, 2.0, 2.0)
        )
        
        correction = self.att_correction(ct.to(device))
        self.att_sino = torch.exp(-self.proj(correction.squeeze(0).permute(1, 0, 2)))
        self.att_op = parallelproj.operators.ElementwiseMultiplicationOperator(self.att_sino)
        self.proj_with_att = parallelproj.CompositeLinearOperator((self.att_op, self.proj))

    def att_correction(self, ct):
        # Assume ct is in range [-1, 1]
        ct_hu = 1000 * ct
        correction = torch.zeros_like(ct_hu)
        correction[ct_hu < 300] = 0.0096 * (ct_hu[ct_hu < 300] + 1000) / 1000
        correction[ct_hu >= 300] = 0.0081 * (ct_hu[ct_hu >= 300] + 1000) / 1000
        return correction

    def A(self, x):
        if x.ndim == 4:
            x = x.squeeze(0)
        return self.proj_with_att(x.permute(1, 0, 2))


    def A_T(self, y):
        return self.proj_with_att.adjoint(y).permute(1, 0, 2).unsqueeze(0)
    

    def OSEM(self, y, num_subsets=10, num_iter=None):
        '''
        Implementation taken from the tutorial in the parallelproj documentation:
        https://parallelproj.readthedocs.io/en/stable/auto_examples/05_algorithms/04_run_osem_projection_data.html
        '''
        
        subset_views, subset_slices = self.proj.lor_descriptor.get_distributed_views_and_slices(
            num_subsets, len(self.proj.out_shape)
        )

        _, subset_slices_non_tof = self.proj.lor_descriptor.get_distributed_views_and_slices(
            num_subsets, 3
        )

        # clear the cached LOR endpoints since we will create many copies of the projector
        self.proj.clear_cached_lor_endpoints()
        pet_subset_linop_seq = []

        # we setup a sequence of subset forward operators each constisting of
        # (1) image-based resolution model
        # (2) subset projector
        # (3) multiplication with the corresponding subset of the attenuation sinogram
        for i in range(num_subsets):

            # make a copy of the full projector and reset the views to project
            subset_proj = copy(self.proj)
            subset_proj.views = subset_views[i]

            if subset_proj.tof:
                subset_att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
                    subset_proj.out_shape, self.att_sino[subset_slices_non_tof[i]]
                )
            else:
                subset_att_op = parallelproj.ElementwiseMultiplicationOperator(
                    self.att_sino[subset_slices_non_tof[i]]
                )

            # add the resolution model and multiplication with a subset of the attenuation sinogram
            pet_subset_linop_seq.append(
                parallelproj.CompositeLinearOperator(
                    [
                        subset_att_op,
                        subset_proj,
                    ]
                )
            )

        pet_subset_linop_seq = parallelproj.LinearOperatorSequence(pet_subset_linop_seq)


        def em_update(
            x_cur: Array,
            data: Array,
            op: parallelproj.LinearOperator,
            s: Array,
            adjoint_ones: Array,
        ) -> Array:
            """EM update

            Parameters
            ----------
            x_cur : Array
                current solution
            data : Array
                data
            op : parallelproj.LinearOperator
                linear forward operator
            s : Array
                contamination
            adjoint_ones : Array
                adjoint of ones

            Returns
            -------
            Array
            """
            ybar = op(x_cur) + s
            return x_cur * op.adjoint(data / ybar.clamp(1e-6)) / adjoint_ones



        num_iter = 20 // len(pet_subset_linop_seq) if num_iter is None else num_iter

        # initialize x
        pet_lin_op = parallelproj.CompositeLinearOperator((self.proj,))
        x = xp.ones(pet_lin_op.in_shape, dtype=xp.float64, device=self.device)

        # calculate A_k^H 1 for all subsets k
        subset_adjoint_ones = [
            x.adjoint(xp.ones(x.out_shape, dtype=xp.float64, device=self.device))
            for x in pet_subset_linop_seq
        ]

        contamination = xp.zeros(pet_lin_op.out_shape, dtype=xp.float64, device=self.device)

        # OSEM iterations
        for i in tqdm(range(num_iter)):
            for k, sl in enumerate(subset_slices):
                x = em_update(
                    x, y[sl], pet_subset_linop_seq[k], contamination[sl], subset_adjoint_ones[k]
                )

        return x