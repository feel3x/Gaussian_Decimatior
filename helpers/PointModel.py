import torch
import numpy as np
from plyfile import PlyData, PlyElement
import os


class PointModel:
    def __init__(self, sh_degree: int = None):
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.opacity_activation = torch.sigmoid
        def inverse_sigmoid(x):
            return torch.log(x/(1-x))
        self.inverse_opacity_activation = inverse_sigmoid



    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def save_ply(self, path: str):
        """Save model parameters to a PLY file (matching original format)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attr, "f4") for attr in self.construct_save_list()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def construct_save_list(self):
        """Generate attribute names in the same order as save_ply concatenation."""
        attributes = ["x", "y", "z"]
        attributes += ["nx", "ny", "nz"]  # normals
        # f_dc and f_rest: flatten over channels
        n_dc = self._features_dc.shape[1] * self._features_dc.shape[2]
        n_rest = self._features_rest.shape[1] * self._features_rest.shape[2]
        attributes += [f"f_dc_{i}" for i in range(n_dc)]
        attributes += [f"f_rest_{i}" for i in range(n_rest)]
        attributes += ["opacity"]
        attributes += [f"scale_{i}" for i in range(self._scaling.shape[1])]
        attributes += [f"rot_{i}" for i in range(self._rotation.shape[1])]
        return attributes

    def load_ply(self, path):
            plydata = PlyData.read(path)

            if(self.max_sh_degree == None):
                self.max_sh_degree = self.get_sh_bands_from_plydata(plydata)

            vertex = plydata.elements[0]

            # xyz
            xyz = np.stack(
                [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])],
                axis=1,
            )

            # opacity
            opacities = np.asarray(vertex["opacity"])[..., np.newaxis]

            # features_dc (first 3 SH coefficients)
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(vertex["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(vertex["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(vertex["f_dc_2"])

            # features_rest
            extra_f_names = [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3

            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(vertex[attr_name])
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
            )

            # scaling
            scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(vertex[attr_name])

            # rotation
            rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(vertex[attr_name])

            # Convert to Parameters
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._xyz = torch.tensor(xyz, dtype=torch.float, device=device)
            self._features_dc = torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous()
            self._features_rest = torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous()
            self._opacity = torch.tensor(opacities, dtype=torch.float, device=device)
            self._scaling = torch.tensor(scales, dtype=torch.float, device=device)
            self._rotation = torch.tensor(rots, dtype=torch.float, device=device)

            self.active_sh_degree = self.max_sh_degree

    def get_sh_bands_from_plydata(self, plydata):
        """
        Returns the number of spherical harmonics (SH) bands in a Gaussian Splatting .PLY file.
        """
        try:
            #Get the vertex element
            if 'vertex' not in plydata:
                raise ValueError("PLY file does not contain vertex data")
            
            vertex = plydata['vertex']
            
            #Count SH coefficient properties     
            sh_properties = []
            
            # Handle different PLY file structures
            if hasattr(vertex, 'dtype') and hasattr(vertex.dtype, 'names') and vertex.dtype.names:
                property_names = vertex.dtype.names
            elif hasattr(vertex, 'data') and len(vertex.data) > 0:
                # Try to get property names from the first data element
                property_names = vertex.data[0].dtype.names if hasattr(vertex.data[0], 'dtype') else []
            elif hasattr(vertex, 'properties'):
                # Alternative: get from properties if available
                property_names = [prop.name for prop in vertex.properties]
            else:
                raise ValueError("Cannot determine property names from PLY vertex data")
            
            # Look for SH-related properties
            for prop_name in property_names:
                if prop_name.startswith('f_dc_') or prop_name.startswith('f_rest_'):
                    sh_properties.append(prop_name)
            
            if not sh_properties:
                #No SH coefficients found
                return 0
            
            #Count DC components (band 0)
            dc_count = len([name for name in sh_properties if name.startswith('f_dc_')])
            
            #Count rest components (bands 1+)
            rest_count = len([name for name in sh_properties if name.startswith('f_rest_')])
            
            #Total SH coefficients
            total_sh_coeffs = dc_count + rest_count
            
            #Calculate number of bands
            #SH coefficients per band: band 0 = 1, band 1 = 3, band 2 = 5, band 3 = 7, etc.
            #Total coeffs = 1 + 3 + 5 + ... + (2*n+1) for n bands
            #This equals (n+1)^2 total coefficients for n+1 bands (0 to n)
            
            #Gaussian Splatting, we typically have 3 color channels (RGB)
            #So we need to divide by 3 first
            if total_sh_coeffs % 3 != 0:
                raise ValueError(f"Invalid number of SH coefficients: {total_sh_coeffs} (not divisible by 3)")
            
            coeffs_per_channel = total_sh_coeffs // 3
            
            #Find the number of bands
            #coeffs_per_channel = (max_band + 1)^2
            #So max_band = sqrt(coeffs_per_channel) - 1
            max_band = int(np.sqrt(coeffs_per_channel)) - 1
            
            # Verify
            expected_coeffs = (max_band + 1) ** 2
            if expected_coeffs != coeffs_per_channel:
                raise ValueError(f"Invalid SH coefficient count: {coeffs_per_channel} per channel doesn't match any valid band configuration")
            
            print("SH Degree: "+ str(max_band))
            return max_band  # Return number of bands (0-indexed max_band + 1)       
            
        except Exception as e:
            raise ValueError(f"Error reading PLY file: {str(e)}")

