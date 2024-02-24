import math
import os.path as osp

import bpy
import trimesh
from sapien.wrapper.coacd import do_coacd

"""
mamba create -n "blender" "python==3.10"
mamba activate blender
pip install bpy trimesh ipdb
"""


def fix_cabinet(filename, threshold=64):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)
    osp.dirname(filename)
    bpy.ops.wm.obj_import(filepath=filename)
    # mesh = trimesh.load_mesh(file_obj=filename)
    for i in range(len(bpy.data.objects)):
        obj = bpy.data.objects[i]
        dims = obj.dimensions
        # if relative dimensions are above some threshold
        comparison_tuples = [
            (dims.x, dims.y, "xy"),
            (dims.x, dims.z, "xz"),
            (dims.y, dims.z, "yz"),
        ]
        must_decompose = False
        for d1, d2, plane in comparison_tuples:
            if d1 > d2:
                factor = d1 / d2
            else:
                factor = d2 / d1
            if factor > threshold:
                print(obj.name, "will cause issues on the GPU!", f"factor={factor}")
                must_decompose = True
                break
        if must_decompose:
            bpy.ops.object.editmode_toggle()
            obj.select_set(True)
            bpy.ops.mesh.select_all(action="SELECT")
            num_cuts = int(math.ceil(factor / threshold))
            bpy.ops.mesh.subdivide(number_cuts=num_cuts - 1)
            # subdivide long edhes, triangulate, then separate?
            bpy.ops.object.editmode_toggle()


if __name__ == "__main__":
    # for file in os.walk("data/partnet_mobility/dataset"):
    #     for model_id in file[1]:
    #         print(model_id)
    #     break

    fix_cabinet("data/partnet_mobility/dataset/1054/cvx_objs/link_2.obj")
    # do_coacd("data/partnet_mobility/dataset/1054/cvx_objs/link_2_convex_1.obj")
