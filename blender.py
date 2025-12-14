import bpy
import numpy as np
from math import radians
from mathutils import Vector

# constants
FPS = 60
TOTAL_TIME = 1.5
TOTAL_FRAMES = FPS * TOTAL_TIME

# delete all objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj, do_unlink=True)
for mesh in bpy.data.meshes:
    bpy.data.meshes.remove(mesh, do_unlink=True)

# material
mat = bpy.data.materials.new(name="Water")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)  # dark gray
bsdf.inputs["Roughness"].default_value = 0.0
bsdf.inputs["Roughness"].default_value = 0.0
bsdf.inputs[3].default_value = 1.33 # IOR
bsdf.inputs[18].default_value = 1.0 # Transmission

def visible_on_frame(obj,frame_visible):
    # make object visible on a single frame
    frame_before = frame_visible - 1
    frame_after = frame_visible + 1
    # hide before
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_before)
    obj.keyframe_insert(data_path="hide_render", frame=frame_before)
    # show at the target frame
    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_visible)
    obj.keyframe_insert(data_path="hide_render", frame=frame_visible)
    # hide after
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_after)
    obj.keyframe_insert(data_path="hide_render", frame=frame_after)

# define deformation
def f(XYZ,t=0):
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]
    x = (1 + 0.75*t)*X
    y = Y
    z = (1 - 0.25*t)*Z + 0.35*t*X**2 - 1.5*t
    return np.array([x, y, z])

# animate
for frame in range(1,int(TOTAL_FRAMES)+1):
    t = frame / TOTAL_FRAMES
    
    bpy.ops.mesh.primitive_uv_sphere_add(radius=4)
    obj = bpy.context.active_object
    
    subsurf = obj.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.levels = 2  # Viewport subdivisions
    subsurf.render_levels = 2  # Render subdivisions

    mesh = obj.data
    reference = np.array([v.co[:] for v in mesh.vertices])

    for i, v in enumerate(mesh.vertices):
        v.co = Vector(f(reference[i],t))
        
    for face in mesh.polygons:
        face.use_smooth = True
        
    if mat is not None:
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    else:
        print("Material not found!")
    
    visible_on_frame(obj,frame)

# add camera
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, -40, 0), rotation=(radians(90), 0, 0), scale=(1, 1, 1))
