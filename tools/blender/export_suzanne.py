import bpy
import os

output = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets/models/suzanne.glb"))
os.makedirs(os.path.dirname(output), exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0))
suzanne = bpy.context.active_object
suzanne.name = "Suzanne"
bpy.ops.object.shade_smooth()

bpy.ops.wm.save_as_mainfile(filepath=os.path.splitext(output)[0] + ".blend")
bpy.ops.export_scene.gltf(filepath=output, export_format="GLB", use_selection=True, export_normals=True)
print("Exported", output)
