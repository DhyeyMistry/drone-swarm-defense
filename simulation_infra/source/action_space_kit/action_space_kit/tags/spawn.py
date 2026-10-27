"""Put the mapped AprilTags (and the room walls they hang on) into a USD stage.

Each tag is a quad textured with its tag36h11 image (including the white quiet zone), in
the tag's own frame: image right = tag +x, image up = tag +y, visible face toward +z. A
thin dark board sits behind it so it reads as a printed sign from any side. Visual only:
no colliders, so they cost nothing in PhysX. Spawn under ``/World/envs/env_0`` before the
scene is cloned and every env gets a copy.
"""

from __future__ import annotations

from dataclasses import dataclass

from action_space_kit.tags.room_map import Tag, tag_quat_wxyz
from action_space_kit.tags.textures import texture_path


@dataclass
class Wall:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float] = (0.78, 0.76, 0.72)


def _preview_material(stage, path: str, color=None, texture: str | None = None):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if texture is None:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    else:
        reader = UsdShade.Shader.Define(stage, f"{path}/StReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _place(xform_prim, pos, quat_wxyz):
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    w, x, y, z = quat_wxyz
    xf.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))


def spawn_tag(stage, root: str, tag: Tag, board_thickness: float = 0.006) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    path = f"{root}/tag_{tag.tag_id}"
    xform = UsdGeom.Xform.Define(stage, path)
    _place(xform.GetPrim(), tag.pos, tag_quat_wxyz(tag))

    h = 0.5 * tag.printed_size_m
    quad = UsdGeom.Mesh.Define(stage, f"{path}/face")
    lift = 0.0015  # just proud of the board
    quad.CreatePointsAttr([Gf.Vec3f(-h, -h, lift), Gf.Vec3f(h, -h, lift), Gf.Vec3f(h, h, lift), Gf.Vec3f(-h, h, lift)])
    quad.CreateFaceVertexCountsAttr([4])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])  # counter-clockwise seen from +z
    quad.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    quad.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    quad.CreateDoubleSidedAttr(False)
    st = UsdGeom.PrimvarsAPI(quad).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    mat = _preview_material(stage, f"{path}/face_mat", texture=str(texture_path(tag.tag_id)))
    UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(mat)

    board = UsdGeom.Cube.Define(stage, f"{path}/board")
    board.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(board)
    bxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.5 * board_thickness))
    bxf.AddScaleOp().Set(Gf.Vec3f(2 * h + 0.02, 2 * h + 0.02, board_thickness))
    bmat = _preview_material(stage, f"{path}/board_mat", color=(0.15, 0.15, 0.17))
    UsdShade.MaterialBindingAPI.Apply(board.GetPrim()).Bind(bmat)


def spawn_wall(stage, root: str, wall: Wall) -> None:
    from pxr import Gf, UsdGeom, UsdShade

    cube = UsdGeom.Cube.Define(stage, f"{root}/{wall.name}")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*wall.center))
    xf.AddScaleOp().Set(Gf.Vec3f(*wall.size))
    mat = _preview_material(stage, f"{root}/{wall.name}_mat", color=wall.color)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)


def spawn_room(root: str, tags: dict[int, Tag], walls: list[Wall]) -> None:
    """Spawn every tag and wall under ``root`` on the current stage."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, root)
    for wall in walls:
        spawn_wall(stage, root, wall)
    for tag in tags.values():
        spawn_tag(stage, root, tag)
