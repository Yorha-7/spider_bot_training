"""A procedural flat terrain that does not depend on NVIDIA-hosted USD assets."""

from pxr import UsdGeom, UsdPhysics

from isaaclab.sim.utils import bind_physics_material, get_current_stage
from isaaclab.terrains import TerrainImporter


class LocalGroundTerrainImporter(TerrainImporter):
    """Create the flat collision plane directly in the active USD stage.

    Isaac Lab's standard ``GroundPlaneCfg`` references
    ``default_environment.usd`` on NVIDIA's asset server.  A locomotion task
    only needs a flat collision surface, so generate one locally instead.
    """

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """Create a Z-up collision plane and bind the configured physics material."""
        path = f"{self.cfg.prim_path}/{name}"
        self.terrain_prim_paths.append(path)

        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, path)
        plane = UsdGeom.Plane.Define(stage, f"{path}/Plane")
        plane.CreateAxisAttr("Z")
        plane.CreateWidthAttr(size[0])
        plane.CreateLengthAttr(size[1])
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

        material_path = f"{path}/physicsMaterial"
        self.cfg.physics_material.func(material_path, self.cfg.physics_material)
        bind_physics_material(str(plane.GetPath()), material_path, stage=stage)
