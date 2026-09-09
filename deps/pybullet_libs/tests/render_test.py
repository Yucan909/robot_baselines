import pybullet as p
import pybullet_data

from rpad.pybullet_libs.camera import Camera


def test_simple_render():
    # Create a pybullet environment.
    client_id = p.connect(p.DIRECT)

    # Load in a plane.
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Load in a simple R2D2.
    r2d2 = p.loadURDF("r2d2.urdf", [0, 0, 1])

    # Create a camera.
    camera = Camera(pos=[2, 2, 1])

    # Render the camera.
    render = camera.render(client_id=client_id)

    p.disconnect(client_id)
