import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from collections import deque
import cv2
import os
import matplotlib.pyplot as plt

from ...base import BaseModule
from ....utils.rendering import Camera


class FixedEyeBlurry(BaseModule):

  def __init__(self, model, data, bm_model, resolution, pos, quat, body="worldbody", channels=None, buffer=None,
               **kwargs):
    """
    A simple eye model using a fixed camera.

    Args:
      model: A MjModel object of the simulation
      data: A MjData object of the simulation
      bm_model: A biomechanical model class object inheriting from BaseBMModel
      resolution: Resolution in pixels [width, height]
      pos: Position of the camera [x, y, z]
      quat: Orientation of the camera as a quaternion [w, x, y, z]
      body (optional): Body to which the camera is attached, default is 'worldbody'
      channels (optional): Which channels to use; 0-2 refer to RGB, 3 is depth. Default value is None, which means that all channels are used (i.e. same as channels=[0,1,2,3])
      buffer (optional): Defines a buffer of given length (in seconds) that is utilized to include prior observations
      **kwargs (optional): Keyword args that may be used

    Raises:
      KeyError: If "rendering_context" is not given (included in kwargs)
    """
    super().__init__(model, data, bm_model, **kwargs)

    self._model = model
    self._data = data

    # Probably already called
    mujoco.mj_forward(self._model, self._data)

    # Set camera specs
    if channels is None:
      channels = [0, 1, 2, 3]
    self._channels = channels
    self._resolution = resolution
    self._pos = pos
    self._quat = quat
    self._body = body

    #limit of images
    self.obs_save_limit = 10
    self.obs_save_number = 0

    # Get rendering context
    if "rendering_context" not in kwargs:
      raise KeyError("rendering_context must be defined")
    self._context = kwargs["rendering_context"]

    # Initialise camera
    self.camera_fixed_eye = Camera(context=self._context, model=model, data=data,
                          resolution=resolution, rgb=True, depth=True, camera_id="fixed-eye")
    self._camera_active = True

    # Append all cameras to self._cameras to be able to display
    # their outputs in human-view/GUI mode (used by simulator.py)
    self._cameras.append(self.camera_fixed_eye)

    # Define a vision buffer for including previous visual observations
    self._buffer = None
    if buffer is not None:
      assert "dt" in kwargs, "dt must be defined in order to include prior observations"
      maxlen = 1 + int(buffer/kwargs["dt"])
      self._buffer = deque(maxlen=maxlen)

  @staticmethod
  def insert(simulation, **kwargs):

    assert "pos" in kwargs, "pos needs to be defined for this perception module"
    assert "quat" in kwargs, "quat needs to be defined for this perception module"

    # Get simulation root
    simulation_root = simulation.getroot()

    # Add assets
    simulation_root.find("asset").append(ET.Element("mesh", name="eye", scale="0.05 0.05 0.05",
                                              file="assets/basic_eye_2.stl"))
    simulation_root.find("asset").append(ET.Element("texture", name="blue-eye", type="cube", gridsize="3 4",
                                              gridlayout=".U..LFRB.D..",
                                              file="assets/blue_eye_texture_circle.png"))
    simulation_root.find("asset").append(ET.Element("material", name="blue-eye", texture="blue-eye", texuniform="true"))

    # Create eye
    eye = ET.Element("body", name="fixed-eye", pos=kwargs["pos"], quat=kwargs["quat"])
    eye.append(ET.Element("geom", name="fixed-eye", type="mesh", mesh="eye", euler="0.69 1.43 0",
                          material="blue-eye", size="0.025", rgba="1.0 1.0 1.0 1.0"))
    eye.append(ET.Element("camera", name="fixed-eye", fovy="90"))

    # Add eye to a body
    body = kwargs.get("body", "worldbody")
    if body == "worldbody":
      simulation_root.find("worldbody").append(eye)
    else:
      eye_body = simulation_root.find(f".//body[@name='{body}'")
      assert eye_body is not None, f"Body with name {body} was not found"
      eye_body.append(eye)

  def get_observation(self, model, data, info=None):

    # Get rgb and depth arrays
    rgb, depth = self.camera_fixed_eye.render()
    assert not np.all(rgb==0), "There's still something wrong with rendering"

    # Normalise
    depth = (depth - 0.5) * 2
    rgb = (rgb / 255.0 - 0.5) * 2

    # Transpose channels
    obs = np.transpose(np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2), [2, 0, 1])

    # Apply blurring to the observation (add blurring here)
    obs = self.blur_observation(obs)

    # Choose channels
    obs = obs[self._channels, :, :]

    # Include prior observation if needed
    if self._buffer is not None:
      # Update buffer
      if len(self._buffer) > 0:
        self._buffer.pop()
      while len(self._buffer) < self._buffer.maxlen:
        self._buffer.appendleft(obs)

      # Use latest and oldest observation, and their difference
      obs = np.concatenate([self._buffer[0], self._buffer[-1], self._buffer[-1] - self._buffer[0]], axis=0)

    return obs

  @property
  def camera_active(self):
    return self._camera_active

  @property
  def _default_encoder(self):
    return {"module": "rl.encoders", "cls": "SmallCNN", "kwargs": {"out_features": 256}}

  def _reset(self, model, data):
    if self._buffer is not None:
      self._buffer.clear()

  # @property
  # def encoder(self):
  #   return small_cnn(observation_shape=self._observation_shape, out_features=256)

  # Should perhaps create a base class for vision modules and define an abstract render function
  def render(self):
    return self.camera_fixed_eye.render()



  def blur_observation(self, obs, kernel_size=(5, 5), sigma=1.0):
    """
    Applies Gaussian blur to the RGB and depth channels of the observation.

    Args:
        obs: The observation to be blurred, expected to be of shape [channels, height, width].
        kernel_size: The size of the Gaussian kernel (width, height).
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The blurred observation.
    """
    # Split RGB and depth channels
    rgb = obs[0:3, :, :]  # RGB channels
    depth = obs[3, :, :]  # Depth channel

    # Convert RGB to [height, width, channels] for OpenCV
    rgb = np.transpose(rgb, (1, 2, 0))  # Shape: [height, width, 3]

    # Convert depth to [height, width] (OpenCV expects 2D for single-channel images)
    depth = depth.squeeze()  # Shape: [height, width]

    # Apply Gaussian blur to RGB and depth
    rgb_blurred = cv2.GaussianBlur(rgb, kernel_size, sigma)  # Shape: [height, width, 3]
    depth_blurred = cv2.GaussianBlur(depth, kernel_size, sigma)  # Shape: [height, width]

    # Convert back to [channels, height, width]
    rgb_blurred = np.transpose(rgb_blurred, (2, 0, 1))  # Shape: [3, height, width]
    depth_blurred = np.expand_dims(depth_blurred, axis=0)  # Shape: [1, height, width]

    # Combine RGB and depth
    blurred_obs = np.concatenate([rgb_blurred, depth_blurred], axis=0)  # Shape: [4, height, width]
    self.save_observation(obs)
    self.save_observation(blurred_obs)
    return blurred_obs

  def save_observation(self, obs):
      if self.obs_save_number >= self.obs_save_limit:
          return
      # Transpose from (channels, height, width) -> (height, width, channels)
      obs = np.transpose(obs, (1, 2, 0))
      # If there are 4 channels (RGB + Depth), remove the depth channel
      if obs.shape[-1] == 4:
          obs = obs[:, :, :3]  # Keep only the first 3 channels (RGB)
      print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")  # Debugging print
      
      if obs.dtype in [np.float32, np.float64]:  # Check if it's a float image
          obs = np.clip(obs, 0, 1)  # Ensure values are within 0-1 range
      script_directory = os.path.dirname(os.path.abspath(__file__))
      path = os.path.join(script_directory, 'obs_dir')
      name = f"obs_{self.obs_save_number}.png"
      path_obs = os.path.join(path, name)
      if not os.path.exists(path):
        os.makedirs(path)
      plt.imsave(path_obs, obs)
      self.obs_save_number += 1
      return