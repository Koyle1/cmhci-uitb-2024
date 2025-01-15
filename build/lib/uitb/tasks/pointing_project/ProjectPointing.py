import numpy as np
import mujoco
import math #added for sinus and cosinus calculations
import pandas as pd #added for data management
import os #added for saving the file

from .reward_functions import NegativeExpDistanceWithHitBonus
from ..base import BaseTask

class ProjectPointing(BaseTask):

  #Data Management / added by me
  #--------------------------------------------------------------------------------
  def _set_start_data(self, position):
      if self.iteration == 0 or self.iteration == (self._max_trials+1):
          return
      self.df.loc[self.iteration, 'start_x'] = position[0]
      self.df.loc[self.iteration, 'start_y'] = position[1]
      self.df.loc[self.iteration, 'start_z'] = position[2]
      

  def _set_end_data_location(self, position, size):
      if self.iteration == 0 or self.iteration == (self._max_trials+1):
          return
      self.df.loc[self.iteration, 'timestamp_start'] = pd.Timestamp.now()
      self.df.loc[self.iteration, 'end_x'] = position[0]
      self.df.loc[self.iteration, 'end_y'] = position[1]
      self.df.loc[self.iteration, 'end_z'] = position[2]
      self.df.loc[self.iteration, 'target_size'] = size

  def _set_end_data_target_hit_time(self):
      if self.iteration == 0 or self.iteration == (self._max_trials+1):
          return
      self.df.loc[self.iteration, 'timestamp_end'] = pd.Timestamp.now()


  def _dump_data(self):
      # How many files are in the directory / for naming the file
      print('Dumping data...')
      '''
      script_directory = os.path.dirname(os.path.abspath(__file__))
      collection_dir = os.path.join(script_directory, 'CollectedData')
      if not os.path.exists(collection_dir):
        os.makedirs(collection_dir)
      all_items = os.listdir(collection_dir)
      file_count = sum(1 for item in all_items if os.path.isfile(os.path.join(collection_dir, item)))
      num = file_count
      name = 'data' + str(num) + '.csv'
      
      # Full path to save the CSV file in the script directory
      file_path = os.path.join(collection_dir, name)
      
      # Save the DataFrame to the specified path
      self.df.to_csv(file_path)
      print('Dumped data as ' + name + '!')  
      '''
      self.iteration = 0
      self.df = self.df.drop(self.df.index)
      
  #--------------------------------------------------------------------------------

  def __init__(self, model, data, end_effector, shoulder, **kwargs):
    super().__init__(model, data, **kwargs)
     
    #For collecting data / added by me
    #------------------------------------------------------------------------
    self.df = pd.DataFrame(columns=['timestamp_start', 'start_x', 'start_y','start_z', 'timestamp_end', 'end_x', 'end_y', 'end_z', 'target_size'])
    self.iteration = 0
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------
    self._ball_status = 0
    #------------------------------------------------------------------------
      
    # This task requires an end-effector to be defined
    if not isinstance(end_effector, list) and len(end_effector) != 2:
      raise RuntimeError("'end_effector' must be a list with two elements: first defining what type of mujoco element "
                         "it is, and second defining the name")
    self._end_effector = end_effector

    # Also a shoulder that is used to define the location of target plane
    if not isinstance(shoulder, list) and len(shoulder) != 2:
      raise RuntimeError("'shoulder' must be a list with two elements: first defining what type of mujoco element "
                         "it is, and second defining the name")
    self._shoulder = shoulder

    # Use early termination if target is not hit in time
    self._steps_since_last_hit = 0
    self._max_steps_without_hit = self._action_sample_freq*4

    # Used for logging states
    self._info = {"target_hit": False, "inside_target": False, "target_spawned": False, "terminated": False,
                  "truncated": False, "termination": False}

    # Define a maximum number of trials (if needed for e.g. evaluation / visualisation)
    self._trial_idx = 0
    self._max_trials = kwargs.get('max_trials', 40)
    self._targets_hit = 0

    # Dwelling based selection -- fingertip needs to be inside target for some time
    self._steps_inside_target = 0
    self._dwell_threshold = int(0.5*self._action_sample_freq)  #for HRL: int(0.25*self._action_sample_freq)

    # Radius limits for target
    self._target_radius_limit = kwargs.get('target_radius_limit', np.array([0.05, 0.15]))
    self._target_radius = self._target_radius_limit[0]  #for HRL: self._target_radius_limit[1]

    # Minimum distance to new spawned targets is twice the max target radius limit
    self._new_target_distance_threshold = 2*self._target_radius_limit[1]

    # Define a default reward function
    #if self.reward_function is None:
    self._reward_function = NegativeExpDistanceWithHitBonus(k=10)

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Define plane where targets will move: 0.55m in front of and 0.1m to the right of shoulder, or the "humphant" body.
    # Note that this body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    self._target_origin = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos + np.array([0.55, -0.1, 0])
    self._target_position = self._target_origin.copy()
    self._target_limits_y = np.array([-0.3, 0.3])
    self._target_limits_z = np.array([-0.3, 0.3])
    
    # For LLC policy  #TODO: remove?
    self._target_qpos = [0,0,0,0,0]
    
    # Update plane location
    model.geom("target-plane").size = np.array([0.005,
                                                (self._target_limits_y[1] - self._target_limits_y[0])/2,
                                                (self._target_limits_z[1] - self._target_limits_z[0])/2])
    model.body("target-plane").pos = self._target_origin

    # Set camera angle TODO need to rethink how cameras are implemented
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([1.1, -0.9, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])
    #model.cam_pos[model.camera_name2id('for_testing')] = np.array([-0.8, -0.6, 1.5])
    #model.cam_quat[model.camera_name2id('for_testing')] = np.array(
    #  [0.718027, 0.4371043, -0.31987, -0.4371043])

  def _update(self, model, data):

    # Set some defaults
    terminated = False
    truncated = False
    self._info["target_spawned"] = False

    # Get end-effector position
    ee_position = getattr(data, self._end_effector[0])(self._end_effector[1]).xpos

    # Distance to target
    dist = np.linalg.norm(self._target_position - (ee_position - self._target_origin))

    # Check if fingertip is inside target
    if dist < self._target_radius:
      self._steps_inside_target += 1
      self._info["inside_target"] = True
      #Set hit time / added by me
      #-----------------------------------------------------------------------------------------------------------------
      if self.iteration != 0:
        self._set_end_data_target_hit_time()
      #-----------------------------------------------------------------------------------------------------------------
    else:
      self._steps_inside_target = 0
      self._info["inside_target"] = False

    if self._info["inside_target"] and self._steps_inside_target >= self._dwell_threshold:

      # Update counters
      self._info["target_hit"] = True
      self._trial_idx += 1
      self._targets_hit += 1
      self._steps_since_last_hit = 0
      self._steps_inside_target = 0
      self._info["acc_dist"] += dist
      self._spawn_target(model, data)
      self._info["target_spawned"] = True

    else:

      self._info["target_hit"] = False

      # Check if time limit has been reached
      self._steps_since_last_hit += 1
      if self._steps_since_last_hit >= self._max_steps_without_hit:
        # Spawn a new target
        self._steps_since_last_hit = 0
        self._trial_idx += 1
        self._info["acc_dist"] += dist
        self._spawn_target(model, data)
        self._info["target_spawned"] = True

    # Check if max number trials reached
    if self._trial_idx >= self._max_trials:
      self._info["dist_from_target"] = self._info["acc_dist"]/self._trial_idx
      truncated = True
      self._info["termination"] = "max_trials_reached"

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # fingertip is inside target
    reward = self._reward_function.get(self, dist-self._target_radius, self._info.copy())

    return reward, terminated, truncated, self._info.copy()

  def _get_state(self, model, data):
    state = dict()
    state["target_position"] = self._target_origin.copy()+self._target_position.copy()
    state["target_radius"] = self._target_radius
    state["trial_idx"] = self._trial_idx
    state["targets_hit"] = self._targets_hit
    state.update(self._info)
    return state

  def _reset(self, model, data):

    # Reset counters
    self._steps_since_last_hit = 0
    self._steps_inside_target = 0
    self._trial_idx = 0
    self._targets_hit = 0

    self._info = {"target_hit": False, "inside_target": False, "target_spawned": False,
                  "terminated": False, "truncated": False, "termination": False, "llc_dist_from_target": 0, "dist_from_target": 0, "acc_dist": 0}

    #Added by me
    #-------------------------------------------------------------------------------------------------------
    if self.iteration != 0:
        self._dump_data()
    self._set_start_data(self._target_origin)
    #-------------------------------------------------------------------------------------------------------
      
    # Spawn a new location
    self._spawn_target(model, data)
    
    #-------------------------------------------------------------------------------------------------------
    self._set_end_data_location(self._target_position, self._target_radius)
    #-------------------------------------------------------------------------------------------------------

    return self._info

#-----------------------------------------------------------------------------------------------------------
#added code for ISO-9241-11 / added by me
  def _determine_Pos_old(self):
      #Fixed Parameters
      num_of_positions_in_circle = 12 #Anzahl verschiedener Target-Positionen in den Kreisen
      num_of_diff_radius = 3 #Anzahl verschiedener Kreise
             
      #Create random values to calculate  angle and radius 
      r_Angle = np.random.randint(0,num_of_positions_in_circle) #verschiedene Positionen im Kreis 
      r_Radius = np.random.randint(1,num_of_diff_radius+1) #verschieden breite Kreise
      

      #Calculate the position of the taregt
      limit_r = self._target_limits_y[1]
    
      rad = r_Radius * (limit_r / num_of_diff_radius)  #Berechnet den gewollten Radius
      #print(f"rad: {rad} for {self.iteration}")
      
      angle_grad = r_Angle * (360 / num_of_positions_in_circle)
      #print(f"angle_grad: {angle_grad} for {self.iteration}")
      
      angle_rad = math.radians(angle_grad)
      #print(f"angle_rad: {angle_rad} for {self.iteration}")
      
      y = rad * math.cos(angle_rad)
      #print(f"y: {y} for {self.iteration}")
      
      z = rad * math.sin(angle_rad)
      #print(f"z: {z} for {self.iteration}")

      return y,z

  def _determine_Size_old(self):
      #Fixed Parameters
      num_of_diff_target_sizes = 10 #Anzahl verschiedener Targetgrößen
      
      #Create a random value to calculate the size of the target
      r_size = np.random.randint(0,num_of_diff_target_sizes+1) #verschiedene größen für einen Kreis
      
      #Calculate the size of the target
      size = self._target_radius_limit[0] + ((self._target_radius_limit[1] - self._target_radius_limit[0]) / num_of_diff_target_sizes) * r_size      

      return size

  def _determine_target(self):
        
    #Parameters - Overwrite if needed
    numBalls = 12
    numRadius = 3
    numSizes = 3
    #--------------------------------
    cSize = self._ball_status // (numRadius - 1)
    if cSize > numSizes:
        self._ball_status = 0
        cSize = 0

    cRadius = ( self._ball_status % (numRadius - 1) ) // (numBalls -1)
    cPosition = ( self._ball_status % (numRadius - 1) ) % (numBalls -1)

    cAngle = self.calculatePos(cPosition, ( numBalls / 2 ), ( numBalls / 2 ) - 1)
    limit_r = self._target_limits_y[1]
        
    rad = cRadius * (limit_r / numRadius)
    angle_grad = cAngle * (360 / numBalls)
    angle_rad = math.radians(angle_grad)
    y = rad * math.cos(angle_rad)
    z = rad * math.sin(angle_rad)
        
    size = self._target_radius_limit[0] + ((self._target_radius_limit[1] - self._target_radius_limit[0]) / numSizes) * cSize   
    return y,z,size

  def calculatePos(self, iterations, sf, sb):
    """
        Führt iterativ die Operationen +5 und -6 aus.
    
        :param start: Der Startwert (int oder float).
        :param iterations: Anzahl der Iterationen (int).
        :return: Endwert nach allen Iterationen.
    """
    result = 0 
    for i in range(iterations):
        if i % 2 == 0:  # Bei geraden Iterationen +5
            result += sf
        else:  # Bei ungeraden Iterationen -6
            result -= sb
    return result
        
    
#-----------------------------------------------------------------------------------------------------------
      
  def _spawn_target(self, model, data):
    
    #Update data management / added by me
    #-----------------------------------------------------------------------------------------------------
    self.iteration += 1
    self._set_start_data(self._target_position)
    #-----------------------------------------------------------------------------------------------------
    
    # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
    for _ in range(10):
      #Legacy code
      #target_y = self._rng.uniform(*self._target_limits_y)
      #target_z = self._rng.uniform(*self._target_limits_z)

      size, target_y, target_z = self._determine_target() #generate random y and z values in the determined schema
      new_position = np.array([0, target_y, target_z])
      distance = np.linalg.norm(self._target_position - new_position)
      if distance > self._new_target_distance_threshold:
        break
    self._target_position = new_position
    
    # Set location
    model.body("target").pos[:] = self._target_origin + self._target_position

    # Sample target radius
    self._target_radius = size

    # Set target radius
    model.geom("target").size[0] = self._target_radius
      
    #Set end data
    #--------------------------------------------------------------------------------------------------
    self._set_end_data_location(self._target_position, self._target_radius)
    #--------------------------------------------------------------------------------------------------
      
    mujoco.mj_forward(model, data)
    
    
  def get_stateful_information(self, model, data):
    # Time features (time left to reach target, time spent inside target)
    targets_hit = -1.0 + 2*(self._trial_idx/self._max_trials)
    dwell_time = -1.0 + 2 * np.min([1.0, self._steps_inside_target / self._dwell_threshold])
    return np.array([dwell_time, targets_hit])
