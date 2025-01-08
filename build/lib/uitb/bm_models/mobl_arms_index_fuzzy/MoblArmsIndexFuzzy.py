from ..base import BaseBMModel
import numpy as np
import mujoco


class MoblArmsIndexFuzzy(BaseBMModel):
    """This model is based on the MoBL ARMS model, see https://simtk.org/frs/?group_id=657 for the original model in OpenSim,
    and https://github.com/aikkala/O2MConverter for the MuJoCo converted model. This model is the same as the one in uitb/bm_models/mobl_arms, except
    the index finger is flexed and it contains a force sensor.
    """

    def __init__(self, model, data, **kwargs):
        super().__init__(model, data, **kwargs)

        # Set shoulder variant; use "none" as default, "patch-v1" for more reasonable movements (not thoroughly tested)
        self.shoulder_variant = kwargs.get("shoulder_variant", "none")

    def _update(self, model, data):
        """Update model constraints and apply noise."""
        # Update shoulder equality constraints
        if self.shoulder_variant.startswith("patch"):
            model.equality("shoulder1_r2_con").data[1] = self._introduce_noise(
                -((np.pi - 2 * data.joint('shoulder_elv').qpos) / np.pi)
            )

            if self.shoulder_variant == "patch-v2":
                # Compute new range values
                shoulder_range = (
                    np.array([-np.pi / 2, np.pi / 9])
                    - 2
                    * np.min(
                        [
                            data.joint('shoulder_elv').qpos,
                            np.pi - data.joint('shoulder_elv').qpos,
                        ]
                    )
                    / np.pi
                    * data.joint('elv_angle').qpos
                )

                # Apply noise
                data.joint('shoulder_rot').range[:] = self._introduce_noise_v2(
                    shoulder_range
                )

            # Perform forward dynamics
            mujoco.mj_forward(model, data)

    @classmethod
    def _get_floor(cls):
        """Return the floor properties (if any)."""
        return None

    def _introduce_noise(self, mv, lower_boundary=-1, upper_boundary=1):
        """
        Introduces noise into the biomechanical model based on the current speed (mv).
        
        Parameters:
            mv (float): Current value to apply noise to.
            lower_boundary (float): Minimum allowed value.
            upper_boundary (float): Maximum allowed value.

        Returns:
            float: Value after applying noise.
        """
        magnitude = 0.1  # Adjust noise magnitude
        attack_variance = abs(mv) ** 2  # Variance scales with magnitude

        # Add noise
        noise = magnitude * np.random.normal(0, attack_variance)
        mv += noise

        # Clamp the value within boundaries
        return max(lower_boundary, min(mv, upper_boundary))

    def _introduce_noise_v2(self, mv):
        """
        Adds noise to a two-element array.

        Parameters:
            mv (np.array): Input array of two elements.

        Returns:
            np.array: Array after applying noise.
        """
        lower_boundary = np.array([-2.57, -0.65])
        upper_boundary = np.array([-1.57, 0.35])
        return np.array([self._introduce_noise(mv[0], lower_boundary=lower_boundary[0], upper_boundary=upper_boundary[0]), self._introduce_noise(mv[1],lower_boundary=lower_boundary[1], upper_boundary=upper_boundary[1])])