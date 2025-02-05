from ..base import BaseBMModel
import numpy as np
from scipy.special import gamma
import mujoco


class MoblArmsIndexEMG_Variance(BaseBMModel):
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

    def sample_inverse_gamma(self, alpha, beta):
        """
            Sample a value from the inverse gamma distribution.

            Parameters:
            alpha (float): Shape parameter.
            beta (float): Scale parameter.

            Returns:
            float: A sampled variance (sigma^2).
        """
        return 1 / np.random.gamma(alpha, 1 / beta)

    def sample_emg_signal(self, sigma2):
        """
            Sample a new EMG signal from the conditional distribution P(x | sigma^2).

        Parameters:
        sigma2 (float): Variance (sigma^2).

        Returns:
            float: A sampled EMG signal.
        """
        return np.random.normal(0, np.sqrt(sigma2))

    def generate_emg_signal(self, lower_boundary, upper_boundary, initial_emg_signal):
        """
            Generate a new EMG signal sampled from the conditional probability distribution.

            Parameters:
            initial_emg_signal (float): The initial EMG signal value.

        Returns:
            float: A new EMG signal sampled from the distribution.
        """
        # Parameters for the inverse gamma distribution
        alpha = 3.0  # Shape parameter
        beta = 2.0   # Scale parameter

        # Sample variance (sigma^2) from the inverse gamma distribution
        sigma2 = self.sample_inverse_gamma(alpha, beta)

        # Sample new EMG signal from the conditional distribution
        new_emg_signal = self.sample_emg_signal(sigma2)
        return max(lower, min(new_emg_signal, upper))



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