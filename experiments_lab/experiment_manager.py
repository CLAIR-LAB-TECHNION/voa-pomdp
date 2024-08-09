from frozendict import frozendict

from experiments_lab.experiment_results_data import ExperimentResults
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default
from modeling.policies.abstract_policy import AbastractPolicy
from experiments_lab.block_stacking_env import LabBlockStackingEnv


default_rewards = frozendict(stacking_reward=1,
                             finish_ahead_of_time_reward_coeff=0.1,
                             sensing_cost_coeff=0.1,
                             stacking_cost_coeff=0.2)

class ExperimentManager:
    def __init__(self, env: LabBlockStackingEnv, policy: AbastractPolicy):
        self.env = env
        self.policy = policy

        # TODO controller and other objects
        # TODO block piles managements
        # TODO clean up from here, env just manages one experiment

    @classmethod
    def from_params(cls,
                    n_blocks: int,
                    max_steps: int,
                    r1_controller: ManipulationController,
                    r2_controller: ManipulationController,
                    gt: GeometryAndTransforms,
                    camera=None,
                    position_estimator=None,
                    ws_x_lims=workspace_x_lims_default,
                    ws_y_lims=workspace_y_lims_default,
                    rewards: dict = default_rewards,):
        raise NotImplementedError

    def run_single_experiment(self, init_block_positions, init_block_belief) -> ExperimentResults:
        results = ExperimentResults(policy_type=self.policy.__class__.__name__,
                                    agent_params=self.policy.get_params(),)



    def clean_up_workspace_and_reset(self):
        pass




