import numpy as np


class ObjectManager:
    """convenience class to manage graspable objects in the mujoco simulation"""

    def __init__(self, mj_model, mj_data):
        self._mj_model = mj_model
        self._mj_data = mj_data

        # manipulated objects have 6dof free joint that must be named in the mcjf.
        all_joint_names = [self._mj_model.joint(i).name for i in range(self._mj_model.njnt)]

        # all bodies that ends with "box"
        self.object_names = [name for name in all_joint_names if name.startswith("block")]
        self.objects_mjdata_dict = {name: self._mj_model.joint(name) for name in self.object_names}
        self.initial_positions_dict = self.get_all_block_positions()

    def reset(self, randomize=False, block_positions=None):
        """
        Reset the object positions in the simulation.
        Args:
            randomize: if True, randomize the positions of the blocks, otherwise set them to initial positions.
        """
        if randomize:
            # randomize block positions
            self.set_all_block_positions([np.random.uniform(-0.2, 0.2, 3) for _ in range(len(self.object_names))])
        else:
            if block_positions:
                self.set_all_block_positions(block_positions)
            else:
                self.set_all_block_positions(list(self.initial_positions_dict.values()))

    def get_block_position(self, block_id: int) -> np.ndarray:
        """
        Get the position of a block in the simulation.
        Args:
            block_id: the id of the block to get the position of.
        Returns:
            the position of the block in format [x, y ,z].
        """
        return self._mj_data.joint(block_id).qpos[:3]

    def get_all_block_positions(self) -> dict:
        """
        Get the positions of all blocks in the simulation.
        Returns:
            a dictionary of block names to their positions, positions will be in format {name: [x, y ,z], ...}.
        """
        return {name: self.get_block_position(self.objects_mjdata_dict[name].id) for name in self.object_names}

    def set_block_position(self, block_id, position):
        """
        Set the position of a block in the simulation.
        Args:
            block_id: the id of the block to set the position of.
            position: the position to set the block to, position will be in format [x, y ,z].
        """
        joint_name = f"block{block_id + 1}_fj"
        joint_id = self._mj_model.joint(joint_name).id
        pos_adrr = self._mj_model.jnt_qposadr[joint_id]
        self._mj_data.qpos[pos_adrr:pos_adrr + 3] = position

    def set_all_block_positions(self, positions):
        """
        Set the positions of all blocks in the simulation.
        Args:
            positions: a list of positions to set the blocks to, positions will be in format [[x, y ,z], ...].
        """
        # set blocks positions
        for i, pos in enumerate(positions):
            self.set_block_position(i, pos)
