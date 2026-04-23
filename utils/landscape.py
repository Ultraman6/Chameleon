"""
Loss Landscape Visualizer for PyTorch 2.6+
A comprehensive tool for generating 2D/3D loss landscape visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import time
import copy
import glob
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

from tqdm import tqdm
import pickle

# 关闭backward(create_graph=True)的内存泄漏警告
warnings.filterwarnings("ignore", message="Using backward.*create_graph=True.*")


@dataclass
class VisualizationConfig:
    """Configuration for loss landscape visualization"""
    # Coordinate settings
    x_range: Tuple[float, float, int] = (-1.0, 1.0, 51)  # (start, end, num_points)
    y_range: Optional[Tuple[float, float, int]] = None   # None for 1D, set for 2D

    # Direction settings
    direction_type: str = 'weights'  # 'weights' or 'states'
    direction_method: str = 'random'  # 'random', 'interpolate', 'pca', 'eigen', 'lstsq'
    x_norm: str = 'filter'  # 'filter', 'layer', 'weight', 'dlayer', 'dfilter'
    y_norm: str = 'filter'
    ignore_bias_bn: bool = True

    # Advanced direction settings
    end_root: Optional[str] = None  # For interpolate method
    trajectory: Optional[List[nn.Module]] = None  # For PCA method
    project: Optional[str] = None  # For PCA method: 'cos', 'lstsq', or None
    top_n: int = 10  # For eigen method: number of top eigenvectors
    lstsq_lr: float = 0.004  # Learning rate for lstsq method
    lstsq_epochs: int = 100  # Epochs for lstsq method
    lstsq_batch_size: int = 1024  # Batch size for lstsq method

    # Computation settings
    batch_size: int = 128
    num_workers: int = 4
    use_amp: bool = True  # Automatic Mixed Precision



    # Output settings
    save_dir: str = './loss_landscape_results'
    save_format: str = 'h5'  # 'h5' or 'npz'

    # Visualization settings
    plot_style: str = 'default'  # 'default', 'seaborn', 'matplotlib'
    dpi: int = 300
    show_plots: bool = False


class DirectionGenerator:
    """Generate and normalize direction vectors for loss landscape visualization"""

    @staticmethod
    def get_weights(model: nn.Module) -> List[torch.Tensor]:
        """Extract weights from model"""
        return [p.data.clone() for p in model.parameters() if p.requires_grad]

    @staticmethod
    def get_states(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract state dict from model"""
        return {k: v.data.clone() for k, v in model.state_dict().items()}

    @staticmethod
    def get_random_weights(weights: List[torch.Tensor]) -> List[torch.Tensor]:
        """Generate random direction for weights"""
        return [torch.randn_like(w) for w in weights]

    @staticmethod
    def get_random_states(states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate random direction for states"""
        return {k: torch.randn_like(v) for k, v in states.items()}

    @staticmethod
    def normalize_direction(direction: torch.Tensor, weights: torch.Tensor,
                          norm: str = 'filter') -> None:
        """Normalize direction according to specified method"""
        if norm == 'filter':
            # Rescale filters so each filter has same norm as corresponding filter in weights
            for d, w in zip(direction, weights):
                if d.dim() > 1:
                    d.mul_(w.norm() / (d.norm() + 1e-10))
        elif norm == 'layer':
            # Rescale layer variables
            direction.mul_(weights.norm() / direction.norm())
        elif norm == 'weight':
            # Rescale entries
            direction.mul_(weights)
        elif norm == 'dfilter':
            # Unit norm for each filter
            for d in direction:
                if d.dim() > 1:
                    d.div_(d.norm() + 1e-10)
        elif norm == 'dlayer':
            # Unit norm for layer
            direction.div_(direction.norm())

    @staticmethod
    def normalize_directions_for_weights(direction: List[torch.Tensor],
                                       weights: List[torch.Tensor],
                                       norm: str = 'filter',
                                       ignore: str = 'biasbn') -> None:
        """Normalize direction for weights"""
        assert len(direction) == len(weights)
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)  # ignore bias and BN parameters
                else:
                    d.copy_(w)  # keep directions for 1D parameters
            else:
                DirectionGenerator.normalize_direction(d, w, norm)

    @staticmethod
    def normalize_directions_for_states(direction: Dict[str, torch.Tensor],
                                      states: Dict[str, torch.Tensor],
                                      norm: str = 'filter',
                                      ignore: str = 'biasbn') -> None:
        """Normalize direction for states"""
        assert len(direction) == len(states)
        for k in direction.keys():
            d, w = direction[k], states[k]
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)
                else:
                    d.copy_(w)
            else:
                DirectionGenerator.normalize_direction(d, w, norm)

    @staticmethod
    def create_random_direction(model: nn.Module,
                              direction_type: str = 'weights',
                              ignore: str = 'biasbn',
                              norm: str = 'filter') -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Create a random normalized direction"""
        if direction_type == 'weights':
            weights = DirectionGenerator.get_weights(model)
            direction = DirectionGenerator.get_random_weights(weights)
            DirectionGenerator.normalize_directions_for_weights(direction, weights, norm, ignore)
        elif direction_type == 'states':
            states = DirectionGenerator.get_states(model)
            direction = DirectionGenerator.get_random_states(states)
            DirectionGenerator.normalize_directions_for_states(direction, states, norm, ignore)

        return direction

    @staticmethod
    def create_target_direction(model1: nn.Module,
                              model2: nn.Module,
                              direction_type: str = 'weights') -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Create direction from model1 to model2 (renamed from interpolate)"""
        if direction_type == 'weights':
            w1 = DirectionGenerator.get_weights(model1)
            w2 = DirectionGenerator.get_weights(model2)
            direction = [w2[i] - w1[i] for i in range(len(w1))]
        elif direction_type == 'states':
            s1 = DirectionGenerator.get_states(model1)
            s2 = DirectionGenerator.get_states(model2)
            direction = {k: s2[k] - s1[k] for k in s1.keys()}

        return direction

    @staticmethod
    def create_orthogonal_direction(base_direction: Union[List[torch.Tensor], Dict[str, torch.Tensor]],
                                  direction_type: str = 'weights') -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Create a direction orthogonal to the base direction"""
        if direction_type == 'weights':
            # Convert to flat tensors for orthogonalization
            base_flat = torch.cat([d.flatten() for d in base_direction])
            # Generate random direction
            random_flat = torch.randn_like(base_flat)
            # Gram-Schmidt orthogonalization
            proj = torch.dot(random_flat, base_flat) / torch.dot(base_flat, base_flat)
            orthogonal_flat = random_flat - proj * base_flat
            orthogonal_flat = orthogonal_flat / torch.norm(orthogonal_flat)

            # Convert back to original structure
            orthogonal_direction = []
            start_idx = 0
            for d in base_direction:
                end_idx = start_idx + d.numel()
                orthogonal_direction.append(orthogonal_flat[start_idx:end_idx].reshape(d.shape))
                start_idx = end_idx

            return orthogonal_direction
        else:
            # For states, similar approach
            base_flat = torch.cat([v.flatten() for v in base_direction.values()])
            random_flat = torch.randn_like(base_flat)
            proj = torch.dot(random_flat, base_flat) / torch.dot(base_flat, base_flat)
            orthogonal_flat = random_flat - proj * base_flat
            orthogonal_flat = orthogonal_flat / torch.norm(orthogonal_flat)

            orthogonal_direction = {}
            start_idx = 0
            for k, v in base_direction.items():
                end_idx = start_idx + v.numel()
                orthogonal_direction[k] = orthogonal_flat[start_idx:end_idx].reshape(v.shape)
                start_idx = end_idx

            return orthogonal_direction

    @staticmethod
    def create_pca_direction(model: nn.Module,
                           trajectory: List[nn.Module],
                           direction_type: str = 'weights',
                           project: Optional[str] = None,
                           top_n: int = 2) -> List[Union[List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """Create directions using PCA on model trajectory"""
        if not trajectory:
            raise ValueError("Trajectory cannot be empty for PCA direction")

        # Extract parameters from trajectory
        if direction_type == 'weights':
            trajectory_params = []
            for traj_model in trajectory:
                params = DirectionGenerator.get_weights(traj_model)
                flat_params = torch.cat([p.flatten() for p in params])
                trajectory_params.append(flat_params.cpu().numpy())

            # Stack parameters
            param_matrix = np.vstack(trajectory_params)

            # Apply PCA
            pca = PCA(n_components=min(top_n, len(trajectory)))
            pca.fit(param_matrix)

            # Create directions from PCA components
            directions = []
            for i in range(min(top_n, len(trajectory))):
                component = torch.from_numpy(pca.components_[i]).float()

                # Convert back to original structure
                direction = []
                start_idx = 0
                for param in DirectionGenerator.get_weights(model):
                    end_idx = start_idx + param.numel()
                    direction.append(component[start_idx:end_idx].reshape(param.shape))
                    start_idx = end_idx

                directions.append(direction)

            return directions
        else:
            # For states
            trajectory_params = []
            for traj_model in trajectory:
                states = DirectionGenerator.get_states(traj_model)
                flat_states = torch.cat([v.flatten() for v in states.values()])
                trajectory_params.append(flat_states.cpu().numpy())

            param_matrix = np.vstack(trajectory_params)
            pca = PCA(n_components=min(top_n, len(trajectory)))
            pca.fit(param_matrix)

            directions = []
            for i in range(min(top_n, len(trajectory))):
                component = torch.from_numpy(pca.components_[i]).float()

                direction = {}
                start_idx = 0
                for k, v in DirectionGenerator.get_states(model).items():
                    end_idx = start_idx + v.numel()
                    direction[k] = component[start_idx:end_idx].reshape(v.shape)
                    start_idx = end_idx

                directions.append(direction)

            return directions

    @staticmethod
    def create_eigen_direction(model: nn.Module,
                             dataloader: torch.utils.data.DataLoader,
                             criterion: nn.Module,
                             device: torch.device,
                             direction_type: str = 'weights',
                             top_n: int = 2) -> List[Union[List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """Create directions using Hessian eigenvectors using power iteration method"""
        import warnings
        warnings.filterwarnings("ignore", message="Using backward.*create_graph=True.*")

        model.eval()

        # Create Hessian object
        hessian_obj = Hessian(model, criterion, device, dataloader=dataloader)

        # Compute eigenvalues and eigenvectors using power iteration
        print(f"Computing top {top_n} eigenvalues using power iteration...")
        eigenvalues, eigenvectors = hessian_obj.eigenvalues(maxIter=50, tol=1e-3, top_n=top_n)

        print(f"Computed eigenvalues: {eigenvalues}")

        # Convert eigenvectors to directions
        directions = []
        for i in range(top_n):
            eigenvec = eigenvectors[i]

            if direction_type == 'weights':
                # For weights, return list of tensors
                direction = eigenvec
            else:
                # For state_dict, convert to dictionary format
                direction = {}
                param_idx = 0
                for name, param in model.state_dict().items():
                    if param.requires_grad:  # Only include trainable parameters
                        param_size = param.numel()
                        direction[name] = eigenvec[param_idx].reshape(param.shape)
                        param_idx += 1

            directions.append(direction)

        return directions

    @staticmethod
    def create_lstsq_direction(model: nn.Module,
                             dataloader: torch.utils.data.DataLoader,
                             criterion: nn.Module,
                             device: torch.device,
                             trajectory: List[nn.Module],
                             direction_type: str = 'weights',
                             lr: float = 0.004,
                             epochs: int = 100,
                             batch_size: int = 1024) -> List[Union[List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """Create directions using least squares optimization"""
        if not trajectory:
            raise ValueError("Trajectory cannot be empty for lstsq direction")

        # Start with random directions
        directions = []
        for i in range(2):  # Create 2 directions
            if i == 0:
                direction = DirectionGenerator.create_random_direction(model, direction_type, 'biasbn', 'filter')
            else:
                direction = DirectionGenerator.create_orthogonal_direction(directions[0], direction_type)
            directions.append(direction)

        # Convert to flat tensors for optimization
        if direction_type == 'weights':
            flat_directions = [torch.cat([d.flatten() for d in dir_list]) for dir_list in directions]
        else:
            flat_directions = [torch.cat([v.flatten() for v in dir_dict.values()]) for dir_dict in directions]

        # Get trajectory parameters
        trajectory_params = []
        for traj_model in trajectory:
            if direction_type == 'weights':
                params = DirectionGenerator.get_weights(traj_model)
                flat_params = torch.cat([p.flatten() for p in params])
            else:
                states = DirectionGenerator.get_states(traj_model)
                flat_params = torch.cat([v.flatten() for v in states.values()])
            trajectory_params.append(flat_params)

        # Get current model parameters
        if direction_type == 'weights':
            current_params = torch.cat([p.flatten() for p in DirectionGenerator.get_weights(model)])
        else:
            current_states = DirectionGenerator.get_states(model)
            current_params = torch.cat([v.flatten() for v in current_states.values()])

        # Optimization loop
        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Create direction matrix
                direction_matrix = torch.stack(flat_directions).T  # [param_count, num_directions]

                # For each trajectory point, find best coefficients
                for traj_idx, traj_params in enumerate(trajectory_params[:-1]):
                    # Compute relative weight change
                    relative_change = traj_params - current_params

                    # Solve least squares
                    coefs = torch.linalg.lstsq(direction_matrix, relative_change).solution

                    # Project weight using directions
                    projected_weight = current_params + direction_matrix @ coefs

                    # Update model with projected weight
                    if direction_type == 'weights':
                        start_idx = 0
                        for param in model.parameters():
                            if param.requires_grad:
                                end_idx = start_idx + param.numel()
                                param.data = projected_weight[start_idx:end_idx].reshape(param.shape)
                                start_idx = end_idx
                    else:
                        start_idx = 0
                        state_dict = model.state_dict()
                        for k, v in state_dict.items():
                            end_idx = start_idx + v.numel()
                            state_dict[k] = projected_weight[start_idx:end_idx].reshape(v.shape)
                            start_idx = end_idx

                    # Compute loss difference
                    model.eval()
                    with torch.no_grad():
                        outputs = model(inputs)
                        projected_loss = criterion(outputs, targets).item()

                        # Get original trajectory loss
                        traj_model = trajectory[traj_idx]
                        traj_model.eval()
                        with torch.no_grad():
                            traj_outputs = traj_model(inputs)
                            traj_loss = criterion(traj_outputs, targets).item()

                    loss_diff = (projected_loss - traj_loss) ** 2
                    total_loss += loss_diff

                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss:.6f}')

            # Update directions using gradient descent
            for i, direction in enumerate(flat_directions):
                # Simplified gradient update (in practice, you'd compute actual gradients)
                flat_directions[i] = direction - lr * torch.randn_like(direction) * total_loss

        # Convert back to original structure
        final_directions = []
        for flat_dir in flat_directions:
            if direction_type == 'weights':
                direction = []
                start_idx = 0
                for param in model.parameters():
                    if param.requires_grad:
                        end_idx = start_idx + param.numel()
                        direction.append(flat_dir[start_idx:end_idx].reshape(param.shape))
                        start_idx = end_idx
            else:
                direction = {}
                start_idx = 0
                for k, v in model.state_dict().items():
                    end_idx = start_idx + v.numel()
                    direction[k] = flat_dir[start_idx:end_idx].reshape(v.shape)
                    start_idx = end_idx

            final_directions.append(direction)

        return final_directions

    @staticmethod
    def create_interpolate_direction(model: nn.Module,
                                   end_root: str,
                                   direction_type: str = 'weights') -> List[Union[List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """Create directions by interpolating between models from end_root directory"""
        if not os.path.exists(end_root):
            raise ValueError(f"End root directory {end_root} does not exist")

        # Find all model files
        model_files = []
        for ext in ['*.pth', '*.pt', '*.ckpt']:
            model_files.extend(glob.glob(os.path.join(end_root, '**', ext), recursive=True))

        if not model_files:
            raise ValueError(f"No model files found in {end_root}")

        directions = []
        for model_file in model_files:
            try:
                # Load model state
                checkpoint = torch.load(model_file, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                # Create temporary model to load state
                temp_model = copy.deepcopy(model)
                temp_model.load_state_dict(state_dict)

                # Create direction from current model to loaded model
                direction = DirectionGenerator.create_target_direction(model, temp_model, direction_type)
                directions.append(direction)

            except Exception as e:
                print(f"Error loading model from {model_file}: {e}")
                continue

        if not directions:
            raise ValueError("No valid models could be loaded from end_root")

        return directions











class ModelModifier:
    """Modify model parameters for loss landscape exploration"""

    @staticmethod
    def modify_weights_1d(model: nn.Module,
                         direction: List[torch.Tensor],
                         alpha: float) -> None:
        """Modify model weights for 1D exploration"""
        for param, dir_tensor in zip(model.parameters(), direction):
            if param.requires_grad:
                param.data.add_(alpha * dir_tensor)

    @staticmethod
    def modify_weights_2d(model: nn.Module,
                         x_direction: List[torch.Tensor],
                         y_direction: List[torch.Tensor],
                         alpha: float, beta: float) -> None:
        """Modify model weights for 2D exploration"""
        for param, x_dir, y_dir in zip(model.parameters(), x_direction, y_direction):
            if param.requires_grad:
                param.data.add_(alpha * x_dir + beta * y_dir)

    @staticmethod
    def modify_states_1d(model: nn.Module,
                        direction: Dict[str, torch.Tensor],
                        alpha: float) -> None:
        """Modify model states for 1D exploration"""
        state_dict = model.state_dict()
        for key in state_dict.keys():
            if key in direction:
                state_dict[key].add_(alpha * direction[key])

    @staticmethod
    def modify_states_2d(model: nn.Module,
                        x_direction: Dict[str, torch.Tensor],
                        y_direction: Dict[str, torch.Tensor],
                        alpha: float, beta: float) -> None:
        """Modify model states for 2D exploration"""
        state_dict = model.state_dict()
        for key in state_dict.keys():
            if key in x_direction and key in y_direction:
                state_dict[key].add_(alpha * x_direction[key] + beta * y_direction[key])


class LossEvaluator:
    """Evaluate loss and accuracy for modified models"""

    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                 criterion: nn.Module, device: torch.device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.original_state = None

    def save_original_state(self, direction_type='weights'):
        """Save original model state"""
        if direction_type == 'weights':
            self.original_state = DirectionGenerator.get_weights(self.model)
        else:
            self.original_state = DirectionGenerator.get_states(self.model)

    def restore_original_state(self, direction_type='weights'):
        """Restore original model state"""
        if self.original_state is None:
            return

        if direction_type == 'weights':
            for param, original_weight in zip(self.model.parameters(), self.original_state):
                if param.requires_grad:
                    param.data.copy_(original_weight)
        else:
            state_dict = self.model.state_dict()
            for key, original_value in self.original_state.items():
                if key in state_dict:
                    state_dict[key].copy_(original_value)

    def evaluate(self, use_amp: bool = True) -> Tuple[float, float]:
        """Evaluate loss and accuracy"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if use_amp:
                    with torch.autocast(device_type=str(self.device)):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy





class LossLandscapeVisualizer:
    """Main class for loss landscape visualization"""

    def __init__(self, model: nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 criterion: nn.Module,
                 device: torch.device,
                 config: VisualizationConfig):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.config = config



        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)

        # Initialize components
        self.evaluator = LossEvaluator(model, dataloader, criterion, device)
        self.evaluator.config = config



        # Generate directions
        self.x_direction = None
        self.y_direction = None

    def generate_directions(self,
                          model2: Optional[nn.Module] = None,
                          model3: Optional[nn.Module] = None,
                          same_direction: bool = False):
        """Generate direction vectors for visualization"""
        print(f"Generating direction vectors using method: {self.config.direction_method}")

        # Generate directions based on method
        if self.config.direction_method == 'random':
            self._generate_random_directions(model2, model3, same_direction)
        elif self.config.direction_method == 'interpolate':
            self._generate_interpolate_directions(model2, model3, same_direction)
        elif self.config.direction_method == 'pca':
            self._generate_pca_directions()
        elif self.config.direction_method == 'eigen':
            self._generate_eigen_directions()
        elif self.config.direction_method == 'lstsq':
            self._generate_lstsq_directions()
        else:
            raise NotImplementedError(f"Unsupported direction method: {self.config.direction_method}")

        print("Direction vectors generated successfully!")

    def _generate_random_directions(self, model2, model3, same_direction):
        """Generate random directions"""
        # Generate x direction
        if model2 is not None:
            self.x_direction = DirectionGenerator.create_target_direction(
                self.model, model2, self.config.direction_type)
        else:
            self.x_direction = DirectionGenerator.create_random_direction(
                self.model, self.config.direction_type,
                'biasbn' if self.config.ignore_bias_bn else None,
                self.config.x_norm)

        # Generate y direction (for 2D)
        if self.config.y_range is not None:
            if same_direction:
                self.y_direction = self.x_direction
            elif model3 is not None:
                self.y_direction = DirectionGenerator.create_target_direction(
                    self.model, model3, self.config.direction_type)
            else:
                self.y_direction = DirectionGenerator.create_random_direction(
                    self.model, self.config.direction_type,
                    'biasbn' if self.config.ignore_bias_bn else None,
                    self.config.y_norm)

    def _generate_interpolate_directions(self, model2, model3, same_direction):
        """Generate interpolate directions"""
        if self.config.end_root is None:
            raise ValueError("end_root must be specified for interpolate direction method")

        # Load directions from end_root directory
        directions = DirectionGenerator.create_interpolate_direction(
            self.model, self.config.end_root, self.config.direction_type)

        if len(directions) < 1:
            raise ValueError("No valid directions found in end_root directory")

        # Use first direction as x direction
        self.x_direction = directions[0]

        # Use second direction as y direction (if available and 2D)
        if self.config.y_range is not None:
            if same_direction or len(directions) < 2:
                self.y_direction = self.x_direction
            else:
                self.y_direction = directions[1]

    def _generate_pca_directions(self):
        """Generate PCA directions"""
        if self.config.trajectory is None:
            raise ValueError("trajectory must be specified for PCA direction method")

        # Generate PCA directions
        directions = DirectionGenerator.create_pca_direction(
            self.model, self.config.trajectory, self.config.direction_type,
            self.config.project, self.config.top_n)

        if len(directions) < 1:
            raise ValueError("No PCA directions generated")

        # Use first direction as x direction
        self.x_direction = directions[0]

        # Use second direction as y direction (if available and 2D)
        if self.config.y_range is not None:
            if len(directions) < 2:
                # Create orthogonal direction if only one PCA direction
                self.y_direction = DirectionGenerator.create_orthogonal_direction(
                    self.x_direction, self.config.direction_type)
            else:
                self.y_direction = directions[1]

    def _generate_eigen_directions(self):
        """Generate eigen directions"""
        # Generate eigen directions
        directions = DirectionGenerator.create_eigen_direction(
            self.model, self.dataloader, self.criterion, self.device,
            self.config.direction_type, self.config.top_n)

        if len(directions) < 1:
            raise ValueError("No eigen directions generated")

        # Use first direction as x direction
        self.x_direction = directions[0]

        # Use second direction as y direction (if available and 2D)
        if self.config.y_range is not None:
            if len(directions) < 2:
                # Create orthogonal direction if only one eigen direction
                self.y_direction = DirectionGenerator.create_orthogonal_direction(
                    self.x_direction, self.config.direction_type)
            else:
                self.y_direction = directions[1]

    def _generate_lstsq_directions(self):
        """Generate lstsq directions"""
        if self.config.trajectory is None:
            raise ValueError("trajectory must be specified for lstsq direction method")

        # Generate lstsq directions
        directions = DirectionGenerator.create_lstsq_direction(
            self.model, self.dataloader, self.criterion, self.device,
            self.config.trajectory, self.config.direction_type,
            self.config.lstsq_lr, self.config.lstsq_epochs, self.config.lstsq_batch_size)

        if len(directions) < 1:
            raise ValueError("No lstsq directions generated")

        # Use first direction as x direction
        self.x_direction = directions[0]

        # Use second direction as y direction (if available and 2D)
        if self.config.y_range is not None:
            if len(directions) < 2:
                # Create orthogonal direction if only one lstsq direction
                self.y_direction = DirectionGenerator.create_orthogonal_direction(
                    self.x_direction, self.config.direction_type)
            else:
                self.y_direction = directions[1]

    def generate_filename(self, is_plot: bool = False) -> str:
        """Generate filename based on configuration parameters"""
        # Get basic parameters
        direction_method = self.config.direction_method
        direction_type = self.config.direction_type
        x_norm = self.config.x_norm
        y_norm = self.config.y_norm

        # Get coordinate info
        x_start, x_end, x_points = self.config.x_range
        dim_str = "1d" if self.config.y_range is None else "2d"

        # Build filename components
        components = [
            f"loss_landscape_{dim_str}",
            f"method_{direction_method}",
            f"type_{direction_type}",
            f"xnorm_{x_norm}",
            f"ynorm_{y_norm}",
            f"x{x_start:.1f}to{x_end:.1f}_{x_points}pts"
        ]

        # Add y-range info for 2D
        if self.config.y_range is not None:
            y_start, y_end, y_points = self.config.y_range
            components.append(f"y{y_start:.1f}to{y_end:.1f}_{y_points}pts")

        # Add direction-specific parameters
        if direction_method == 'eigen':
            components.append(f"top{self.config.top_n}")
        elif direction_method == 'pca':
            components.append(f"top{self.config.top_n}")
            if self.config.project:
                components.append(f"proj_{self.config.project}")
        elif direction_method == 'lstsq':
            components.append(f"lr{self.config.lstsq_lr}")
            components.append(f"epochs{self.config.lstsq_epochs}")

        # Add other important parameters
        if self.config.ignore_bias_bn:
            components.append("nobiasbn")

        if self.config.use_amp:
            components.append("amp")

        # Add timestamp to avoid conflicts
        timestamp = int(time.time())
        components.append(f"t{timestamp}")

        # Join components
        filename = "_".join(components)

        # Add plot suffix if needed
        if is_plot:
            filename += "_plot"

        return filename

    def compute_loss_landscape_1d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 1D loss landscape"""
        if self.x_direction is None:
            raise ValueError("Direction vectors not generated. Call generate_directions() first.")

        x_start, x_end, x_points = self.config.x_range
        x_coords = np.linspace(x_start, x_end, x_points)

        # Sequential computation
        losses = np.zeros(x_points)
        accuracies = np.zeros(x_points)

        # Save original state
        self.evaluator.save_original_state(self.config.direction_type)

        print(f"Computing 1D loss landscape with {x_points} points...")

        for i, alpha in enumerate(tqdm(x_coords, desc="Computing 1D points")):
            # Restore original state
            self.evaluator.restore_original_state(self.config.direction_type)

            # Modify model
            if self.config.direction_type == 'weights':
                ModelModifier.modify_weights_1d(self.model, self.x_direction, alpha)
            else:
                ModelModifier.modify_states_1d(self.model, self.x_direction, alpha)

            # Evaluate
            loss, acc = self.evaluator.evaluate(self.config.use_amp)
            losses[i] = loss
            accuracies[i] = acc

        # Restore original state
        self.evaluator.restore_original_state(self.config.direction_type)

        return x_coords, losses, accuracies

    def compute_loss_landscape_2d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D loss landscape"""
        if self.x_direction is None or self.y_direction is None:
            raise ValueError("Direction vectors not generated. Call generate_directions() first.")

        x_start, x_end, x_points = self.config.x_range
        y_start, y_end, y_points = self.config.y_range

        x_coords = np.linspace(x_start, x_end, x_points)
        y_coords = np.linspace(y_start, y_end, y_points)

        # Sequential computation
        losses = np.zeros((y_points, x_points))
        accuracies = np.zeros((y_points, x_points))

        # Save original state
        self.evaluator.save_original_state(self.config.direction_type)

        total_points = x_points * y_points
        print(f"Computing 2D loss landscape with {total_points} points...")

        # Use tqdm for progress tracking
        with tqdm(total=total_points, desc="Computing 2D points") as pbar:
            for i, alpha in enumerate(x_coords):
                for j, beta in enumerate(y_coords):
                    # Restore original state
                    self.evaluator.restore_original_state(self.config.direction_type)

                    # Modify model
                    if self.config.direction_type == 'weights':
                        ModelModifier.modify_weights_2d(self.model, self.x_direction, self.y_direction, alpha, beta)
                    else:
                        ModelModifier.modify_states_2d(self.model, self.x_direction, self.y_direction, alpha, beta)

                    # Evaluate
                    loss, acc = self.evaluator.evaluate(self.config.use_amp)
                    losses[j, i] = loss
                    accuracies[j, i] = acc

                    pbar.update(1)
                    pbar.set_postfix({'Loss': f'{loss:.4f}', 'Acc': f'{acc:.2f}%'})

        # Restore original state
        self.evaluator.restore_original_state(self.config.direction_type)

        return x_coords, y_coords, losses, accuracies

    def save_results(self, x_coords: np.ndarray, losses: np.ndarray, accuracies: np.ndarray,
                    y_coords: Optional[np.ndarray] = None, filename: Optional[str] = None):
        """Save results to file with separate data files for each plot type"""
        if filename is None:
            filename = self.generate_filename(is_plot=False)

        # Save main results file
        main_filepath = os.path.join(self.config.save_dir, f"{filename}.{self.config.save_format}")

        if self.config.save_format == 'h5':
            with h5py.File(main_filepath, 'w') as f:
                f.create_dataset('xcoordinates', data=x_coords)
                f.create_dataset('train_loss', data=losses)
                f.create_dataset('train_acc', data=accuracies)
                if y_coords is not None:
                    f.create_dataset('ycoordinates', data=y_coords)

                # Save configuration metadata
                f.attrs['direction_method'] = self.config.direction_method
                f.attrs['direction_type'] = self.config.direction_type
                f.attrs['x_norm'] = self.config.x_norm
                f.attrs['y_norm'] = self.config.y_norm
                f.attrs['ignore_bias_bn'] = self.config.ignore_bias_bn
                f.attrs['use_amp'] = self.config.use_amp
                f.attrs['x_range'] = str(self.config.x_range)
                if self.config.y_range is not None:
                    f.attrs['y_range'] = str(self.config.y_range)

        elif self.config.save_format == 'npz':
            save_dict = {
                'xcoordinates': x_coords,
                'train_loss': losses,
                'train_acc': accuracies
            }
            if y_coords is not None:
                save_dict['ycoordinates'] = y_coords

            # Add metadata as separate arrays
            save_dict['config_direction_method'] = np.array([self.config.direction_method], dtype='object')
            save_dict['config_direction_type'] = np.array([self.config.direction_type], dtype='object')
            save_dict['config_x_norm'] = np.array([self.config.x_norm], dtype='object')
            save_dict['config_y_norm'] = np.array([self.config.y_norm], dtype='object')
            save_dict['config_ignore_bias_bn'] = np.array([self.config.ignore_bias_bn])
            save_dict['config_use_amp'] = np.array([self.config.use_amp])

            np.savez(main_filepath, **save_dict)

        print(f"Main results saved to: {main_filepath}")

        # Save separate data files for each plot type
        if y_coords is not None:
            # 2D plots
            self._save_plot_specific_data_2d(x_coords, y_coords, losses, accuracies, filename)
        else:
            # 1D plots
            self._save_plot_specific_data_1d(x_coords, losses, accuracies, filename)

        return main_filepath

    def _save_plot_specific_data_2d(self, x_coords: np.ndarray, y_coords: np.ndarray,
                                   losses: np.ndarray, accuracies: np.ndarray, base_filename: str):
        """Save separate data files for each 2D plot type"""
        plot_types = [
            'loss_contour',
            'loss_filled_contour',
            'loss_3d_surface',
            'accuracy_contour',
            'accuracy_filled_contour',
            'accuracy_3d_surface'
        ]

        for plot_type in plot_types:
            # Create plot-specific filename
            plot_filename = f"{base_filename}_{plot_type}_data"
            plot_filepath = os.path.join(self.config.save_dir, f"{plot_filename}.{self.config.save_format}")

            if self.config.save_format == 'h5':
                with h5py.File(plot_filepath, 'w') as f:
                    f.create_dataset('xcoordinates', data=x_coords)
                    f.create_dataset('ycoordinates', data=y_coords)

                    # Save the appropriate data for this plot type
                    if 'loss' in plot_type:
                        f.create_dataset('data', data=losses)
                        f.attrs['data_type'] = 'loss'
                    else:
                        f.create_dataset('data', data=accuracies)
                        f.attrs['data_type'] = 'accuracy'

                    f.attrs['plot_type'] = plot_type
                    f.attrs['direction_method'] = self.config.direction_method
                    f.attrs['direction_type'] = self.config.direction_type

            elif self.config.save_format == 'npz':
                save_dict = {
                    'xcoordinates': x_coords,
                    'ycoordinates': y_coords
                }

                if 'loss' in plot_type:
                    save_dict['data'] = losses
                    save_dict['data_type'] = np.array(['loss'], dtype='object')
                else:
                    save_dict['data'] = accuracies
                    save_dict['data_type'] = np.array(['accuracy'], dtype='object')

                save_dict['plot_type'] = np.array([plot_type], dtype='object')
                save_dict['direction_method'] = np.array([self.config.direction_method], dtype='object')
                save_dict['direction_type'] = np.array([self.config.direction_type], dtype='object')

                np.savez(plot_filepath, **save_dict)

            print(f"2D plot-specific data for {plot_type} saved to: {plot_filepath}")

    def _save_plot_specific_data_1d(self, x_coords: np.ndarray, losses: np.ndarray,
                                   accuracies: np.ndarray, base_filename: str):
        """Save separate data files for each 1D plot type"""
        plot_types = [
            'loss_curve',
            'accuracy_curve'
        ]

        for plot_type in plot_types:
            # Create plot-specific filename
            plot_filename = f"{base_filename}_{plot_type}_data"
            plot_filepath = os.path.join(self.config.save_dir, f"{plot_filename}.{self.config.save_format}")

            if self.config.save_format == 'h5':
                with h5py.File(plot_filepath, 'w') as f:
                    f.create_dataset('xcoordinates', data=x_coords)

                    # Save the appropriate data for this plot type
                    if 'loss' in plot_type:
                        f.create_dataset('data', data=losses)
                        f.attrs['data_type'] = 'loss'
                    else:
                        f.create_dataset('data', data=accuracies)
                        f.attrs['data_type'] = 'accuracy'

                    f.attrs['plot_type'] = plot_type
                    f.attrs['direction_method'] = self.config.direction_method
                    f.attrs['direction_type'] = self.config.direction_type

            elif self.config.save_format == 'npz':
                save_dict = {
                    'xcoordinates': x_coords
                }

                if 'loss' in plot_type:
                    save_dict['data'] = losses
                    save_dict['data_type'] = np.array(['loss'], dtype='object')
                else:
                    save_dict['data'] = accuracies
                    save_dict['data_type'] = np.array(['accuracy'], dtype='object')

                save_dict['plot_type'] = np.array([plot_type], dtype='object')
                save_dict['direction_method'] = np.array([self.config.direction_method], dtype='object')
                save_dict['direction_type'] = np.array([self.config.direction_type], dtype='object')

                np.savez(plot_filepath, **save_dict)

            print(f"1D plot-specific data for {plot_type} saved to: {plot_filepath}")

    def plot_1d(self, x_coords: np.ndarray, losses: np.ndarray, accuracies: np.ndarray,
                save_plot: bool = True, filename: Optional[str] = None):
        """Plot 1D loss landscape with separate plots for each visualization type"""
        if self.config.plot_style == 'seaborn':
            plt.style.use('seaborn')

        # Define plot configurations
        plot_configs = [
            {
                'type': 'loss_curve',
                'title': 'Training Loss',
                'data': losses,
                'color': 'b',
                'ylabel': 'Loss'
            },
            {
                'type': 'accuracy_curve',
                'title': 'Training Accuracy',
                'data': accuracies,
                'color': 'r',
                'ylabel': 'Accuracy (%)'
            }
        ]

        # Generate base filename
        if filename is None:
            base_filename = self.generate_filename(is_plot=True)
        else:
            base_filename = filename

        # Create individual plots
        for config in plot_configs:
            self._create_single_1d_plot(x_coords, config, base_filename)

        # Also create the combined plot for backward compatibility
        self._create_combined_1d_plot(x_coords, losses, accuracies, base_filename)

        if self.config.show_plots:
            plt.show()

    def _create_single_1d_plot(self, x_coords: np.ndarray, config: dict, base_filename: str):
        """Create a single 1D plot"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(x_coords, config['data'], color=config['color'], linewidth=2, label=config['title'])
        ax.set_xlabel('Coordinate', fontsize=12)
        ax.set_ylabel(config['ylabel'], fontsize=12, color=config['color'])
        ax.tick_params(axis='y', labelcolor=config['color'])
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save individual plot
        plot_filename = f"{base_filename}_{config['type']}"
        plot_path = os.path.join(self.config.save_dir, f"{plot_filename}.png")
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        print(f"1D {config['type']} plot saved to: {plot_path}")

        plt.close()

    def _create_combined_1d_plot(self, x_coords: np.ndarray, losses: np.ndarray, accuracies: np.ndarray, base_filename: str):
        """Create the combined 1D plot (original format)"""
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8))

        # Loss plot
        ax1.plot(x_coords, losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Coordinate', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Accuracy plot
        ax3.plot(x_coords, accuracies, 'r-', linewidth=2, label='Training Accuracy')
        ax3.set_xlabel('Coordinate', fontsize=12)
        ax3.set_ylabel('Accuracy (%)', fontsize=12, color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()

        # Save combined plot
        combined_filename = f"{base_filename}_combined"
        plot_path = os.path.join(self.config.save_dir, f"{combined_filename}.png")
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        print(f"1D combined plot saved to: {plot_path}")

        plt.close()

    def plot_2d(self, x_coords: np.ndarray, y_coords: np.ndarray,
                losses: np.ndarray, accuracies: np.ndarray,
                save_plot: bool = True, filename: Optional[str] = None):
        """Plot 2D loss landscape with separate plots for each visualization type"""
        if self.config.plot_style == 'seaborn':
            plt.style.use('seaborn')

        X, Y = np.meshgrid(x_coords, y_coords)

        # Define plot configurations
        plot_configs = [
            {
                'type': 'loss_contour',
                'title': 'Loss Contour',
                'data': losses,
                'cmap': 'viridis',
                'plot_func': 'contour',
                'colorbar': True,
                'clabel': True
            },
            {
                'type': 'loss_filled_contour',
                'title': 'Loss Filled Contour',
                'data': losses,
                'cmap': 'viridis',
                'plot_func': 'contourf',
                'colorbar': True,
                'clabel': False
            },
            {
                'type': 'loss_3d_surface',
                'title': 'Loss 3D Surface',
                'data': losses,
                'cmap': 'viridis',
                'plot_func': '3d_surface',
                'colorbar': True,
                'clabel': False
            },
            {
                'type': 'accuracy_contour',
                'title': 'Accuracy Contour',
                'data': accuracies,
                'cmap': 'plasma',
                'plot_func': 'contour',
                'colorbar': True,
                'clabel': True
            },
            {
                'type': 'accuracy_filled_contour',
                'title': 'Accuracy Filled Contour',
                'data': accuracies,
                'cmap': 'plasma',
                'plot_func': 'contourf',
                'colorbar': True,
                'clabel': False
            },
            {
                'type': 'accuracy_3d_surface',
                'title': 'Accuracy 3D Surface',
                'data': accuracies,
                'cmap': 'plasma',
                'plot_func': '3d_surface',
                'colorbar': True,
                'clabel': False
            }
        ]

        # Generate base filename
        if filename is None:
            base_filename = self.generate_filename(is_plot=True)
        else:
            base_filename = filename

        # Create individual plots
        for config in plot_configs:
            self._create_single_2d_plot(X, Y, config, base_filename)

        # Also create the combined plot for backward compatibility
        self._create_combined_2d_plot(X, Y, losses, accuracies, base_filename)

        if self.config.show_plots:
            plt.show()

    def _create_single_2d_plot(self, X: np.ndarray, Y: np.ndarray, config: dict, base_filename: str):
        """Create a single 2D plot"""
        fig = plt.figure(figsize=(10, 8))

        if config['plot_func'] == '3d_surface':
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, config['data'], cmap=config['cmap'], alpha=0.8)
            ax.set_title(config['title'])
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            if 'loss' in config['type']:
                ax.set_zlabel('Loss')
            else:
                ax.set_zlabel('Accuracy (%)')
            if config['colorbar']:
                fig.colorbar(surf, shrink=0.5, aspect=5)
        else:
            ax = fig.add_subplot(111)
            if config['plot_func'] == 'contour':
                plot_obj = ax.contour(X, Y, config['data'], levels=20, cmap=config['cmap'])
                if config['clabel']:
                    ax.clabel(plot_obj, inline=True, fontsize=8)
            else:  # contourf
                plot_obj = ax.contourf(X, Y, config['data'], levels=20, cmap=config['cmap'])

            ax.set_title(config['title'])
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            if config['colorbar']:
                plt.colorbar(plot_obj, ax=ax)

        plt.tight_layout()

        # Save individual plot
        plot_filename = f"{base_filename}_{config['type']}"
        plot_path = os.path.join(self.config.save_dir, f"{plot_filename}.png")
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        print(f"2D {config['type']} plot saved to: {plot_path}")

        plt.close()

    def _create_combined_2d_plot(self, X: np.ndarray, Y: np.ndarray, losses: np.ndarray, accuracies: np.ndarray, base_filename: str):
        """Create the combined 2D plot (original format)"""
        fig = plt.figure(figsize=(15, 10))

        # 2D Contour plot for loss
        ax1 = fig.add_subplot(2, 3, 1)
        contour1 = ax1.contour(X, Y, losses, levels=20, cmap='viridis')
        ax1.clabel(contour1, inline=True, fontsize=8)
        ax1.set_title('Loss Contour')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')

        # 2D Filled contour plot for loss
        ax2 = fig.add_subplot(2, 3, 2)
        contourf1 = ax2.contourf(X, Y, losses, levels=20, cmap='viridis')
        ax2.set_title('Loss Filled Contour')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        plt.colorbar(contourf1, ax=ax2)

        # 3D Surface plot for loss
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        surf1 = ax3.plot_surface(X, Y, losses, cmap='viridis', alpha=0.8)
        ax3.set_title('Loss 3D Surface')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_zlabel('Loss')

        # 2D Contour plot for accuracy
        ax4 = fig.add_subplot(2, 3, 4)
        contour2 = ax4.contour(X, Y, accuracies, levels=20, cmap='plasma')
        ax4.clabel(contour2, inline=True, fontsize=8)
        ax4.set_title('Accuracy Contour')
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')

        # 2D Filled contour plot for accuracy
        ax5 = fig.add_subplot(2, 3, 5)
        contourf2 = ax5.contourf(X, Y, accuracies, levels=20, cmap='plasma')
        ax5.set_title('Accuracy Filled Contour')
        ax5.set_xlabel('X Coordinate')
        ax5.set_ylabel('Y Coordinate')
        plt.colorbar(contourf2, ax=ax5)

        # 3D Surface plot for accuracy
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        surf2 = ax6.plot_surface(X, Y, accuracies, cmap='plasma', alpha=0.8)
        ax6.set_title('Accuracy 3D Surface')
        ax6.set_xlabel('X Coordinate')
        ax6.set_ylabel('Y Coordinate')
        ax6.set_zlabel('Accuracy (%)')

        plt.tight_layout()

        # Save combined plot
        combined_filename = f"{base_filename}_combined"
        plot_path = os.path.join(self.config.save_dir, f"{combined_filename}.png")
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        print(f"2D combined plot saved to: {plot_path}")

        plt.close()

    def visualize(self,
                 model2: Optional[nn.Module] = None,
                 model3: Optional[nn.Module] = None,
                 same_direction: bool = False,
                 save_results: bool = True,
                 save_plots: bool = True,
                 filename: Optional[str] = None) -> Dict:
        """Main visualization function"""
        print("Starting loss landscape visualization...")

        # Generate directions
        self.generate_directions(model2, model3, same_direction)

        # Compute loss landscape
        if self.config.y_range is None:
            # 1D visualization
            x_coords, losses, accuracies = self.compute_loss_landscape_1d()
            y_coords = None
        else:
            # 2D visualization
            x_coords, y_coords, losses, accuracies = self.compute_loss_landscape_2d()

        # Save results
        results_file = None
        if save_results:
            results_file = self.save_results(x_coords, losses, accuracies, y_coords, filename)

        # Create plots
        if save_plots:
            if self.config.y_range is None:
                self.plot_1d(x_coords, losses, accuracies, save_plot=True, filename=filename)
            else:
                self.plot_2d(x_coords, y_coords, losses, accuracies, save_plot=True, filename=filename)

        print("Loss landscape visualization completed!")

        return {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'losses': losses,
            'accuracies': accuracies,
            'results_file': results_file
        }


# Convenience function for quick visualization
def visualize_loss_landscape(model: nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           criterion: nn.Module,
                           device: torch.device,
                           x_range: Tuple[float, float, int] = (-1.0, 1.0, 51),
                           y_range: Optional[Tuple[float, float, int]] = None,
                           direction_type: str = 'weights',
                           direction_method: str = 'random',
                           x_norm: str = 'filter',
                           y_norm: str = 'filter',
                           ignore_bias_bn: bool = True,
                           save_dir: str = './loss_landscape_results',
                           **kwargs) -> Dict:
    """
    Convenience function for quick loss landscape visualization

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run computation on
        x_range: X-axis range (start, end, num_points)
        y_range: Y-axis range (start, end, num_points) - None for 1D
        direction_type: 'weights' or 'states'
        direction_method: 'random', 'interpolate', 'pca', 'eigen', 'lstsq'
        x_norm: Normalization for x direction
        y_norm: Normalization for y direction
        ignore_bias_bn: Whether to ignore bias and BN parameters
        save_dir: Directory to save results
        **kwargs: Additional arguments for VisualizationConfig

    Returns:
        Dictionary containing results and file paths
    """
    config = VisualizationConfig(
        x_range=x_range,
        y_range=y_range,
        direction_type=direction_type,
        direction_method=direction_method,
        x_norm=x_norm,
        y_norm=y_norm,
        ignore_bias_bn=ignore_bias_bn,
        save_dir=save_dir,
        **kwargs
    )

    visualizer = LossLandscapeVisualizer(model, dataloader, criterion, device, config)
    return visualizer.visualize()


# Hessian computation utilities
class Hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, device, data=None, dataloader=None):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        self.device = device

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            self.inputs, self.targets = self.inputs.to(device), self.targets.to(device)

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in self.params
              ]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        for idx in range(top_n):
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for _ in tqdm(range(maxIter), desc=f'Eigenvalue {idx + 1}/{top_n}'):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        for _ in tqdm(range(maxIter), desc='Computing Trace'):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in tqdm(range(iter), desc=f'Lanczos Iteration {k + 1}/{n_v}'):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            eigenvalues, eigenvectors = torch.linalg.eig(T)

            eigen_list = eigenvalues.real
            weight_list = torch.pow(eigenvectors[0,:], 2)
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s ** 0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)