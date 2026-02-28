"""
Class for generating interventions using inverse methods
"""

import os
import json
import argparse
import uuid
from datetime import datetime
import sys
from abc import ABC, abstractmethod
from typing import List, Optional
import matplotlib.pyplot as plt
import random
from datetime import timezone
import logging
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from utils.models.command_models import BatchOpType, CommandModel
from utils.models.instruction_models import ElectricalInstructionModel
from utils.medea_models import InterventionModel
from utils.command_generator import generate_command_model
from utils.constants import (
    INVERSE_TIMEOUT_MS,
    MOMBOT_ROOT_DIR
)
from utils.vis_utils import plot_active_learning_snapshot

################################################################################


class CandidateIntervention():
    """
    Class for candidate interventions
    """

    def __init__(self,
                 intervention_model: InterventionModel,
                 id: int,
                 candidate_intervention_folder_path: str):
        self.intervention_model = intervention_model
        self.id = id
        self.candidate_intervention_folder_path = candidate_intervention_folder_path
        self.prediction_objective_values = []
        self.objective_value = None

        # Upload to candidate_interventions folder
        self.file_name = f"intervention_{self.intervention_model.intervention_id}_candidate_{self.id}.json"
        self.file_path = f"{self.candidate_intervention_folder_path}/{self.file_name}"
        with open(self.file_path, "w") as file:
            file.write(self.intervention_model.model_dump_json(indent=2))

    def set_objective_value(self, objective_value: float):
        self.objective_value = objective_value

    def append_prediction_objective_value(self, prediction_objective_value: float):
        self.prediction_objective_values.append(prediction_objective_value)

################################################################################

class ObjectiveEvaluatorManager():
    """
    Objective evaluator for straightness index
    """

    def __init__(self, evaluation_path: str, evaluator: str):
        self.evaluation_path = evaluation_path
        self.evaluator = evaluator

    def evaluate(self, prediction_video_file_path: str):
        video_file_name_stripped = prediction_video_file_path.split(
            '/')[-1][:-4]
        evaluation_file_path = f"{self.evaluation_path}/{video_file_name_stripped}_{self.evaluator}.txt"
        start = datetime.now()
        os.system(f'python3 '
                  f'objective_evaluator.py '
                  f'"{prediction_video_file_path}" '  # Video file path
                  # Behavior variable, e.g. straightness_index, e.g. velocity
                  f'"{self.evaluator}" '
                  f'"{evaluation_file_path}"')      # Output path
        end = datetime.now()
        print(f"Evaluation time taken: {end - start}")
        with open(evaluation_file_path, 'r') as file:
            content = file.read()
            return float(content.split('=')[-1].strip())


################################################################################

def make_features(X: np.ndarray, feature_config: dict = None) -> np.ndarray:
    """
    Add nonlinear features to X based on config.
    
    feature_config keys (all default True):
        - 'pre_sq': pre_val²
        - 'dur_sq': duration²
        - 'pre_cu': pre_val³
        - 'dur_cu': duration³
        - 'interaction': pre_val × duration
    """
    if feature_config is None:
        feature_config = {
            'pre_sq': True,
            'dur_sq': True,
            'pre_cu': False,
            'dur_cu': False,
            'interaction': True
        }
    
    features = [X]
    
    if feature_config.get('pre_sq', True):
        features.append(X[:, 0:1] ** 2)
    if feature_config.get('dur_sq', True):
        features.append(X[:, 1:2] ** 2)
    if feature_config.get('pre_cu', False):
        features.append(X[:, 0:1] ** 3)
    if feature_config.get('dur_cu', False):
        features.append(X[:, 1:2] ** 3)
    if feature_config.get('interaction', True):
        features.append(X[:, 0:1] * X[:, 1:2])
    
    return np.hstack(features)


class HeteroscedasticModel:
    """
    Two-stage model:
    1. Linear regression for mean prediction
    2. KNN regression on squared residuals for local variance estimation
    """
    
    def __init__(self, json_path: str, n_neighbors: int = 5, knn_weights: str = 'distance', feature_config: dict = None):
        self.mean_model = LinearRegression()
        self.variance_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=knn_weights)
        self.scaler = StandardScaler()
        self.feature_config = feature_config
        self.fitted = False

        self.load_data(json_path)
        self.fit(self.X, self.Y)

    def load_data(self, model_json: str):
        with open(model_json, 'r') as file:
            model_data = json.load(file)

        data = {
            'velocity_pre': [],
            'estim_duration': [],
            'velocity_post': []
        }

        for batch_index in model_data:
            data['velocity_pre'].append(model_data[batch_index][0])
            data['estim_duration'].append(model_data[batch_index][1])
            data['velocity_post'].append(model_data[batch_index][2])

        df = pd.DataFrame(data)
        self.X = np.array(df[['velocity_pre', 'estim_duration']])
        self.Y = np.array(df['velocity_post'])
        self.X_nonlinear = np.hstack([self.X, (self.X[:, 0:1] ** 2), (self.X[:, 1:2] ** 2), (self.X[:, 0:1] * self.X[:, 1:2])])

        logging.info("Loaded FM data: %d samples", len(self.X))
        
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit both the mean and variance models."""
        # Add nonlinear features
        X_features = make_features(X, self.feature_config)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit mean model
        self.mean_model.fit(X_scaled, Y)
        
        # Compute residuals
        Y_pred = self.mean_model.predict(X_scaled)
        residuals_squared = (Y - Y_pred) ** 2
        
        # Fit variance model on squared residuals
        n_neighbors = min(self.variance_model.n_neighbors, len(X) - 1)
        self.variance_model.set_params(n_neighbors=max(1, n_neighbors))
        
        # Use both columns for the variance model
        self.variance_scaler = StandardScaler()
        X_for_variance = self.variance_scaler.fit_transform(X)
        self.variance_model.fit(X_for_variance, residuals_squared)
        
        self.fitted = True
        
    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Predict the mean."""
        X_features = make_features(X, self.feature_config)
        X_scaled = self.scaler.transform(X_features)
        return self.mean_model.predict(X_scaled)
    
    def predict_variance(self, X: np.ndarray) -> np.ndarray:
        """Predict local variance (from squared residuals model)."""
        X_scaled = self.variance_scaler.transform(X)
        return self.variance_model.predict(X_scaled)
    
    def predict(self, X: np.ndarray):
        """
        Predict mean and variance for input X.
        
        Parameters
        ----------
        X : np.ndarray
            Shape (2,) for single point or (n, 2) for multiple points
            Columns are [pre_val, duration]
        
        Returns
        -------
        mean : float or np.ndarray
        variance : float or np.ndarray
        """
        # Handle single point input
        if X.ndim == 1:
            X = X.reshape(1, -1)
            mean = self.predict_mean(X)[0]
            variance = self.predict_variance(X)[0]
            return mean, variance
        else:
            return self.predict_mean(X), self.predict_variance(X)
        
    def compute_ucb_acquisition(
        self, 
        pre_velocity: float, 
        durations: np.ndarray,
        ucb_beta: float = 0.5,
        ucb_bandwidth: float = 20000
    ) -> tuple:
        """
        Compute UCB acquisition function for a given pre_velocity across durations.
        
        Parameters
        ----------
        pre_velocity : float
            The observed pre-stimulation velocity
        durations : np.ndarray
            Array of candidate durations to evaluate
        ucb_beta : float
            Exploration weight (higher = more exploration)
        ucb_bandwidth : float
            Bandwidth for density calculation (ms)
            
        Returns
        -------
        acquisition : np.ndarray
            UCB acquisition values for each duration
        variance_slice : np.ndarray
            Raw variance predictions
        exploration_bonus : np.ndarray
            Exploration bonus values
        best_idx : int
            Index of the best duration
        """
        # Build query points
        X_query = np.column_stack([
            np.full(len(durations), pre_velocity),
            durations
        ])
        
        # Get variance predictions
        variance_slice = self.predict_variance(X_query)
        
        # Compute exploration bonus based on density of existing training points
        density = np.array([
            np.sum(np.abs(self.X[:, 1] - d) < ucb_bandwidth) 
            for d in durations
        ])
        exploration_bonus = 1.0 / (density + 1)
        
        # Normalize both terms
        variance_normalized = variance_slice / (variance_slice.max() + 1e-8)
        exploration_normalized = exploration_bonus / (exploration_bonus.max() + 1e-8)
        
        # Combine into UCB acquisition
        acquisition = variance_normalized + ucb_beta * exploration_normalized
        
        # Find best
        best_idx = np.argmax(acquisition)
        
        return acquisition, variance_slice, exploration_bonus, best_idx
    
# =============================================================================
# INVERSE CLASS
# =============================================================================

class Inverse(ABC):
    """
    Abstract class for inverse methods
    """

    def __init__(self,
                 batch_folder_path: str,
                 pre_velocity_file_path: str,
                 timeout_seconds: int,
                 intervention_id: int,
                 ucb_beta: float = 0.5,
                 ucb_bandwidth: float = 50000):
        self.batch_id = batch_folder_path.split('-')[1]
        self.batch_folder_path = batch_folder_path
        self.pre_velocity_file_path = pre_velocity_file_path
        self.intervention_id = intervention_id
        self.padded_intervention_id = str(self.intervention_id).zfill(6)
        self.timeout_seconds = timeout_seconds
        self.evaluator = "velocity"

        self.all_interventions_folder_path = f"{self.batch_folder_path}/interventions"
        self.videos_folder_path = f"{self.batch_folder_path}/videos"
        self.intervention_folder_path = f"{self.all_interventions_folder_path}/intervention-{self.padded_intervention_id}"
        self.candidate_intervention_folder_path = f"{self.intervention_folder_path}/candidate_interventions"
        self.predictions_folder_path = f"{self.intervention_folder_path}/fm_predictions"
        self.evaluation_path = f"{self.intervention_folder_path}/evaluations"

        os.makedirs(self.intervention_folder_path, exist_ok=True)
        os.makedirs(self.candidate_intervention_folder_path, exist_ok=True)
        os.makedirs(self.predictions_folder_path, exist_ok=True)
        os.makedirs(self.evaluation_path, exist_ok=True)

        self.objective_evaluator = ObjectiveEvaluatorManager(evaluation_path=self.evaluation_path,
                                                             evaluator=self.evaluator)
        
        # Load HeteroscedasticModel from JSON
        self.ucb_beta = ucb_beta
        self.ucb_bandwidth = ucb_bandwidth
        self.predictive_model = HeteroscedasticModel(
            json_path="nonlinear_fm_cache.json",
            n_neighbors=5,
            feature_config={
                'pre_sq': True,
                'dur_sq': True,
                'pre_cu': False,
                'dur_cu': False,
                'interaction': True
            }
        )

        self.candidate_intervention_counter = 0

    def run(self):
        # Read pre_velocity from the text file written by babysitter
        logging.info("Reading pre_velocity from %s", self.pre_velocity_file_path)
        with open(self.pre_velocity_file_path, 'r') as f:
            pre_velocity = float(f.read().strip())

        logging.info("Pre-intervention velocity: %s pixels/s", pre_velocity)

        search_space = np.arange(0, 300000, 1000, dtype=float)  # duration of ESTIM in ms

        # Use UCB acquisition to select best duration
        acquisition, variance_slice, exploration_bonus, best_idx = \
            self.predictive_model.compute_ucb_acquisition(
                pre_velocity=pre_velocity,
                durations=search_space,
                ucb_beta=self.ucb_beta,
                ucb_bandwidth=self.ucb_bandwidth
            )
        
        best_estim_duration = search_space[best_idx]

        # Get predicted post_velocity for the selected duration
        best_predicted_post_v, best_variance = self.predictive_model.predict(
            np.array([pre_velocity, best_estim_duration])
        )

        logging.info(
            f"UCB selection: Duration={best_estim_duration}ms, "
            f"Predicted post_v={best_predicted_post_v:.3f}, "
            f"Variance={best_variance:.4f}, "
            f"Acquisition={acquisition[best_idx]:.4f}"
        )

        # Generate snapshot visualization
        snapshot_path = f"{self.intervention_folder_path}/active_learning_snapshot_{self.padded_intervention_id}.png"
        plot_active_learning_snapshot(
            model=self.predictive_model,
            X_train=self.predictive_model.X,
            Y_train=self.predictive_model.Y,
            pre_velocity=pre_velocity,
            selected_duration=best_estim_duration,
            predicted_post_v=best_predicted_post_v,
            output_path=snapshot_path,
            ucb_beta=self.ucb_beta,
            ucb_bandwidth=self.ucb_bandwidth,
            iteration=self.intervention_id
        )

        # Create the intervention
        best_intervention = self.make_candidate_intervention(estim_duration=int(best_estim_duration), pre_velocity=pre_velocity)


        # # Exhaustively search the space of possible durations
        # inputs = np.array([[pre_velocity, duration] for duration in search_space])
        # post_velocities = []
        # predictive_variances = []

        # for model_input in inputs:
        #     post_v_prediction, predictive_variance = self.predictive_model.predict(model_input)
        #     post_velocities.append(post_v_prediction)
        #     predictive_variances.append(predictive_variance)

        #     logging.info(f"Input: {model_input}, Predicted post_v: {post_v_prediction:0.2f}, Predictive Variance: {predictive_variance}")

        # # Select the best candidate intervention
        # best_estim_duration = inputs[np.argmax(predictive_variances)][1] # duration with highest uncertainty
        # best_predicted_post_v = post_velocities[np.argmax(predictive_variances)]
        # logging.info(f"Best intervention found: E-Stim Duration {best_estim_duration}ms")
        # best_intervention = self.make_candidate_intervention(estim_duration=best_estim_duration)

        # Write to final intervention file
        logging.info(f"Finished searching, writing intervention file")
        final_intervention_file_path = f"{self.intervention_folder_path}/intervention-{self.padded_intervention_id}.json"
        with open(final_intervention_file_path, "w") as file:
            file.write(best_intervention.intervention_model.model_dump_json(indent=2))

    def make_candidate_intervention(self, estim_duration: int, pre_velocity: float = None) -> CandidateIntervention:
        instructions = ElectricalInstructionModel(
            current_ma=20,
            angle_degrees=270,
            frequency_hz=0,  # DC current only
            duration_ms=estim_duration,
        )

        estim_command = CommandModel(
            type=BatchOpType.Electrical,
            timestamp=datetime.now(timezone.utc),
            batch_id=self.batch_id,
            command_id=str(uuid.uuid4()),
            runs=1,
            instructions=[instructions],
        )

        # Wrap the commands into an intervention model
        intervention_model = InterventionModel(
            batch_id=self.batch_id,
            intervention_id=self.padded_intervention_id,
            pre_velocity=pre_velocity,
            commands=[estim_command],
        )

        return CandidateIntervention(intervention_model=intervention_model,
                                     id=self.candidate_intervention_counter,
                                     candidate_intervention_folder_path=self.candidate_intervention_folder_path)

    def generate_candidate_intervention(self) -> CandidateIntervention:
        """
        Generates a (random) candidate intervention
        """
        commands = []

        instructions = ElectricalInstructionModel(
            current_ma=20,
            angle_degrees=270,
            frequency_hz=0,  # DC current only
            duration_ms=random.randint(0, 360) * 1000,
        )

        estim_command = CommandModel(
            type=BatchOpType.Electrical,
            timestamp=datetime.now(timezone.utc),
            batch_id=self.batch_id,
            command_id=str(uuid.uuid4()),
            runs=1,
            instructions=[instructions],
        )

        commands.append(estim_command)

        # Wrap the commands into an intervention model
        intervention_model = InterventionModel(
            batch_id=self.batch_id,
            intervention_id=self.padded_intervention_id,
            commands=commands,
        )

        return CandidateIntervention(intervention_model=intervention_model,
                                     id=self.candidate_intervention_counter,
                                     candidate_intervention_folder_path=self.candidate_intervention_folder_path)

################################################################################

if __name__ == '__main__':
    # Babysitter spawns Inverse and then waits for Inverse to generate an intervention file.
    # So at the end, Inverse.run() should generate an intervention file in the appropriate folder.

    parser = argparse.ArgumentParser()
    parser.add_argument('batch_folder_path', type=str)
    parser.add_argument('pre_velocity_file_path', type=str)
    parser.add_argument('intervention_id', type=int)
    args = parser.parse_args()

    batch_id = args.batch_folder_path.split('-')[-1]

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] [%(module)s-{batch_id}] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{args.batch_folder_path}/logs/inverse.log')
        ],
    )
    try:
        logging.info(f"Searching for intervention {args.intervention_id}")

        inverse = Inverse(batch_folder_path=args.batch_folder_path,
                        pre_velocity_file_path=args.pre_velocity_file_path,
                        intervention_id=args.intervention_id,
                        timeout_seconds=INVERSE_TIMEOUT_MS // 1000)
        inverse.run()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        os.system("./scripts/reset_medea.sh")
        sys.exit(0)

