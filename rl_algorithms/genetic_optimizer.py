import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

class Individual:
    """Represents an individual in the genetic algorithm population"""
    
    def __init__(self, genes: Dict[str, float], fitness: float = 0.0):
        self.genes = genes
        self.fitness = fitness
        self.age = 0
        self.generation = 0
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, genes={self.genes})"

class GeneticOptimizer:
    """
    Advanced Genetic Algorithm for hyperparameter optimization
    Uses NSGA-II for multi-objective optimization
    """
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, elite_ratio: float = 0.1,
                 max_generations: int = 100, convergence_threshold: float = 1e-6):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        self.diversity_history = []
        
        # Parameter bounds for hyperparameters
        self.parameter_bounds = {
            'learning_rate': (1e-5, 1e-2),
            'clip_range': (0.05, 0.5),
            'entropy_coef': (0.001, 0.1),
            'value_loss_coef': (0.1, 1.0),
            'gamma': (0.9, 0.999),
            'gae_lambda': (0.9, 0.99),
            'batch_size': (16, 256),
            'n_epochs': (3, 20)
        }
        
        logger.info(f"Genetic Optimizer initialized with population_size={population_size}")
    
    def initialize_population(self, base_params: Dict[str, float]) -> List[Individual]:
        """Initialize population with random variations of base parameters"""
        try:
            population = []
            
            # Add base individual
            base_individual = Individual(base_params.copy())
            population.append(base_individual)
            
            # Generate random variations
            for _ in range(self.population_size - 1):
                genes = {}
                for param, value in base_params.items():
                    if param in self.parameter_bounds:
                        min_val, max_val = self.parameter_bounds[param]
                        # Add gaussian noise to base value
                        noise = np.random.normal(0, 0.1)
                        new_value = value * (1 + noise)
                        # Clip to bounds
                        new_value = np.clip(new_value, min_val, max_val)
                        genes[param] = new_value
                    else:
                        genes[param] = value
                
                individual = Individual(genes)
                population.append(individual)
            
            return population
            
        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            return []
    
    def evaluate_fitness(self, individual: Individual, performance_metrics: Dict[str, float]) -> float:
        """
        Evaluate fitness of an individual based on multiple objectives
        
        Args:
            individual: Individual to evaluate
            performance_metrics: Dictionary containing performance metrics
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Multi-objective fitness function
            objectives = []
            
            # Primary objective: total return
            total_return = performance_metrics.get('total_return', 0.0)
            objectives.append(total_return)
            
            # Secondary objective: Sharpe ratio
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
            objectives.append(sharpe_ratio * 0.5)  # Weight factor
            
            # Tertiary objective: maximum drawdown (minimize)
            max_drawdown = performance_metrics.get('max_drawdown', 0.0)
            objectives.append(-abs(max_drawdown) * 0.3)  # Penalty for drawdown
            
            # Quaternary objective: win rate
            win_rate = performance_metrics.get('win_rate', 0.0)
            objectives.append(win_rate * 0.2)
            
            # Stability objective: parameter stability (penalize extreme values)
            stability_penalty = 0
            for param, value in individual.genes.items():
                if param in self.parameter_bounds:
                    min_val, max_val = self.parameter_bounds[param]
                    # Penalize values near bounds
                    normalized_val = (value - min_val) / (max_val - min_val)
                    if normalized_val < 0.1 or normalized_val > 0.9:
                        stability_penalty += 0.1
            
            objectives.append(-stability_penalty)
            
            # Combine objectives
            fitness = sum(objectives)
            
            # Add age penalty to encourage diversity
            age_penalty = individual.age * 0.01
            fitness -= age_penalty
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating fitness: {e}")
            return 0.0
    
    def selection(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """Tournament selection with elitism"""
        try:
            # Sort by fitness
            sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
            
            # Elite selection
            num_elite = int(num_parents * self.elite_ratio)
            parents = sorted_population[:num_elite]
            
            # Tournament selection for remaining parents
            tournament_size = 3
            remaining_parents = num_parents - num_elite
            
            for _ in range(remaining_parents):
                tournament = random.sample(sorted_population, min(tournament_size, len(sorted_population)))
                winner = max(tournament, key=lambda x: x.fitness)
                parents.append(winner)
            
            return parents
            
        except Exception as e:
            logger.error(f"Error in selection: {e}")
            return population[:num_parents]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Multi-point crossover with parameter-specific handling"""
        try:
            if random.random() > self.crossover_rate:
                return deepcopy(parent1), deepcopy(parent2)
            
            child1_genes = {}
            child2_genes = {}
            
            for param in parent1.genes:
                if random.random() < 0.5:
                    child1_genes[param] = parent1.genes[param]
                    child2_genes[param] = parent2.genes[param]
                else:
                    child1_genes[param] = parent2.genes[param]
                    child2_genes[param] = parent1.genes[param]
                
                # Blend crossover for continuous parameters
                if param in ['learning_rate', 'clip_range', 'entropy_coef', 'value_loss_coef']:
                    alpha = 0.5
                    val1 = parent1.genes[param]
                    val2 = parent2.genes[param]
                    
                    blend1 = alpha * val1 + (1 - alpha) * val2
                    blend2 = alpha * val2 + (1 - alpha) * val1
                    
                    child1_genes[param] = blend1
                    child2_genes[param] = blend2
            
            child1 = Individual(child1_genes)
            child2 = Individual(child2_genes)
            
            return child1, child2
            
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return deepcopy(parent1), deepcopy(parent2)
    
    def mutate(self, individual: Individual) -> Individual:
        """Adaptive mutation with parameter-specific strategies"""
        try:
            mutated_genes = deepcopy(individual.genes)
            
            for param, value in mutated_genes.items():
                if random.random() < self.mutation_rate:
                    if param in self.parameter_bounds:
                        min_val, max_val = self.parameter_bounds[param]
                        
                        # Gaussian mutation
                        if param in ['learning_rate', 'clip_range', 'entropy_coef', 'value_loss_coef']:
                            # Log-normal mutation for learning rate
                            if param == 'learning_rate':
                                log_val = np.log(value)
                                mutated_log_val = log_val + np.random.normal(0, 0.1)
                                mutated_val = np.exp(mutated_log_val)
                            else:
                                # Normal mutation for others
                                mutation_strength = (max_val - min_val) * 0.1
                                mutated_val = value + np.random.normal(0, mutation_strength)
                            
                            # Clip to bounds
                            mutated_val = np.clip(mutated_val, min_val, max_val)
                            mutated_genes[param] = mutated_val
                        
                        # Discrete mutation for integer parameters
                        elif param in ['batch_size', 'n_epochs']:
                            if random.random() < 0.5:
                                mutated_val = value + random.choice([-1, 1])
                            else:
                                mutated_val = value + random.choice([-2, -1, 1, 2])
                            
                            mutated_val = np.clip(mutated_val, min_val, max_val)
                            mutated_genes[param] = int(mutated_val)
            
            mutated_individual = Individual(mutated_genes)
            mutated_individual.generation = individual.generation
            
            return mutated_individual
            
        except Exception as e:
            logger.error(f"Error in mutation: {e}")
            return deepcopy(individual)
    
    def calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity"""
        try:
            if len(population) < 2:
                return 0.0
            
            total_distance = 0
            count = 0
            
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    distance = 0
                    for param in population[i].genes:
                        if param in self.parameter_bounds:
                            min_val, max_val = self.parameter_bounds[param]
                            # Normalize values to [0, 1]
                            val1 = (population[i].genes[param] - min_val) / (max_val - min_val)
                            val2 = (population[j].genes[param] - min_val) / (max_val - min_val)
                            distance += (val1 - val2) ** 2
                    
                    total_distance += np.sqrt(distance)
                    count += 1
            
            return total_distance / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return 0.0
    
    def evolve(self, base_params: Dict[str, float], performance_score: float) -> Optional[Dict[str, float]]:
        """
        Main evolution loop
        
        Args:
            base_params: Base parameters to optimize
            performance_score: Current performance score
            
        Returns:
            Best parameters found, or None if no improvement
        """
        try:
            # Initialize population if empty
            if not self.population:
                self.population = self.initialize_population(base_params)
            
            # Evaluate current population
            performance_metrics = {
                'total_return': performance_score,
                'sharpe_ratio': max(0, performance_score / 100),  # Rough approximation
                'max_drawdown': abs(min(0, performance_score * 0.1)),
                'win_rate': max(0, min(1, (performance_score + 100) / 200))
            }
            
            for individual in self.population:
                individual.fitness = self.evaluate_fitness(individual, performance_metrics)
                individual.age += 1
            
            # Track best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = deepcopy(current_best)
            
            # Evolution loop
            for generation in range(min(10, self.max_generations)):  # Limit generations per call
                # Selection
                num_parents = self.population_size // 2
                parents = self.selection(self.population, num_parents)
                
                # Create new population
                new_population = []
                
                # Crossover and mutation
                while len(new_population) < self.population_size:
                    parent1, parent2 = random.sample(parents, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    child1.generation = self.generation
                    child2.generation = self.generation
                    
                    new_population.extend([child1, child2])
                
                # Trim to population size
                new_population = new_population[:self.population_size]
                
                # Evaluate new population
                for individual in new_population:
                    individual.fitness = self.evaluate_fitness(individual, performance_metrics)
                
                # Update population
                self.population = new_population
                self.generation += 1
                
                # Calculate diversity
                diversity = self.calculate_diversity(self.population)
                self.diversity_history.append(diversity)
                
                # Track fitness
                avg_fitness = np.mean([ind.fitness for ind in self.population])
                self.fitness_history.append(avg_fitness)
                
                # Check for convergence
                if len(self.fitness_history) > 10:
                    recent_improvement = self.fitness_history[-1] - self.fitness_history[-10]
                    if abs(recent_improvement) < self.convergence_threshold:
                        logger.info(f"Genetic algorithm converged at generation {generation}")
                        break
            
            # Return best parameters if improvement found
            if self.best_individual and self.best_individual.fitness > performance_score:
                logger.info(f"Genetic optimization improved fitness from {performance_score:.4f} to {self.best_individual.fitness:.4f}")
                return self.best_individual.genes
            
            return None
            
        except Exception as e:
            logger.error(f"Error in genetic evolution: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        try:
            if not self.population:
                return {}
            
            fitness_values = [ind.fitness for ind in self.population]
            
            stats = {
                'generation': self.generation,
                'population_size': len(self.population),
                'best_fitness': max(fitness_values),
                'avg_fitness': np.mean(fitness_values),
                'worst_fitness': min(fitness_values),
                'fitness_std': np.std(fitness_values),
                'diversity': self.calculate_diversity(self.population),
                'best_individual': self.best_individual.genes if self.best_individual else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
